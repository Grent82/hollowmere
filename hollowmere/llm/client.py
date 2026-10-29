"""Async client for an OpenAI-compatible chat server (llama.cpp's llama-server, vLLM, ...).

Every cognitive call goes through `LLM.structured()`, which returns a validated
Pydantic object. The original project parsed free text with string splitting and
fell back to hard-coded "failsafe" answers; here the server is asked for JSON
matching a schema, the reply is validated, and validation errors are fed back to
the model for a bounded number of repair attempts.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Protocol, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from hollowmere.config import LLMConfig

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)

_MODES = ["json_schema", "json_object", "prompt"]


class LLMError(RuntimeError):
    pass


@dataclass
class LLMStats:
    calls: int = 0
    failures: int = 0
    repairs: int = 0
    in_flight: int = 0
    total_latency: float = 0.0
    by_label: dict[str, int] = field(default_factory=dict)
    repairs_by_label: dict[str, int] = field(default_factory=dict)
    failures_by_label: dict[str, int] = field(default_factory=dict)

    def count(self, table: dict[str, int], label: str) -> None:
        table[label] = table.get(label, 0) + 1

    def as_dict(self) -> dict:
        avg = self.total_latency / self.calls if self.calls else 0.0
        return {
            "calls": self.calls,
            "failures": self.failures,
            "repairs": self.repairs,
            "in_flight": self.in_flight,
            "avg_latency": round(avg, 2),
        }


class LLM(Protocol):
    stats: LLMStats

    async def structured(
        self, system: str, user: str, schema: type[T], *, label: str, temperature: float | None = None
    ) -> T: ...

    async def aclose(self) -> None: ...


def extract_json(text: str) -> object:
    """Pull the first JSON value out of a model reply (tolerates <think>, fences, chatter)."""
    text = _THINK_RE.sub("", text)
    # An unterminated <think> block (truncated reply) leaves nothing usable before it.
    if "<think>" in text.lower():
        text = text[text.lower().rindex("</think>") + 8 :] if "</think>" in text.lower() else ""
    text = _FENCE_RE.sub("", text.strip())
    starts = [i for i in (text.find("{"), text.find("[")) if i >= 0]
    if not starts:
        raise ValueError("no JSON found in reply")
    decoder = json.JSONDecoder()
    value, _ = decoder.raw_decode(text[min(starts) :])
    return value


def schema_for(model: type[BaseModel]) -> dict:
    return model.model_json_schema()


class OpenAICompatLLM:
    def __init__(self, cfg: LLMConfig, transport: httpx.AsyncBaseTransport | None = None):
        self.cfg = cfg
        self.stats = LLMStats()
        self._mode = cfg.structured_output
        self._send_thinking_flag = cfg.disable_thinking
        self._sem = asyncio.Semaphore(max(1, cfg.max_concurrency))
        self._http = httpx.AsyncClient(
            base_url=cfg.base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {cfg.key}"},
            timeout=httpx.Timeout(cfg.timeout_s, connect=10),
            transport=transport,
        )
        self._log_path = Path(cfg.log_path) if cfg.log_path else None
        if self._log_path:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)

    async def aclose(self) -> None:
        await self._http.aclose()

    # ------------------------------------------------------------------ public

    async def structured(
        self, system: str, user: str, schema: type[T], *, label: str, temperature: float | None = None
    ) -> T:
        json_schema = schema_for(schema)
        system = (
            f"{system}\n\nReply with a single JSON object that matches this JSON Schema. "
            f"No prose, no markdown.\n{json.dumps(json_schema, separators=(',', ':'))}"
        )
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        last_error = ""
        for attempt in range(3):
            content, truncated = await self._chat(messages, schema.__name__, json_schema, label, temperature)
            try:
                return schema.model_validate(extract_json(content))
            except (ValueError, ValidationError) as e:
                last_error = str(e)[:600]
                self.stats.repairs += 1
                self.stats.count(self.stats.repairs_by_label, label)
                log.debug("invalid reply for %s (attempt %d): %s", label, attempt + 1, last_error)
                self._log(label, f"invalid reply (attempt {attempt + 1}): {last_error}", messages, content)
                messages = messages[:2] + [
                    {"role": "assistant", "content": content[:2000]},
                    {
                        "role": "user",
                        "content": (
                            f"That reply was cut off after {self.cfg.max_tokens} tokens. Answer again, "
                            "much more briefly, with only the JSON object."
                            if truncated else
                            f"That reply was invalid: {last_error}\n"
                            "Answer again with only the corrected JSON object."
                        ),
                    },
                ]
        self.stats.failures += 1
        self.stats.count(self.stats.failures_by_label, label)
        raise LLMError(f"{label}: no valid reply after 3 attempts ({last_error})")

    # ----------------------------------------------------------------- private

    def _payload(self, messages: list[dict], name: str, json_schema: dict, temperature: float | None) -> dict:
        payload: dict = {
            "model": self.cfg.model,
            "messages": messages,
            "temperature": self.cfg.temperature if temperature is None else temperature,
            "max_tokens": self.cfg.max_tokens,
        }
        if self._mode == "json_schema":
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": name, "schema": json_schema},
            }
        elif self._mode == "json_object":
            payload["response_format"] = {"type": "json_object"}
        if self._send_thinking_flag:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        payload.update(self.cfg.extra_body)
        return payload

    def _downgrade(self, error_text: str) -> bool:
        """After a 4xx, drop the optional features one at a time. Returns False if nothing is left to drop."""
        if self._send_thinking_flag:
            self._send_thinking_flag = False
        elif self._mode != "prompt":
            self._mode = _MODES[_MODES.index(self._mode) + 1]
        else:
            return False
        log.warning("server rejected request (%s); now using mode=%s thinking_flag=%s",
                    error_text[:200], self._mode, self._send_thinking_flag)
        return True

    async def _chat(
        self, messages: list[dict], name: str, json_schema: dict, label: str, temperature: float | None
    ) -> tuple[str, bool]:
        """Returns the reply text and whether max_tokens cut it off."""
        async with self._sem:
            self.stats.in_flight += 1
            try:
                for net_attempt in range(4):
                    payload = self._payload(messages, name, json_schema, temperature)
                    started = time.perf_counter()
                    try:
                        resp = await self._http.post("/chat/completions", json=payload)
                    except httpx.HTTPError as e:
                        self._log(label, f"cannot reach server: {e!r}", messages)
                        if net_attempt == 3:
                            raise LLMError(f"{label}: cannot reach LLM server: {e!r}") from e
                        await asyncio.sleep(2 * (net_attempt + 1))
                        continue
                    if resp.status_code >= 400:
                        self._log(label, f"HTTP {resp.status_code}", messages, resp.text)
                    if 400 <= resp.status_code < 500 and resp.status_code != 429:
                        if "exceed_context_size" in resp.text or "context size" in resp.text:
                            # Not a rejected feature: dropping JSON mode would only break every later call.
                            raise LLMError(f"{label}: prompt too long for the server's context; start "
                                           f"llama-server with a larger --ctx-size or fewer --parallel slots "
                                           f"({resp.text[:200]})")
                        if self._downgrade(resp.text):
                            continue
                        raise LLMError(f"{label}: HTTP {resp.status_code}: {resp.text[:300]}")
                    if resp.status_code >= 400:
                        await asyncio.sleep(2 * (net_attempt + 1))
                        continue
                    try:
                        choice = resp.json()["choices"][0]
                        content = choice["message"].get("content") or ""
                    except (ValueError, KeyError, IndexError, TypeError, AttributeError):
                        # A proxy can answer 200 with an error page or {"error": ...}: retry like a 5xx.
                        log.warning("%s: unusable reply from server: %s", label, resp.text[:200])
                        self._log(label, "unusable reply from server", messages, resp.text)
                        if net_attempt == 3:
                            raise LLMError(f"{label}: unusable reply from server: {resp.text[:200]}") from None
                        await asyncio.sleep(2 * (net_attempt + 1))
                        continue
                    elapsed = time.perf_counter() - started
                    self.stats.calls += 1
                    self.stats.total_latency += elapsed
                    self.stats.count(self.stats.by_label, label)
                    return content, choice.get("finish_reason") == "length"
                raise LLMError(f"{label}: server kept failing")
            finally:
                self.stats.in_flight -= 1

    def _log(self, label: str, error: str, messages: list[dict], reply: str = "") -> None:
        """Only failures are logged: successful calls are thousands a day and say nothing."""
        if not self._log_path:
            return
        record = {"t": datetime.now().astimezone().isoformat(timespec="seconds"),"label": label, "error": error,
                  "prompt": messages[1]["content"], "reply": reply[:4000]}
        with self._log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
