import json

import httpx
from pydantic import BaseModel

from hollowmere.config import LLMConfig
from hollowmere.llm.client import OpenAICompatLLM


class Mood(BaseModel):
    mood: int


async def test_downgrades_on_400_and_repairs_bad_json():
    seen = []
    replies = iter(["<think>let me think</think>not json at all", '```json\n{"mood": 4}\n```'])

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        if "response_format" in body and body["response_format"]["type"] == "json_schema":
            return httpx.Response(400, json={"error": "unsupported response_format"})
        content = next(replies)
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": content}}]})

    llm = OpenAICompatLLM(LLMConfig(log_path=""), transport=httpx.MockTransport(handler))
    out = await llm.structured("sys", "user", Mood, label="t")
    assert out.mood == 4
    # first the thinking flag is dropped, then json_schema falls back to json_object
    assert "chat_template_kwargs" in seen[0] and "chat_template_kwargs" not in seen[1]
    assert seen[-1]["response_format"] == {"type": "json_object"}
    # the repair turn tells the model what was wrong
    assert "invalid" in seen[-1]["messages"][-1]["content"]
    assert llm.stats.repairs == 1 and llm.stats.calls == 2
    await llm.aclose()


async def test_cut_off_reply_asks_for_a_shorter_one():
    seen = []
    replies = iter([({"mood": 4, "pad": "x" * 50}, "length"), ({"mood": 4}, "stop")])

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        body, reason = next(replies)
        content = json.dumps(body)[:-10] if reason == "length" else json.dumps(body)
        return httpx.Response(200, json={"choices": [
            {"message": {"role": "assistant", "content": content}, "finish_reason": reason}]})

    llm = OpenAICompatLLM(LLMConfig(log_path="", max_tokens=50), transport=httpx.MockTransport(handler))
    assert (await llm.structured("sys", "user", Mood, label="t")).mood == 4
    assert "cut off after 50 tokens" in seen[-1]["messages"][-1]["content"]
    await llm.aclose()


async def test_a_prompt_too_long_for_the_context_does_not_downgrade():
    import pytest

    from hollowmere.llm.client import LLMError

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": {"code": 400, "type": "exceed_context_size_error",
                                                   "message": "request (2136 tokens) exceeds the available "
                                                              "context size (2048 tokens), try increasing it"}})

    llm = OpenAICompatLLM(LLMConfig(log_path=""), transport=httpx.MockTransport(handler))
    with pytest.raises(LLMError, match="ctx-size"):
        await llm.structured("sys", "user", Mood, label="t")
    assert llm._mode == "json_schema" and llm._send_thinking_flag  # later calls keep JSON mode
    await llm.aclose()


async def test_server_embeddings_are_batched_and_cached(monkeypatch):
    from hollowmere.memory import embed as embed_mod
    from hollowmere.memory.embed import ServerEmbedder

    batches = []

    def handler(request: httpx.Request) -> httpx.Response:
        texts = json.loads(request.content)["input"]
        batches.append(len(texts))
        return httpx.Response(200, json={"data": [{"index": i, "embedding": [1.0, float(len(t))]}
                                                  for i, t in enumerate(texts)]})

    monkeypatch.setattr(embed_mod, "EMBED_BATCH", 3)
    emb = ServerEmbedder(LLMConfig(), "e")
    emb._http = httpx.AsyncClient(base_url="http://x", transport=httpx.MockTransport(handler))
    out = await emb.embed([f"text {i}" for i in range(7)])
    assert out.shape == (7, 2) and batches == [3, 3, 1]
    again = await emb.embed(["text 2", "text 2", "new"])
    assert batches[-1] == 1 and (again[0] == out[2]).all()  # only the new text went to the server


def test_the_key_can_come_from_the_environment(monkeypatch):
    import pytest
    cfg = LLMConfig(api_key="ignored", api_key_env="HOLLOWMERE_TEST_KEY")
    monkeypatch.setenv("HOLLOWMERE_TEST_KEY", "sk-test")
    assert cfg.key == "sk-test"
    monkeypatch.delenv("HOLLOWMERE_TEST_KEY")
    with pytest.raises(SystemExit, match="HOLLOWMERE_TEST_KEY"):
        cfg.key


async def test_a_reply_without_choices_is_retried(monkeypatch):
    import hollowmere.llm.client as client_mod

    async def no_wait(_):
        return None
    monkeypatch.setattr(client_mod.asyncio, "sleep", no_wait)
    replies = iter([httpx.Response(200, text="<html>gateway hiccup</html>"),
                    httpx.Response(200, json={"error": {"message": "upstream reset"}}),
                    httpx.Response(200, json={"choices": [{"message": {"content": '{"mood": 3}'}}]})])
    llm = OpenAICompatLLM(LLMConfig(log_path=""), transport=httpx.MockTransport(lambda r: next(replies)))
    assert (await llm.structured("sys", "user", Mood, label="t")).mood == 3
    await llm.aclose()


async def test_an_embedding_hiccup_is_retried(monkeypatch):
    from hollowmere.memory import embed as embed_mod
    from hollowmere.memory.embed import ServerEmbedder

    async def no_wait(_):
        return None
    monkeypatch.setattr(embed_mod.asyncio, "sleep", no_wait)
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) < 3:
            return httpx.Response(502, text="bad gateway")
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [0.6, 0.8]}]})

    emb = ServerEmbedder(LLMConfig(), "e")
    emb._http = httpx.AsyncClient(base_url="http://x", transport=httpx.MockTransport(handler))
    assert (await emb.embed(["the well"])).shape == (1, 2) and len(calls) == 3


async def test_only_failed_calls_are_logged(tmp_path):
    log_file = tmp_path / "llm.jsonl"
    replies = iter(["not json", '{"mood": 4}', '{"mood": 5}'])

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": next(replies)}}]})

    llm = OpenAICompatLLM(LLMConfig(log_path=str(log_file)), transport=httpx.MockTransport(handler))
    await llm.structured("sys", "user", Mood, label="t")  # one bad reply, then a good one
    await llm.structured("sys", "user", Mood, label="t")  # good at once
    records = [json.loads(line) for line in log_file.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 1
    assert records[0]["reply"] == "not json" and "invalid reply" in records[0]["error"]
    assert records[0]["prompt"] == "user"
    await llm.aclose()
