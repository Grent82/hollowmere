"""Typed configuration loaded from config.toml."""

from __future__ import annotations

import os
import tomllib
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    base_url: str = "http://127.0.0.1:8080/v1"
    api_key: str = "local"
    api_key_env: str = ""  # name of an environment variable holding the key; wins over api_key when set
    model: str = "qwen3.8"
    max_concurrency: int = 2
    temperature: float = 0.7
    max_tokens: int = 1500
    timeout_s: float = 240
    structured_output: Literal["json_schema", "json_object", "prompt"] = "json_schema"
    disable_thinking: bool = True
    log_path: str = "saves/llm_log.jsonl"

    @property
    def key(self) -> str:
        if not self.api_key_env:
            return self.api_key
        value = os.environ.get(self.api_key_env)
        if not value:
            raise SystemExit(f"set the environment variable {self.api_key_env} to the API key "
                             f"(PowerShell: $env:{self.api_key_env} = \"sk-...\")")
        return value


class EmbeddingConfig(BaseModel):
    provider: Literal["auto", "server", "hashing"] = "auto"
    model: str = "embedding"
    hashing_dim: int = 768


class SimConfig(BaseModel):
    start: datetime = datetime(1347, 11, 2, 5, 0)
    tick_seconds: int = 60
    tick_interval: float = 0.6
    save_path: str = "saves/hollowmere.db"
    agents: list[str] = Field(default_factory=list)
    reflection_threshold: int = 150
    vision_radius: int = 6
    town_review_days: int = 7


class ServerConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8000


class Config(BaseModel):
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embeddings: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    sim: SimConfig = Field(default_factory=SimConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)


def load_config(path: str | Path = "config.toml") -> Config:
    path = Path(path)
    if not path.exists():
        return Config()
    with path.open("rb") as f:
        return Config.model_validate(tomllib.load(f))
