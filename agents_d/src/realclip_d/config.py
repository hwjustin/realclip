"""D owns its configuration; no automatic parent-directory discovery."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import Field, model_validator

from .contracts import Contract


class Settings(Contract):
    provider: Literal["openai", "openai_compatible"] = "openai"
    model: str = Field(default="gpt-6-luna", min_length=1)
    base_url: str | None = None
    api_key_env: str | None = None
    local_context_margin_tokens: int = Field(default=1024, ge=0, le=16384)
    local_max_request_bytes: int = Field(default=128_000_000, ge=1000, le=512_000_000)
    local_extra_body: dict = Field(default_factory=dict)
    reasoning_effort: Literal["none", "low", "medium", "high", "xhigh", "max"] = "high"
    research_max_turns: int = Field(default=8, ge=1, le=50)
    observation_max_turns: int = Field(default=8, ge=1, le=50)
    observation_max_crops: int = Field(default=12, ge=0, le=100)
    observation_summary_channels: list[str] = Field(
        default_factory=lambda: ["DNA", "RNA", "ER", "Mito", "AGP"], min_length=1)
    observation_crop_max_pixels: int = Field(default=4_000_000, ge=1, le=16_000_000)
    max_output_tokens: int = Field(default=4096, ge=256, le=32000)
    # Live requests use UTF-8 bytes as a conservative text-token bound. Permit
    # larger explicit reservations without discarding any research history.
    max_input_tokens: int = Field(default=200000, ge=1000, le=1000000)
    # Accepted only to read historical configurations; no memory/context switching.
    research_context_bytes: int = Field(default=120000, ge=8000, le=180000)
    max_tool_calls: int = Field(default=24, ge=1, le=100)
    web_search: bool = True
    web_max_calls: int = Field(default=3, ge=0, le=10)
    pubchem_requests_per_second: float = Field(default=2, gt=0, le=5)
    # Conservative configurable reservations; these are not billing statements.
    input_usd_per_million: float = Field(default=0.125, ge=0)
    output_usd_per_million: float = Field(default=0.5, ge=0)
    web_search_usd_per_call: float = Field(default=0.01, ge=0.01)
    budget_usd: float | None = Field(default=None, gt=0)
    request_timeout_seconds: float = Field(default=300, gt=0, le=600)

    @model_validator(mode="after")
    def web_calls_fit(self):
        if self.provider == 'openai':
            if self.model != 'gpt-6-luna' or self.base_url is not None:
                raise ValueError('The original OpenAI profile requires gpt-6-luna and the official endpoint; choose openai_compatible for a local model')
            if self.input_usd_per_million < .125 or self.output_usd_per_million < .5:
                raise ValueError('OpenAI accounting rates must retain their conservative minimums')
        else:
            url = urlsplit(self.base_url or '')
            if url.scheme not in {'http', 'https'} or not url.hostname or url.username or url.password or url.query or url.fragment:
                raise ValueError('openai_compatible requires an explicit http(s) base_url without embedded credentials or query parameters')
            if self.web_search:
                raise ValueError('Hosted OpenAI web search is unavailable locally; set web_search=false. Official database tools remain available.')
        if self.web_search and self.web_max_calls == 0:
            raise ValueError("Enabled web search requires a positive web_max_calls")
        if (any(not channel.strip() for channel in self.observation_summary_channels)
                or len(set(self.observation_summary_channels)) != len(self.observation_summary_channels)):
            raise ValueError("Observation summary channels must be nonempty and unique")
        return self


def load_settings(path: Path | None) -> Settings:
    return Settings.model_validate(json.loads(path.read_text())) if path else Settings()
