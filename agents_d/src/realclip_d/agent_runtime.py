"""Two isolated SDK runs, offline fixtures, and explicitly budgeted live calls.

Mock uses the real Agents SDK Runner and its function-tool execution loop. Only
the model and public-knowledge data are synthetic; it makes no network requests.
"""
from __future__ import annotations

import asyncio
import base64
from dataclasses import asdict, is_dataclass, replace
from datetime import datetime, timezone
import hashlib
import inspect
import json
import mimetypes
import os
from pathlib import Path
import time
from typing import Any, AsyncIterator, Literal
import uuid

from agents import Agent, AgentOutputSchema, Model, ModelResponse, ModelSettings, OpenAIResponsesModel
from agents import RunConfig, Runner, WebSearchTool, function_tool
from agents.usage import Usage
from agents.model_settings import ModelRetrySettings
from openai import AsyncOpenAI, Omit
from openai.types.responses import ResponseFunctionToolCall, ResponseOutputMessage
from openai.types.responses import ResponseOutputText
from openai.types.shared import Reasoning

from .config import Settings
from .contracts import ResearchResultV4
from .contracts import (
    ObservationInput, ResearchResultV3, SampleInput, Source,
    ToolResult,
)
from .tool_views import build_model_view, research_view
from .tools.article_reading import ArticleMode, DEFAULT_MAX_CHARS
from .language import validate_agent_input, validate_agent_output, validate_model_request
from .observation_contracts import (ObservationRequest, ObservationOutput, ObservationResult,
                                    assemble_result, validate_observation_references)
from .observation_protocol import check_observation_text
from .inputs import image_manifest

Mode = Literal["mock", "live"]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_jsonable(value), indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _append(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(_jsonable(value), ensure_ascii=False) + "\n")


def sanitize_request(value: Any) -> Any:
    """Record text and image hashes, never duplicate image bytes in call logs."""
    value = _jsonable(value)
    if isinstance(value, dict):
        return {key: sanitize_request(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_request(item) for item in value]
    if isinstance(value, str) and value.startswith("data:image/"):
        return {"image_sha256": hashlib.sha256(value.encode()).hexdigest(),
                "encoded_bytes": len(value)}
    return value


class BudgetError(RuntimeError):
    """A reservation could not be made; no model request was sent."""


class UnknownUsageError(RuntimeError):
    """A model request may have been billed; explicit reconciliation is required."""


class BudgetLedger:
    """Persist conservative reservations shared by the two sequential stages.

    Usage reports and configured prices are estimates, not billing statements.
    A failed request remains reserved and blocks every later live request.
    """

    def __init__(self, path: Path, budget_usd: float | None, settings: Settings):
        self.path, self.settings = path, settings
        if path.exists():
            self.state = json.loads(path.read_text())
            if budget_usd != self.state["budget_usd"]:
                raise BudgetError("Saved budget differs; use a new run directory")
        else:
            self.state = {"budget_usd": budget_usd, "spent_estimate_usd": 0.0,
                          "blocked": False, "requests": []}
        if any(row["status"] != "complete" for row in self.state["requests"]):
            self.state["blocked"] = True

    def reserve(self, stage: str, web_enabled: bool) -> str:
        if self.state["blocked"]:
            raise UnknownUsageError("A prior request has unknown usage; ledger is blocked")
        budget = self.state["budget_usd"]
        if budget is None or budget <= 0:
            raise BudgetError("Live mode requires an explicit positive USD budget")
        s = self.settings
        reservation = (s.max_input_tokens * s.input_usd_per_million / 1_000_000
                       + s.max_output_tokens * s.output_usd_per_million / 1_000_000
                       + (s.web_max_calls * s.web_search_usd_per_call if web_enabled else 0))
        reserved = sum(r["reserved_usd"] for r in self.state["requests"]
                       if r["status"] != "complete")
        if self.state["spent_estimate_usd"] + reserved + reservation > budget:
            raise BudgetError(f"Budget cannot reserve ${reservation:.6f} for the next request")
        request_id = uuid.uuid4().hex
        self.state["requests"].append({"request_id": request_id, "stage": stage,
                                      "reserved_usd": reservation, "status": "pending",
                                      "created_at": _now()})
        self._save()
        return request_id

    def complete(self, request_id: str, usage: dict[str, Any], web_calls: int) -> None:
        row = self._row(request_id)
        input_tokens = usage.get("input_tokens")
        output_tokens = usage.get("output_tokens")
        if not isinstance(input_tokens, int) or not isinstance(output_tokens, int):
            self.unknown(request_id, "Provider did not report input/output tokens")
            raise UnknownUsageError("Provider usage is missing; ledger is blocked")
        s = self.settings
        amount = (input_tokens * s.input_usd_per_million / 1_000_000
                  + output_tokens * s.output_usd_per_million / 1_000_000
                  + web_calls * s.web_search_usd_per_call)
        row.update(status="complete", usage=usage, web_calls=web_calls,
                   cost_estimate_usd=amount, completed_at=_now())
        self.state["spent_estimate_usd"] += amount
        # Detect an unexpected provider result before any subsequent request.
        if input_tokens > s.max_input_tokens or output_tokens > s.max_output_tokens:
            self.state["blocked"] = True
            row["warning"] = "Provider usage exceeded the configured reservation"
        self._save()

    def unknown(self, request_id: str, reason: str) -> None:
        self._row(request_id).update(status="unknown", error=reason, updated_at=_now())
        self.state["blocked"] = True
        self._save()

    def _row(self, request_id: str) -> dict[str, Any]:
        return next(row for row in self.state["requests"] if row["request_id"] == request_id)

    def _save(self) -> None:
        _write(self.path, self.state)


class RecordedModel(Model):
    """Record SDK model boundaries and prevent automatic paid retry/replay."""

    def __init__(self, inner: Model, settings: Settings, stage_dir: Path,
                 stage: str, ledger: BudgetLedger | None, live: bool, web_enabled: bool):
        self.inner, self.settings, self.stage_dir = inner, settings, stage_dir
        self.stage, self.ledger, self.live, self.web_enabled = stage, ledger, live, web_enabled
        self.calls = 0
        self.web_sources: list[Source] = []
        self.web_retrieved_at = _now()
        self.force_final = False

    async def get_response(self, *args: Any, **kwargs: Any) -> ModelResponse:
        self.calls += 1
        turn_limit = (self.settings.research_max_turns if self.stage == "research"
                      else self.settings.observation_max_turns)
        if ((self.force_final or self.calls >= turn_limit)
                and kwargs.get("model_settings") is not None):
            kwargs["model_settings"] = replace(kwargs["model_settings"], tool_choice="none")
        request = {"system_instructions": kwargs.get("system_instructions", args[0] if args else None),
                   "input": kwargs.get("input", args[1] if len(args) > 1 else None),
                   "model": self.settings.model,
                   "model_settings": kwargs.get("model_settings"),
                   "tools": [{"name": getattr(t, "name", type(t).__name__),
                              "description": getattr(t, "description", None),
                              "parameters": getattr(t, "params_json_schema", None)}
                             for t in kwargs.get("tools", [])],
                   "output_schema": kwargs["output_schema"].json_schema() if kwargs.get("output_schema") else None,
                   "previous_response_id": kwargs.get("previous_response_id"),
                   "conversation_id": kwargs.get("conversation_id")}
        validate_model_request(_jsonable(request))
        if request["previous_response_id"] or request["conversation_id"]:
            raise ValueError("D requires local independent conversations, not stored continuation IDs")
        request_id = None
        if self.live:
            if self.settings.provider == 'openai_compatible':
                from .observation_transport import current_transport
                if current_transport() is not None:
                    raise BudgetError('Local models use inline images; disable the OpenAI file-upload transport')
                size = await self.inner.count_input(args, kwargs,
                    self.stage_dir / 'input_token_counts' / f'{self.calls:03d}.json')
            elif self.stage == "observation":
                from .observation_transport import current_transport
                transport = current_transport()
                if transport is not None:
                    if self.ledger is None or self.ledger.state["blocked"]:
                        raise UnknownUsageError("Observation ledger is unavailable or blocked")
                    # Keep local source-image receipts above unchanged. Only the
                    # actual count/generation inputs use uploaded byte references.
                    if "input" in kwargs:
                        kwargs = {**kwargs, "input": await transport.prepare_input(kwargs["input"])}
                    elif len(args) > 1:
                        args = (args[0], await transport.prepare_input(args[1]), *args[2:])
                size = await self._count_observation_input(args, kwargs)
            else:
                # Research is text only; retain its conservative local bound.
                size = len(json.dumps(_jsonable(request), ensure_ascii=False).encode()) + 16384
            if size > self.settings.max_input_tokens:
                raise BudgetError("Input token bound exceeds max_input_tokens; resize inputs or configure the bound")
            assert self.ledger is not None
            request_id = self.ledger.reserve(self.stage, self.web_enabled)
        number = f"{self.calls:03d}"
        _write(self.stage_dir / "requests" / f"{number}.json", sanitize_request(request))
        try:
            response = await self.inner.get_response(*args, **kwargs)
            payload = _jsonable(response)
            _write(self.stage_dir / "responses" / f"{number}.json", payload)
            self._capture_web_sources(payload, f"responses/{number}.json")
            usage = getattr(response, "raw_usage", None)
            if self.live:
                if usage is None:
                    raise UnknownUsageError("Provider returned no raw usage")
                web_calls = sum(item.get("type") == "web_search_call" for item in payload.get("output", []))
                assert request_id is not None and self.ledger is not None
                self.ledger.complete(request_id, _jsonable(usage), web_calls)
            _append(self.stage_dir / "calls.jsonl", {"call": self.calls, "request_id": request_id,
                    "status": "complete", "usage": _jsonable(usage or response.usage), "mode": "live" if self.live else "mock"})
            return response
        except BaseException as exc:
            if request_id and self.ledger:
                self.ledger.unknown(request_id, f"{type(exc).__name__}: {exc}")
            _append(self.stage_dir / "calls.jsonl", {"call": self.calls, "request_id": request_id,
                    "status": "unknown" if self.live else "failed", "error": str(exc)})
            raise

    async def _count_observation_input(self, args: tuple, kwargs: dict) -> int:
        """Count the converted vision request before reserving/sending generation.

        Use the installed SDK's converter for both requests, including tool and
        output schemas and the full conversation. No image-byte/token heuristic,
        truncation, or fallback on a failed count. Mock runs never enter here.
        """
        if not isinstance(self.inner, OpenAIResponsesModel):
            raise BudgetError("Live observation requires the Responses token-count adapter")
        if self.ledger is None or self.ledger.state["blocked"]:
            raise UnknownUsageError("Observation ledger is unavailable or blocked")
        if not self.ledger.state["budget_usd"]:
            raise BudgetError("Live observation requires an explicit positive budget")
        bound = inspect.signature(self.inner.get_response).bind(*args, **kwargs)
        bound.apply_defaults()
        arguments = dict(bound.arguments)
        arguments.pop("tracing", None)
        create = self.inner._build_response_create_kwargs(**arguments)
        counter = self.inner._get_client().responses.input_tokens.count
        accepted = set(inspect.signature(counter).parameters)
        payload = {key: value for key, value in create.items()
                   if key in accepted and not isinstance(value, Omit) and value is not None}
        payload["truncation"] = "disabled"
        encoded = json.dumps(_jsonable(payload), ensure_ascii=False).encode()
        if len(encoded) > 512_000_000:
            raise BudgetError("Observation payload exceeds the 512 MB request limit")
        receipt = {"model": self.settings.model,
                   "count_payload_sha256": hashlib.sha256(encoded).hexdigest(),
                   "encoded_bytes": len(encoded), "phase": "before_generation",
                   "safety_margin_tokens": 16384}
        path = self.stage_dir / "input_token_counts" / f"{self.calls:03d}.json"
        began = time.monotonic()
        try:
            from .observation_transport import current_transport
            transport = current_transport()
            if transport is None:
                result = await counter(**payload)
                count, cache_hit = result.input_tokens, False
            else:
                count, cache_hit = await transport.count(payload, counter)
            if type(count) is not int or count < 0:
                raise BudgetError("Token counter did not return a valid input count")
            receipt.update(status="counted", input_tokens=count,
                           input_token_bound=count + 16384, cache_hit=cache_hit,
                           duration_seconds=time.monotonic() - began)
            _write(path, receipt)
            return count + 16384
        except BaseException as exc:
            receipt.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            _write(path, receipt)
            raise

    def _capture_web_sources(self, payload: Any, cache_ref: str | None = None) -> None:
        def walk(value: Any, in_web_call: bool = False) -> None:
            if isinstance(value, dict):
                in_web_call = in_web_call or value.get("type") == "web_search_call"
                if (in_web_call or value.get("type") == "url_citation") and value.get("url"):
                    url = value["url"]
                    source = Source(source_id="web_" + hashlib.sha256(url.encode()).hexdigest()[:16],
                                    url=url, retrieved_at=self.web_retrieved_at,
                                    kind="web_search", title=value.get("title"), cache_ref=cache_ref)
                    if source.source_id not in {s.source_id for s in self.web_sources}:
                        self.web_sources.append(source)
                for item in value.values():
                    walk(item, in_web_call)
            elif isinstance(value, list):
                for item in value:
                    walk(item, in_web_call)
        walk(payload)

    def stream_response(self, *args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        raise NotImplementedError("D uses recorded non-streaming calls only")


class _ToolRecorder:
    def __init__(self, stage_dir: Path, max_calls: int, research: bool = False):
        self.stage_dir, self.max_calls, self.calls = stage_dir, max_calls, 0
        self.research = research
        self.article_coverage: dict[str, list[tuple[int, int]]] = {}
        self.sources: dict[str, Source] = {}
        self.cp_ids: set[str] = set()

    async def call(self, name: str, function: Any, *args: Any, **kwargs: Any) -> str:
        self.calls += 1
        if self.calls > self.max_calls:
            result = ToolResult(status="error", message="Run tool-call limit reached")
        else:
            # The workflow is sequential and clients are synchronous. Execute
            # directly so offline runs need no executor threads; live HTTP has
            # its own bounded timeout and retry policy.
            result = function(*args, **kwargs)
        normalized = []
        for source in result.sources:
            if source.cache_ref:
                try:
                    relative = Path(source.cache_ref).resolve().relative_to(self.stage_dir.resolve())
                    source = source.model_copy(update={"cache_ref": str(relative)})
                except ValueError:
                    pass
            normalized.append(source)
        result = result.model_copy(update={"sources": normalized})
        for source in result.sources:
            self.sources[source.source_id] = source
        if name == "query_cp":
            for row in result.data.get("rows", []):
                self.cp_ids.add(row["feature_id"])
        model_view = build_model_view(name, result)
        if name == "read_article" and model_view.status == "ok":
            self._article_exposure(model_view)
        if self.research:
            model_view = research_view(name, result, model_view if name == "read_article" else None)
        _append(self.stage_dir / "tools.jsonl", {"tool": name, "arguments": _jsonable(args),
                    "keyword_arguments": _jsonable(kwargs), "result": result.model_dump(mode="json"),
                    "model_result": model_view.model_dump(mode="json")})
        _write(self.stage_dir / "sources.json", list(self.sources.values()))
        _write(self.stage_dir / "evidence.json", {"cp_feature_ids": sorted(self.cp_ids)})
        return model_view.model_dump_json()

    def _article_exposure(self, view: ToolResult) -> None:
        """Track full-text pages actually returned by this tool, never inferred reading."""
        reading = view.data.get("reading", {})
        metadata = view.data["_model_view"]
        digest = metadata.get("full_text_content_sha256")
        if not digest:
            return
        key = ":".join(sorted(source.source_id for source in view.sources
                              if source.url.rstrip("/").endswith("fullTextXML"))) + ":" + digest
        intervals = self.article_coverage.setdefault(key, [])
        if reading and reading.get("mode") == "full_text" and reading.get("text"):
            intervals.append((reading["offset"], reading["end"]))
        merged: list[tuple[int, int]] = []
        for start, end in sorted(intervals):
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        self.article_coverage[key] = merged
        complete = bool(metadata["full_text_total_characters"]) and merged == [(0, metadata["full_text_total_characters"])]
        view.data["_model_view"].update(
            full_text_characters_provided=sum(end - start for start, end in merged),
            full_text_complete=complete and not metadata.get("language_projection"),
            **({"English_presentation_complete": complete} if metadata.get("language_projection") else {}),
            full_text_coverage_note="Coverage of full_text-mode pages returned in this run; does not establish model comprehension.",
        )


class MockKnowledge:
    """Deliberately synthetic knowledge fixtures; never scientific evidence."""
    def __init__(self):
        self.source = Source(source_id="mock_knowledge", url="https://example.invalid/d-synthetic-fixture",
                             retrieved_at="2000-01-01T00:00:00+00:00", kind="mock",
                             title="Synthetic D plumbing fixture; not biological evidence")

    def resolve(self, identity: Any) -> ToolResult:
        return ToolResult(status="ok", sources=[self.source], data={"synthetic": True,
                          "identity": _jsonable(identity), "note": "No real compound was resolved"})

    def annotations(self, cid: int, heading: str) -> ToolResult:
        return ToolResult(status="not_found", sources=[self.source], data={"synthetic": True},
                          message="Synthetic fixture has no pharmacology annotation")

    def chembl(self, **kwargs: Any) -> ToolResult:
        return ToolResult(status="not_found", sources=[self.source], data={"synthetic": True})

    def search(self, query: str, limit: int, **kwargs: Any) -> ToolResult:
        return ToolResult(status="not_found", sources=[self.source], data={"synthetic": True, "articles": []})

    def read(self, article_id: str, source: str, **kwargs: Any) -> ToolResult:
        return ToolResult(status="unavailable", sources=[self.source], data={"synthetic": True,
                          "access": "unavailable"}, message="Synthetic fixture has no article")


class ScriptedModel(Model):
    """A deterministic model driving real SDK tool calls before final output."""
    def __init__(self, steps: list[list[tuple[str, dict[str, Any]]]], final: dict[str, Any]):
        self.steps, self.final, self.calls = steps, final, 0

    async def get_response(self, *args: Any, **kwargs: Any) -> ModelResponse:
        index, self.calls = self.calls, self.calls + 1
        if index < len(self.steps):
            output = [ResponseFunctionToolCall(id=f"fc_{index}_{i}", call_id=f"call_{index}_{i}",
                      type="function_call", name=name, arguments=json.dumps(arguments), status="completed")
                      for i, (name, arguments) in enumerate(self.steps[index])]
        else:
            output = [ResponseOutputMessage(id=f"msg_{index}", type="message", role="assistant",
                      status="completed", content=[ResponseOutputText(type="output_text",
                      text=json.dumps(self.final), annotations=[])])]
        return ModelResponse(output=output, usage=Usage(requests=1, input_tokens=0,
                             output_tokens=0, total_tokens=0), response_id=f"mock_{index}")

    def stream_response(self, *args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        raise NotImplementedError("Scripted model is non-streaming")


def _prompt(name: str, version: str = "v3") -> str:
    """Select the version explicitly; retain V1 prompts for old run artifacts."""
    return (Path(__file__).parent / "prompts" / f"{name}_{version}.md").read_text()


def _research_messages(payload: dict[str, Any], output_format: str | None = None) -> list[dict[str, Any]]:
    """Keep serialization instructions separate from research and sample context."""
    messages = []
    if output_format is not None:
        messages.append({"role": "developer", "content": output_format})
    messages.append({"role": "user", "content": json.dumps(payload, ensure_ascii=False)})
    return messages


def _model_settings(settings: Settings, web: bool) -> ModelSettings:
    if settings.provider == 'openai_compatible':
        return ModelSettings(max_tokens=settings.max_output_tokens, parallel_tool_calls=False,
            preserve_raw_usage=True, retry=ModelRetrySettings(max_retries=0),
            extra_body=settings.local_extra_body or None)
    return ModelSettings(reasoning=Reasoning(effort=settings.reasoning_effort),
                         max_tokens=settings.max_output_tokens, store=False,
                         parallel_tool_calls=False, preserve_raw_usage=True,
                         retry=ModelRetrySettings(max_retries=0),
                         response_include=["web_search_call.action.sources"] if web else None,
                         extra_args={"service_tier": "default", **({"max_tool_calls": settings.web_max_calls} if web else {})})


class _DAsyncOpenAI(AsyncOpenAI):
    """Prime SDK platform metadata without an executor in this local runtime.

    Pinned OpenAI 3.35.1 otherwise probes the OS through an executor on the
    first request of each clone. Setting its metadata cache synchronously also
    covers clients cloned by Agents SDK. This makes no HTTP request.
    """
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        from openai._base_client import get_platform
        self._platform = get_platform()


def _live_model(settings: Settings) -> tuple[Model, AsyncOpenAI]:
    if settings.provider == 'openai_compatible':
        from .local_provider import LocalChatModel
        key = os.environ.get(settings.api_key_env or 'LOCAL_MODEL_API_KEY', 'not-required')
        client = _DAsyncOpenAI(base_url=settings.base_url, api_key=key,
                               max_retries=0, timeout=settings.request_timeout_seconds)
        return LocalChatModel(model=settings.model, openai_client=client, settings=settings), client
    if not os.environ.get(settings.api_key_env or "OPENAI_API_KEY"):
        raise ValueError("Live mode requires OPENAI_API_KEY")
    if settings.budget_usd is None:
        raise BudgetError("Live mode requires an explicit budget_usd")
    client = _DAsyncOpenAI(api_key=os.environ[settings.api_key_env or "OPENAI_API_KEY"],
                           base_url="https://api.openai.com/v1", max_retries=0, timeout=settings.request_timeout_seconds)
    return OpenAIResponsesModel(model=settings.model, openai_client=client), client


def _research_payload(sample: SampleInput) -> dict[str, Any]:
    return {"sample_id": sample.sample_id, "compound": sample.compound.model_dump(mode="json"),
            "known_moa": sample.known_moa,
            "context": sample.context.model_dump(mode="json", exclude={"notes"} if not sample.context.notes else None)}


async def run_research(sample: SampleInput, settings: Settings, stage_dir: Path,
                       mode: Mode = "mock", budget: BudgetLedger | None = None,
                       *, pubchem_limiter=None) -> ResearchResultV4:
    validate_agent_input(_research_payload(sample))
    from .research_protocol import mock_research_v3
    from .tools import CachedHTTP, ChEMBLClient, EuropePMCClient, PubChemClient
    stage_dir.mkdir(parents=True, exist_ok=True)
    recorder = _ToolRecorder(stage_dir, settings.max_tool_calls, research=True)
    _write(stage_dir / "sources.json", [])
    _write(stage_dir / "evidence.json", {"cp_feature_ids": []})
    client = http = None
    mock = MockKnowledge()
    if mode == "live":
        inner, client = _live_model(settings)
        http = CachedHTTP(stage_dir / "cache", allow_network=True,
                          pubchem_requests_per_second=settings.pubchem_requests_per_second,
                          **({"pubchem_limiter": pubchem_limiter} if pubchem_limiter is not None else {}))
        pubchem, chembl, papers = PubChemClient(http), ChEMBLClient(http), EuropePMCClient(http)
        resolve, annotations = pubchem.resolve_compound, pubchem.get_annotations
        chembl_query, search, read = chembl.get_evidence, papers.search, papers.read_article
    elif mode == "mock":
        resolve, annotations, chembl_query, search, read = mock.resolve, mock.annotations, mock.chembl, mock.search, mock.read
        fixture = mock_research_v3(sample, [mock.source]).model_dump(mode="json")
        final = {"sample_id": sample.sample_id,
                 "compound_identification": {"summary": fixture["compound_identification"]["summary"]},
                 "compound_research_and_analysis": {
                     "findings": [{k: row[k] for k in ("finding", "reported_conditions")}
                                  for row in fixture["compound_research_and_analysis"]["findings"]],
                     "research_gaps": fixture["compound_research_and_analysis"]["research_gaps"]},
                 "morphology_hypotheses": [{k: row[k] for k in ("id", "change", "basis", "applicability")}
                                           for row in fixture["morphology_hypotheses"]],
                 "observation_guidance": fixture["observation_guidance"]}
        inner = ScriptedModel([[('resolve_compound', {})], [('search_literature', {'query': 'synthetic morphology', 'limit': 2})]], final)
    else:
        raise ValueError("Runtime mode must be mock or live; replay is handled by the workflow")

    @function_tool
    async def resolve_compound() -> str:
        """Resolve the supplied compound identity; preserve ambiguous candidates."""
        return await recorder.call("resolve_compound", resolve, sample.compound)

    @function_tool
    async def get_pubchem_annotations(cid: int, heading: str = "Pharmacology and Biochemistry") -> str:
        """Read official PubChem PUG-View annotations with provider references."""
        return await recorder.call("get_pubchem_annotations", annotations, cid, heading)

    @function_tool
    async def get_chembl_evidence(inchikey: str | None = None, chembl_id: str | None = None,
                                  max_records: int = 10) -> str:
        """Read ChEMBL exact-molecule and labeled parent mechanisms / activities.

        Scientific records and functional target annotations are retained.
        Bulky PDB/PDBe structure-index lists stay in the complete backend receipt;
        the model view reports their counts per target component.
        Large repeated functional annotations may use a shared table within this
        response: component record_indexes point into target_annotation_records.
        All values and component associations remain available without another call.
        """
        return await recorder.call("get_chembl_evidence", chembl_query, inchikey=inchikey,
                                   chembl_id=chembl_id, max_records=max_records)

    @function_tool
    async def search_literature(query: str, limit: int = 10, cursor: str | None = None,
                                result_offset: int = 0) -> str:
        """Search Europe PMC for scientific literature on a compound's properties,
        mechanisms of action, molecular targets, and cellular effects, including
        reported morphological changes. Use the supplied experimental context
        where relevant when searching for evidence that helps explain how the
        compound affects cells. Returns article records to help select relevant
        papers for further reading.

        Every retrieved candidate is listed with its full title and identifiers.
        Results contain complete abstracts; follow next_request for additional
        abstracts and next_page_request for more database hits. Read relevant
        original papers, including compound-specific mechanism/cellular studies;
        a narrow search with no hits does not establish absence of evidence.
        Source-provided English titles are preferred. Abstracts without a complete
        English counterpart are explicitly unavailable; originals stay on disk.

        Args:
            query: A Europe PMC search query about the compound, its properties,
                mechanisms, targets, or cellular effects.
            limit: The number of article records to request from Europe PMC;
                default 10, allowed range 1–100.
            cursor: Europe PMC database-page cursor from next_page_request; null
                for the first page. Keep query and limit when continuing.
            result_offset: Abstract-list offset from next_request, normally 0.
                Keep query, limit and cursor when continuing within a page.
        """
        return await recorder.call("search_literature", search, query, limit,
                                   cursor=cursor, result_offset=result_offset)

    @function_tool
    async def read_article(article_id: str, source: str = "MED", mode: ArticleMode = "overview",
                           section_ids: list[str] | None = None, offset: int = 0,
                           max_chars: int = DEFAULT_MAX_CHARS) -> str:
        """Read a paper's overview, chosen sections, or full text without keyword filtering.

        Use the abstract and complete outline to choose what to read about the
        compound's properties, mechanisms, cellular effects, and morphology.
        Consult methods for experimental conditions and expand reading as needed.
        Selected sections/full text are automatically expanded locally up to
        96,000 characters per response, without extra model or HTTP calls.
        For longer selections, follow reading.next_request until the chosen text
        is complete, or explain why further reading is unnecessary. Access and
        text exposure do not establish comprehension.
        English presentation omits explicitly translated duplicate abstracts and
        labels content unavailable in English. Offsets and hashes refer to this
        presentation; it does not imply complete access to all original text.

        Args:
            article_id: Europe PMC record ID or PMCID, obtained from search results.
            source: Europe PMC source code, normally MED; ignored for PMCID lookup.
            mode: overview returns abstract and complete outline; sections reads selected
                sections including subsections; full_text reads all available XML text.
            section_ids: IDs from the outline for sections mode; otherwise null or empty.
            offset: Starting character offset in the selected text, normally 0 or the
                value from reading.next_request. Keep mode and section_ids when continuing.
            max_chars: Initial text-page size, 1..24000; default 12000. Sections
                and full_text automatically expand up to 96000 characters;
                overview retains the requested page size.
        """
        return await recorder.call("read_article", read, article_id, source, mode=mode,
                                   section_ids=section_ids, offset=offset, max_chars=max_chars,
                                   auto_read=True)

    web = mode == "live" and settings.web_search
    model = RecordedModel(inner, settings, stage_dir, "research", budget, mode == "live", web)
    agent = Agent(name="D Research Agent", instructions=_prompt("research"), model=model,
                  model_settings=_model_settings(settings, web),
                  tools=[resolve_compound, get_pubchem_annotations, get_chembl_evidence, search_literature, read_article]
                        + ([WebSearchTool()] if web else []), output_type=ResearchResultV4)
    payload = _research_payload(sample)
    payload["web_retrieved_at"] = model.web_retrieved_at
    payload["run_limits"] = {"max_model_turns": settings.research_max_turns,
                             "max_function_calls": settings.max_tool_calls}
    _write(stage_dir / "request_input.json", payload)
    output_format = _prompt("research_output_format", "v3")
    (stage_dir / "output_format.md").write_text(output_format)
    _write(stage_dir / "output_contract.json", {"name": "ResearchResultV4", "schema_version": "4"})
    _write(stage_dir / "output_schema.json", AgentOutputSchema(ResearchResultV4).json_schema())
    try:
        result = await Runner.run(agent, _research_messages(payload, output_format),
                                  max_turns=settings.research_max_turns,
                                  run_config=RunConfig(tracing_disabled=True))
        parsed = ResearchResultV4.model_validate(result.final_output)
        validate_agent_output(parsed.model_dump(mode="json"))
        _write(stage_dir / "findings.json", parsed.compound_research_and_analysis)
        _write(stage_dir / "usage.json", result.context_wrapper.usage)
        return parsed
    finally:
        for source in model.web_sources:
            recorder.sources[source.source_id] = source
        _write(stage_dir / "sources.json", list(recorder.sources.values()))
        if client:
            await client.close()
        if http and hasattr(http, "close"):
            http.close()


def observation_messages(sample: SampleInput, view: ObservationInput | ObservationRequest) -> list[dict[str, Any]]:
    """Send actual bytes with safe labels; never leak image filenames or identity."""
    content: list[dict[str, Any]] = [{"type": "input_text", "text": view.model_dump_json()}]
    for image in sample.images:
        data = image.path.read_bytes()
        mime = mimetypes.guess_type(image.path.name)[0] or "image/png"
        label = next(row for row in view.images if row["image_id"] == image.image_id)
        if hashlib.sha256(data).hexdigest() != label["sha256"]:
            raise ValueError("Image bytes changed after observation input preparation")
        content.extend([{"type": "input_text", "text": json.dumps(label)},
                        {"type": "input_image", "image_url": f"data:{mime};base64,{base64.b64encode(data).decode()}", "detail": "high"}])
    return [{"role": "user", "content": content}]


async def run_observation(sample: SampleInput, view: ObservationRequest, settings: Settings,
                          stage_dir: Path, mode: Mode = "mock",
                          budget: BudgetLedger | None = None):
    """Run Observation's four-module reviewed report before final summarization."""
    from .observation_pipeline import run_pipeline
    return await run_pipeline(sample, view, settings, stage_dir, mode, budget)
