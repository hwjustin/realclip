"""Independent, budgeted compression of a completed Observation report.

Run with ``python -m realclip_d.summarize``. This module reads upstream artifacts
and writes a separate result; it never invokes Research, Observation or CP.
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import replace
import json
from pathlib import Path
import sys
from typing import Any

from agents import Agent, Model, ModelResponse, RunConfig, Runner
from agents.usage import Usage
from openai.types.responses import ResponseOutputMessage, ResponseOutputText

from . import agent_runtime as runtime
from .config import Settings, load_settings
from .language import validate_agent_input, validate_agent_output
from .storage import digest, file_hash, freeze_json, read_json, run_lock, seal_stage, write_json
from .summarize_contracts import SummarizeConstraints
from .summarize_inputs import build_request, make_provisional_manifest


def default_settings() -> Settings:
    """Use the existing pilot model and accounting rates, without web/tools."""
    return Settings(reasoning_effort="high", web_search=False, web_max_calls=0,
                    max_input_tokens=400000, max_output_tokens=8192,
                    input_usd_per_million=0.25, output_usd_per_million=0.75)


def prompt_path() -> Path:
    return Path(__file__).parent / "prompts" / "summarize_v1.md"


class TextFixtureModel(Model):
    """Return raw description text through the real SDK without any network."""

    def __init__(self, text: str):
        self.text = text

    async def get_response(self, *args: Any, **kwargs: Any) -> ModelResponse:
        message = ResponseOutputMessage(id="summary_fixture", type="message", role="assistant",
            status="completed", content=[ResponseOutputText(type="output_text", text=self.text,
                                                           annotations=[])])
        return ModelResponse(output=[message],
            usage=Usage(requests=1, input_tokens=0, output_tokens=0, total_tokens=0),
            response_id="summary_fixture")

    def stream_response(self, *args: Any, **kwargs: Any):
        raise NotImplementedError("Summarize fixtures are non-streaming")


def _usage(directory: Path) -> dict:
    calls = [json.loads(line) for line in (directory / "calls.jsonl").read_text().splitlines()
             ] if (directory / "calls.jsonl").exists() else []
    return {"requests": len(calls), **{
        key: sum((row.get("usage") or {}).get(key, 0) for row in calls)
        for key in ("input_tokens", "output_tokens", "total_tokens")}}


async def run_summarize(observation_dir: Path, manifest: dict, output_dir: Path,
                        settings: Settings | None = None, *,
                        constraints: SummarizeConstraints | None = None,
                        mode: str = "mock", budget: runtime.BudgetLedger | None = None,
                        fixture: str | None = None) -> dict:
    """Generate one plain description with at most one metered model request.

    Length and content guidance belong to the prompt. The application records
    the returned text without citation checks, filtering or corrective calls.
    Only basic nonempty/English output and source/budget preflight remain.
    """
    if mode not in {"mock", "live"}:
        raise ValueError("Use mock or live mode")
    settings = settings or default_settings()
    constraints = constraints or SummarizeConstraints()
    observation_dir, output_dir = observation_dir.resolve(), output_dir.resolve()
    if output_dir.is_relative_to(observation_dir) or observation_dir.is_relative_to(output_dir):
        raise ValueError("Summarize output must be separate from its upstream Observation directory")
    if budget is not None and (budget.path.resolve().is_relative_to(observation_dir)
                               or budget.path.resolve().is_relative_to(output_dir)):
        raise ValueError("Shared budget ledger must be outside source and sealed output directories")
    # All read-only source, target and manifest configuration checks
    # precede constructing a live client or making a budget reservation.
    request = build_request(observation_dir, manifest, constraints)
    validate_agent_input(request)
    if mode == "live" and (budget is None or settings.budget_usd is None):
        raise runtime.BudgetError("Live Summarize requires an explicit shared budget ledger and USD cap")
    if mode == "live":
        assert budget is not None
        metered_fields = ("model", "max_input_tokens", "max_output_tokens",
                          "input_usd_per_million", "output_usd_per_million")
        if (budget.state["budget_usd"] != settings.budget_usd
                or any(getattr(budget.settings, key) != getattr(settings, key) for key in metered_fields)):
            raise runtime.BudgetError("Shared ledger cap, model, prices and reservation bounds must match settings")
    if mode == "mock" and not isinstance(fixture, str):
        raise ValueError("Mock mode requires an explicit synthetic plain-text --fixture output")
    with run_lock(output_dir):
        if (output_dir / "result.json").exists() or any((output_dir / "requests").glob("*.json")):
            raise ValueError("Summarize has prior results/attempts; use a new output directory")
        frozen_settings = {**settings.model_dump(mode="json"), "mode": mode}
        freeze_json(output_dir / "settings.json", frozen_settings)
        freeze_json(output_dir / "input.json", request)
        freeze_json(output_dir / "visual_manifest.json", manifest)
        freeze_json(output_dir / "sources.json", [])
        freeze_json(output_dir / "binding.json", {
            "observation_directory": str(observation_dir),
            "observation_result_sha256": file_hash(observation_dir / "result.json"),
            "observation_receipt_sha256": file_hash(observation_dir / "complete.json"),
            "input_sha256": digest(request), "manifest_sha256": digest(manifest),
            "prompt_sha256": file_hash(prompt_path()), "upstream_read_only": True})
        prompt = prompt_path().read_text(encoding="utf-8")
        (output_dir / "instructions.md").write_text(prompt, encoding="utf-8")
        if mode == "live":
            inner, client = runtime._live_model(settings)
        else:
            assert isinstance(fixture, str)
            inner, client = TextFixtureModel(fixture), None

        class SummarizeRecordedModel(runtime.RecordedModel):
            async def get_response(self, *args: Any, **kwargs: Any):
                if self.calls >= 1:
                    raise runtime.BudgetError("Summarize model-call limit reached")
                return await super().get_response(*args, **kwargs)

        model = SummarizeRecordedModel(inner, settings, output_dir, "summarize", budget,
                                       mode == "live", False)
        agent = Agent(name="D Observation summarize", instructions=prompt, model=model,
                      model_settings=replace(runtime._model_settings(settings, False), truncation="disabled"),
                      output_type=None, tools=[])
        messages: list[dict] = [{"role": "user", "content": json.dumps(request, ensure_ascii=False)}]
        try:
            response = await Runner.run(agent, messages, max_turns=1,
                                        run_config=RunConfig(tracing_disabled=True))
            raw = response.final_output
            if not isinstance(raw, str) or not raw.strip():
                raise ValueError("Summarize must return a nonempty plain-text description")
            description = raw.strip()
            validate_agent_output({"description": description})
            result = {"schema_version": "2", "sample_id": request["target_treatment_well"]["sample_id"],
                      "description": description, "word_count": len(description.split()), "mode": mode}
            write_json(output_dir / "result.json", result)
            (output_dir / "description.txt").write_text(description + "\n", encoding="utf-8")
            write_json(output_dir / "usage.json", _usage(output_dir))
            seal_stage(output_dir)
            return result
        except BaseException as exc:
            write_json(output_dir / "failure.json", {"error_type": type(exc).__name__, "error": str(exc),
                       "model_calls": model.calls, "upstream_rerun": False})
            raise
        finally:
            # Do not mutate a sealed successful directory after creating receipt.
            if not (output_dir / "complete.json").exists():
                write_json(output_dir / "usage.json", _usage(output_dir))
            if client is not None:
                await client.close()


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Standalone post-Observation Summarize module")
    sub = root.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare", help="Offline: bind an explicit provisional treatment collection")
    prepare.add_argument("--observation", required=True, type=Path)
    prepare.add_argument("--channels", required=True, nargs="+")
    prepare.add_argument("--output", required=True, type=Path, help="New manifest JSON file")
    for name in ("check", "run"):
        command = sub.add_parser(name)
        command.add_argument("--observation", required=True, type=Path)
        command.add_argument("--manifest", required=True, type=Path)
        command.add_argument("--constraints", type=Path)
        if name == "run":
            command.add_argument("--output", required=True, type=Path)
            command.add_argument("--mode", choices=("mock", "live"), default="mock")
            command.add_argument("--fixture", type=Path, help="UTF-8 plain-text synthetic description for mock SDK runs")
            command.add_argument("--config", type=Path)
            command.add_argument("--budget-usd", type=float)
            command.add_argument("--budget-ledger", type=Path)
            command.add_argument("--credential-file", type=Path)
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "prepare":
            manifest = make_provisional_manifest(args.observation, args.channels)
            if args.output.resolve().is_relative_to(args.observation.resolve()):
                raise ValueError("Manifest must be written outside the upstream Observation directory")
            freeze_json(args.output, manifest)
            print(json.dumps({"status": "prepared", "images": len(manifest["images"]),
                              "manifest": str(args.output), "model_calls": 0}))
            return 0
        constraints = (SummarizeConstraints.model_validate(read_json(args.constraints))
                       if args.constraints else SummarizeConstraints())
        manifest = read_json(args.manifest)
        if args.command == "check":
            request = build_request(args.observation, manifest, constraints)
            print(json.dumps({"status": "checked", "model_calls": 0,
                "target_treatment_well": request["target_treatment_well"],
                "images": len(request["clip_visual_manifest"]["images"]),
                "reviewed_context_words": sum(len(row["text"].split())
                    for row in request["reviewed_description_context"]),
                "eligible_observations": len(request["eligible_observations"])}))
            return 0
        settings = load_settings(args.config) if args.config else default_settings()
        if args.budget_usd is not None:
            settings = Settings.model_validate({**settings.model_dump(), "budget_usd": args.budget_usd})
        # A preflight failure must not load credentials or construct a paid client.
        build_request(args.observation, manifest, constraints)
        if args.mode == "live":
            if settings.budget_usd is None or args.budget_ledger is None:
                raise runtime.BudgetError("Live runs require --budget-usd and --budget-ledger")
            if (args.budget_ledger.resolve().is_relative_to(args.observation.resolve())
                    or args.budget_ledger.resolve().is_relative_to(args.output.resolve())):
                raise ValueError("Shared budget ledger must be outside source and sealed output directories")
            if args.credential_file is not None:
                from .research_pilot import load_authorized_key
                load_authorized_key(args.credential_file)
            # Sequential CLI calls serialize on the ledger's parent directory.
            with run_lock(args.budget_ledger.resolve().parent):
                budget = runtime.BudgetLedger(args.budget_ledger, settings.budget_usd, settings)
                result = asyncio.run(run_summarize(args.observation, manifest, args.output, settings,
                    constraints=constraints, mode="live", budget=budget))
        else:
            result = asyncio.run(run_summarize(args.observation, manifest, args.output, settings,
                constraints=constraints,
                fixture=args.fixture.read_text(encoding="utf-8") if args.fixture else None))
        print(result["description"])
        return 0
    except Exception as exc:
        print(f"summarize: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
