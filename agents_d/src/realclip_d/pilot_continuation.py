"""Explicit research synthesis from saved evidence; no repeated retrieval.

This is a new, recorded reasoning request by the same research-agent definition.
It is never used when a previous request's usage is unknown. Original attempts
and errors remain intact, and every new request uses the original shared ledger.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from agents import Agent, RunConfig, Runner

from .agent_runtime import BudgetLedger, RecordedModel, _jsonable, _live_model, _model_settings, _research_messages, _research_payload
from .config import Settings
from .contracts import ResearchResultV2, ResearchResultV3, SampleInput, Source, research_counts, validate_research_references
from .research_protocol import research_receipts, validate_research_protocol
from .language import validate_agent_input, validate_agent_output
from .storage import file_hash, freeze_json, read_json, run_lock, seal_stage, verify_stage, write_json


def _saved_output_type(stage: Path):
    """Continue with the original request's schema, never silently upgrade it."""
    requests = sorted((stage / "requests").glob("*.json"))
    original = read_json(requests[0]) if requests else {}
    properties = (original.get("output_schema") or {}).get("properties", {})
    metadata_path = stage / "output_contract.json"
    metadata = read_json(metadata_path) if metadata_path.is_file() else None
    if metadata is not None and metadata != {"name": "ResearchResultV3", "schema_version": "3"}:
        raise ValueError("Unsupported saved research output contract")
    modern = "compound_identification" in properties
    if metadata is not None and not modern:
        raise ValueError("Saved contract metadata disagrees with the original model request schema")
    return ResearchResultV3 if modern else ResearchResultV2


async def complete_saved(directory: Path, credential_file: Path) -> dict:
    from .pilot_evidence import build_evidence_packet
    from .research_pilot import load_authorized_key

    manifest = read_json(directory / "manifest.json")
    settings = Settings.model_validate(manifest["settings"])
    if settings.budget_usd != 5 or manifest["budget_cap_usd"] != 5 or manifest["second_agent_enabled"]:
        raise ValueError("Continuation must use the original research-only shared budget")
    load_authorized_key(credential_file)
    statuses = []
    with run_lock(directory):
        ledger = BudgetLedger(directory / "budget.json", 5, settings)
        if ledger.state["blocked"]:
            raise ValueError("Unknown usage must be reconciled before any continuation")
        for row in manifest["samples"]:
            stage = directory / "samples" / row["sample_id"] / "research"
            if (stage / "complete.json").exists():
                verify_stage(stage)
                statuses.append(read_json(stage.parent / "status.json"))
                continue
            path = Path(row["input_path"])
            if file_hash(path) != row["input_sha256"]:
                raise ValueError("Prepared pilot input changed")
            sample = SampleInput.model_validate(read_json(path))
            validate_agent_input({"compound": sample.compound.model_dump(mode="json"),
                                  "known_moa": sample.known_moa, "context": sample.context.model_dump(mode="json")})
            if sample.images or sample.cp_file:
                raise ValueError("Research pilot must not include image or CP inputs")
            initial_path = stage.parent / "initial_status.json"
            original_status = read_json(initial_path if initial_path.is_file() else stage.parent / "status.json")
            freeze_json(stage.parent / "initial_status.json", original_status)
            output_type = _saved_output_type(stage)
            candidate_path = stage / "candidate_result.json"
            if candidate_path.is_file():
                # Recheck the exact original output locally before considering
                # a new model call. A corrected validator must not cost a call.
                try:
                    saved_candidate = output_type.model_validate(read_json(candidate_path))
                    validate_agent_output(saved_candidate.model_dump(mode="json"))
                    saved_sources = [Source.model_validate(item) for item in read_json(stage / "sources.json")]
                    validate_research_references(saved_candidate, saved_sources)
                    validate_research_protocol(saved_candidate, sample, research_receipts(stage))
                    if saved_candidate.sample_id != sample.sample_id:
                        raise ValueError("Saved candidate returned a different sample ID")
                except ValueError as error:
                    freeze_json(stage / "candidate_revalidation.json", {"valid": False, "error": str(error)})
                else:
                    freeze_json(stage / "result.json", saved_candidate.model_dump(mode="json"))
                    freeze_json(stage / "validation.json", {"references_valid": True,
                        "biological_accuracy_validated": False, "offline_candidate_revalidated": True,
                        "additional_model_calls": 0})
                    seal_stage(stage)
                    status = {"sample_id": sample.sample_id, "name": row["name"], "status": "complete",
                              **research_counts(saved_candidate),
                              "offline_candidate_revalidated": True, "initial_error": original_status}
                    write_json(stage.parent / "status.json", status)
                    statuses.append(status)
                    print(json.dumps(status, ensure_ascii=False), flush=True)
                    continue
            prior = sorted((stage / "requests").glob("*.json"))
            continuations = sorted(stage.glob("continuation_*"))
            if continuations:
                raise ValueError("A saved continuation already exists; it will not be resubmitted")
            if len(prior) >= settings.research_max_turns or len(ledger.state["requests"]) >= 48:
                raise ValueError("Original shared request-count limit has been reached")
            original_prompt = read_json(prior[0]).get("system_instructions") if prior else None
            if not isinstance(original_prompt, str) or not original_prompt.strip():
                raise ValueError("Saved original research prompt is required for a continuation")
            format_path = stage / "output_format.md"
            output_format = format_path.read_text() if format_path.is_file() else None
            packet = build_evidence_packet(stage)
            instruction = ("Finalize the four-stage research result using only these previously shown tool receipts. "
                           "No new retrieval is performed. Distinguish source findings from predictions and state "
                           "remaining research gaps. Return the structured result now."
                           if output_type is ResearchResultV3 else
                           "Finalize the research result from these saved tool receipts only. "
                           "Earlier retrieval stopped before final synthesis. No new retrieval is performed. "
                           "Search hits are not read articles. Only actual supplied excerpts establish article evidence. "
                           "Do not claim full-article reading. A full_text literature fact must cite the actual "
                           "fullTextXML Source ID; same-article metadata may be co-cited. "
                           "Return the structured result now; report remaining gaps.")
            payload = {**_research_payload(sample),
                       "saved_evidence": packet,
                       "continuation": instruction}
            continuation = stage / "continuation_01"
            freeze_json(continuation / "input.json", payload)
            freeze_json(continuation / "binding.json", {
                "same_agent": "D Research Agent", "prompt_sha256": hashlib.sha256(original_prompt.encode()).hexdigest(),
                "prompt_origin": str(prior[0].relative_to(stage)),
                "output_format_sha256": file_hash(format_path) if output_format is not None else None,
                "original_error": original_status, "original_model_requests": len(prior),
                "shared_budget": str(directory / "budget.json"), "web_enabled": False,
                "new_retrieval": False, "maximum_additional_requests": 1,
            })
            print(f"Synthesizing saved research evidence only: {row['name']}", flush=True)
            client = None
            try:
                inner, client = _live_model(settings)
                model = RecordedModel(inner, settings, continuation, "research", ledger, True, False)
                agent = Agent(name="D Research Agent", instructions=original_prompt, model=model,
                              model_settings=_model_settings(settings, False), tools=[], output_type=output_type)
                response = await Runner.run(agent, _research_messages(payload, output_format), max_turns=1,
                                            run_config=RunConfig(tracing_disabled=True))
                candidate = output_type.model_validate(response.final_output)
                freeze_json(continuation / "candidate_result.json", candidate.model_dump(mode="json"))
                validate_agent_output(candidate.model_dump(mode="json"))
                sources = [Source.model_validate(item) for item in packet["sources"]]
                validate_research_references(candidate, sources)
                validate_research_protocol(candidate, sample, research_receipts(stage))
                if candidate.sample_id != sample.sample_id:
                    raise ValueError("Synthesis returned a different sample ID")
                # Preserve an earlier invalid candidate if present; final validated
                # result is a separate immutable artifact.
                if not (stage / "candidate_result.json").exists():
                    freeze_json(stage / "candidate_result.json", candidate.model_dump(mode="json"))
                freeze_json(stage / "result.json", candidate.model_dump(mode="json"))
                freeze_json(stage / "validation.json", {"references_valid": True,
                    "biological_accuracy_validated": False, "completed_from_saved_evidence": True})
                freeze_json(continuation / "usage.json", _jsonable(response.context_wrapper.usage))
                seal_stage(stage)
                status = {"sample_id": sample.sample_id, "name": row["name"], "status": "complete",
                          **research_counts(candidate),
                          "completed_from_saved_evidence": True, "initial_error": original_status}
            except Exception as error:
                status = {"sample_id": sample.sample_id, "name": row["name"], "status": "failed",
                          "error_type": type(error).__name__, "error": str(error), "initial_error": original_status}
            finally:
                if client:
                    await client.close()
            write_json(stage.parent / "status.json", status)
            statuses.append(status)
            print(json.dumps(status, ensure_ascii=False), flush=True)
            if ledger.state["blocked"]:
                break
        outcome = {"samples": statuses, "spent_estimate_usd": ledger.state["spent_estimate_usd"],
                   "blocked": ledger.state["blocked"], "second_agent_calls": 0, "budget_cap_usd": 5}
        write_json(directory / "status.json", outcome)
        return outcome
