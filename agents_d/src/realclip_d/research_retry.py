"""Prepare an explicitly authorized technical-failure subset, never auto-retry."""
from pathlib import Path
import shutil

from .contracts import SampleInput
from .inputs import research_input
from .research_campaign import identity_key
from .storage import freeze_json, read_json, verify_stage


TECHNICAL_FAILURES = frozenset({"transient_provider_error", "local_input_size_guard",
                                "mixed_language_tool_content"})


def prepare_technical_rerun(candidates: list[dict], failed_cases: list[dict],
                            runs: Path, output: Path) -> dict:
    """Freeze a separate run's inputs/cache; completed and policy cases excluded.

    Caller must supply the user's authorized allowlist. Nothing here invokes a
    model, reconciles unknown charges, modifies an old stage or repeats a request.
    """
    runs, output = runs.resolve(), output.resolve()
    if (output / "manifest.json").exists() or any(output.glob("samples/*/research/requests/*.json")):
        raise ValueError("Targeted rerun requires a fresh directory")
    failures = {row["compound_id"]: row for row in failed_cases}
    approved, seen = [], set()
    for row in candidates:
        sid = row["compound_id"]
        failure = failures.get(sid)
        if sid in seen or failure is None or failure["category"] not in TECHNICAL_FAILURES:
            raise ValueError("Only unique, explicitly listed technical failures may be rerun")
        if row["category"] != failure["category"] or row["original_stage"] != failure["original_stage"]:
            raise ValueError("Retry candidate does not match its saved failure")
        seen.add(sid)
        stage = Path(row["original_stage"]).resolve()
        stage.relative_to(runs)
        payload = read_json(stage / "input.json")
        sample = SampleInput.model_validate(payload)
        if sample.sample_id != sid or sample.compound.inchikey != row["inchikey"]:
            raise ValueError("Retry identity does not match its original input")
        if read_json(stage / "binding.json").get("mode") != "live":
            raise ValueError("Retry requires a saved live attempt")
        approved.append((row, stage, payload, identity_key(payload)))

    keys = {item[3] for item in approved}
    completed = {}
    for pattern in ("*/samples/*/research/complete.json", "*/research/complete.json"):
        for seal in runs.glob(pattern):
            stage = seal.parent
            if read_json(stage / "binding.json").get("mode") != "live":
                continue
            key = identity_key(read_json(stage / "input.json"))
            if key in keys:
                verify_stage(stage)
                completed[key] = str(stage)

    scheduled, reused = [], []
    for row, stage, payload, key in approved:
        if key in completed:
            reused.append({**row, "completed_stage": completed[key]})
            continue
        sample = SampleInput.model_validate(payload)
        if research_input(sample) != payload:
            raise ValueError("Retry must preserve the original research input")
        path = output / "prepared_inputs" / f"{sample.sample_id}.json"
        freeze_json(path, payload)
        cache = stage / "cache"
        if cache.exists():
            shutil.copytree(cache, output / "samples" / sample.sample_id / "research/cache", dirs_exist_ok=False)
        scheduled.append({**row, "input_path": str(path)})
    plan = {"authorized_candidates": candidates, "scheduled": scheduled, "already_complete_excluded": reused,
            "provider_rejected_cases_held": [r["compound_id"] for r in failed_cases
                                              if r["category"] == "provider_policy_rejection"],
            "second_agent_enabled": False, "automatic_paid_retry": False,
            "old_attempts_and_unknown_reservations_unchanged": True}
    freeze_json(output / "targeted_plan.json", plan)
    return plan
