"""Three research-only cases under one explicitly authorized shared budget."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import re
import shlex

from .agent_runtime import BudgetLedger
from .config import Settings
from .contracts import CompoundIdentity, SampleInput, research_counts
from .storage import file_hash, freeze_json, read_json, run_lock, write_json
from .tools import CachedHTTP, PubChemClient
from .workflow import _research


CASES = [
    {"sample_id": "D_R001", "name": "Nocodazole", "cid": 4122,
     "known_moa": ["Microtubule destabilizer"],
     "selection_reason": "Microtubule perturbation: distinguish direct morphological evidence from mechanism-based inference.",
     "selection_source": "https://pubchem.ncbi.nlm.nih.gov/compound/4122"},
    {"sample_id": "D_R002", "name": "Staurosporine", "cid": 44259,
     "known_moa": ["Broad-spectrum protein kinase inhibitor"],
     "selection_reason": "Broad kinase perturbation: preserve multi-target and experimental-context dependence.",
     "selection_source": "https://pubchem.ncbi.nlm.nih.gov/compound/44259"},
    {"sample_id": "D_R003", "name": "Rapamycin", "cid": 5284616,
     "known_moa": ["mTOR inhibitor"],
     "selection_reason": "Growth-signaling perturbation: test inference limits and observability when evidence is sparse.",
     "selection_source": "https://pubchem.ncbi.nlm.nih.gov/compound/5284616"},
]


def pilot_settings() -> Settings:
    return Settings(budget_usd=5, research_max_turns=16, max_output_tokens=12000,
                    max_input_tokens=250000, input_usd_per_million=0.25,
                    output_usd_per_million=0.75)


def load_authorized_key(path: Path) -> None:
    """Read only the explicit authorized key, never source a shell env file."""
    found = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"\s*(?:export\s+)?OPENAI_API_KEY\s*=\s*(.*)$", line)
        if match:
            tokens = shlex.split(match.group(1), comments=True, posix=True)
            if len(tokens) != 1 or not tokens[0]:
                raise ValueError("Authorized OPENAI_API_KEY entry is empty or malformed")
            found.append(tokens[0])
    if len(found) != 1:
        raise ValueError("Expected exactly one OPENAI_API_KEY in the authorized credential file")
    os.environ["OPENAI_API_KEY"] = found[0]
    os.environ.pop("OPENAI_BASE_URL", None)  # This pilot uses OpenAI's official API.


def prepare(directory: Path) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    settings = pilot_settings()
    reservation = (settings.max_input_tokens * settings.input_usd_per_million / 1e6
                   + settings.max_output_tokens * settings.output_usd_per_million / 1e6
                   + settings.web_max_calls * settings.web_search_usd_per_call)
    maximum = len(CASES) * settings.research_max_turns * reservation
    if maximum > 5:
        raise ValueError("Frozen worst-case model request plan exceeds the shared cap")
    prepared = []
    http = CachedHTTP(directory / "preparation_cache")
    try:
        client = PubChemClient(http)
        for case in CASES:
            input_path = directory / "inputs" / f"{case['sample_id']}.json"
            if input_path.exists():
                sample = SampleInput.model_validate(read_json(input_path))
            else:
                result = client.resolve_compound(CompoundIdentity(name=case["name"], pubchem_cid=case["cid"]))
                freeze_json(directory / "identity_preparation" / f"{case['sample_id']}.json", result.model_dump(mode="json"))
                if result.status != "ok":
                    raise ValueError(f"Public identity preparation failed for {case['name']}: {result.status}")
                compound = result.data["compound"]
                sample = SampleInput(sample_id=case["sample_id"], compound=CompoundIdentity(
                    name=case["name"], pubchem_cid=case["cid"], inchikey=compound["InChIKey"]),
                    known_moa=case["known_moa"],
                    context={"cell_line": "U2OS", "dose": None, "treatment_duration": None})
                freeze_json(input_path, sample.model_dump(mode="json", exclude={
                    "images": True, "cp_file": True, "context": {"notes"}}))
            prepared.append({**case, "input_path": str(input_path.resolve()), "input_sha256": file_hash(input_path)})
    finally:
        http.close()
    manifest = {"version": "research-only-three-case-v1", "samples": prepared,
                "settings": settings.model_dump(mode="json"), "budget_cap_usd": 5,
                "maximum_reserved_plan_usd": maximum, "reservation_per_request_usd": reservation,
                "model": settings.model, "reasoning_effort": settings.reasoning_effort,
                "second_agent_enabled": False, "images_or_cp_supplied": False,
                "paid_retries": False, "shared_ledger": "budget.json",
                "credential_file": ".env",
                "pricing_sources": ["https://developers.openai.com/api/docs/models/gpt-6-luna", "https://developers.openai.com/api/docs/pricing"],
                "price_note": "Conservative long-context cache-write/output rates; standard service. Estimates are not billing reconciliation."}
    freeze_json(directory / "manifest.json", manifest)
    return manifest


async def run(directory: Path, credential_file: Path) -> dict:
    if not (directory / "manifest.json").is_file():
        raise ValueError("Run prepare before issuing model requests")
    manifest = read_json(directory / "manifest.json")
    settings = Settings.model_validate(manifest["settings"])
    if settings.budget_usd != 5 or manifest["budget_cap_usd"] != 5 or manifest["second_agent_enabled"]:
        raise ValueError("Pilot scope or budget differs from the authorized frozen plan")
    load_authorized_key(credential_file)
    with run_lock(directory):
        budget = BudgetLedger(directory / "budget.json", 5, settings)
        statuses = []
        for row in manifest["samples"]:
            if budget.state["blocked"]:
                statuses.append({"sample_id": row["sample_id"], "status": "blocked_unknown_usage"})
                continue
            path = Path(row["input_path"])
            if file_hash(path) != row["input_sha256"]:
                raise ValueError("Prepared pilot input changed")
            sample = SampleInput.model_validate(read_json(path))
            stage = directory / "samples" / sample.sample_id / "research"
            print(f"Starting research only: {row['name']} ({sample.sample_id})", flush=True)
            try:
                result = await _research(sample, settings, stage, "live", budget)
                status = {"sample_id": sample.sample_id, "name": row["name"], "status": "complete",
                          **research_counts(result)}
            except Exception as error:
                status = {"sample_id": sample.sample_id, "name": row["name"], "status": "failed",
                          "error_type": type(error).__name__, "error": str(error)}
            write_json(stage.parent / "status.json", status)
            statuses.append(status)
            write_json(directory / "status.json", {"samples": statuses, "spent_estimate_usd": budget.state["spent_estimate_usd"],
                       "blocked": budget.state["blocked"], "second_agent_calls": 0})
            print(json.dumps(status, ensure_ascii=False), flush=True)
        outcome = {"samples": statuses, "spent_estimate_usd": budget.state["spent_estimate_usd"],
                   "blocked": budget.state["blocked"], "second_agent_calls": 0,
                   "budget_cap_usd": 5}
        write_json(directory / "status.json", outcome)
        return outcome


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["prepare", "run", "complete"])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--credential-file", type=Path, default=Path(".env"))
    args = parser.parse_args()
    if args.phase == "prepare":
        value = prepare(args.run_dir.resolve())
        print(json.dumps({"samples": len(value["samples"]), "budget_cap_usd": 5,
                          "maximum_reserved_plan_usd": value["maximum_reserved_plan_usd"], "paid_calls": 0}))
    elif args.phase == "run":
        print(json.dumps(asyncio.run(run(args.run_dir.resolve(), args.credential_file)), ensure_ascii=False))
    else:
        from .pilot_continuation import complete_saved
        print(json.dumps(asyncio.run(complete_saved(args.run_dir.resolve(), args.credential_file)), ensure_ascii=False))


if __name__ == "__main__":
    main()
