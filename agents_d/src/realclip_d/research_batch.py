"""Bounded compound-level parallelism; each worker runs the complete researcher."""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
import os
from pathlib import Path
import re
import threading
import time
import uuid

from .agent_runtime import BudgetLedger, BudgetError, UnknownUsageError
from .config import Settings
from .contracts import SampleInput, research_counts
from .inputs import research_input
from .language import validate_agent_input
from .storage import digest, freeze_json, read_json, run_lock, write_json
from .tools.http import PubChemLimiter
from .workflow import _research


def batch_settings() -> Settings:
    """The profile used by the latest successful three-compound research run."""
    return Settings(research_max_turns=50, max_output_tokens=12000,
                    max_input_tokens=1000000, max_tool_calls=100,
                    input_usd_per_million=0.25, output_usd_per_million=0.75)


class SharedBudgetLedger(BudgetLedger):
    """Serialize budget reservations and writes across this batch's threads."""

    def __init__(self, *args, **kwargs):
        self._guard = threading.RLock()
        super().__init__(*args, **kwargs)

    def reserve(self, *args, **kwargs):
        with self._guard:
            return super().reserve(*args, **kwargs)

    def complete(self, *args, **kwargs):
        with self._guard:
            return super().complete(*args, **kwargs)

    def unknown(self, *args, **kwargs):
        with self._guard:
            return super().unknown(*args, **kwargs)

    def snapshot(self) -> dict:
        with self._guard:
            return deepcopy(self.state)


class CompoundBudget:
    """One compound's view of a shared budget; failed requests are never retried."""
    def __init__(self, pool, sample_id):
        self.pool, self.sample_id = pool, sample_id

    def reserve(self, stage, web_enabled):
        return self.pool.reserve_for(self.sample_id, stage, web_enabled)

    def complete(self, *args, **kwargs):
        return self.pool.complete(*args, **kwargs)

    def unknown(self, *args, **kwargs):
        return self.pool.unknown(*args, **kwargs)


class IsolatedBudgetLedger(SharedBudgetLedger):
    """Hold uncertain charges, stop that compound, and continue unrelated ones.

    This changes neither provider rejection handling nor model retry behavior.
    Every unsettled reservation still counts against the one shared USD cap.
    Provider usage exceeding the reservation continues to block the whole pool.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if any(not row.get("sample_id") for row in self.state["requests"]):
            raise BudgetError("Failure-isolated budgets require compound-owned requests; use a new run directory")
        self.blocked_samples = {row["sample_id"] for row in self.state["requests"] if row["status"] != "complete"}
        self.state["blocked"] = any(row.get("warning") for row in self.state["requests"])
        self.state["unknown_usage_scope"] = "compound"

    def for_sample(self, sample_id):
        return CompoundBudget(self, sample_id)

    def reserve_for(self, sample_id, stage, web_enabled):
        with self._guard:
            if self.state["blocked"] or sample_id in self.blocked_samples:
                raise UnknownUsageError("This compound or the shared pool is blocked; no request was sent")
            s, cap = self.settings, self.state["budget_usd"]
            if cap is None or cap <= 0:
                raise BudgetError("Live mode requires an explicit positive USD budget")
            amount = (s.max_input_tokens * s.input_usd_per_million + s.max_output_tokens * s.output_usd_per_million) / 1e6
            amount += s.web_max_calls * s.web_search_usd_per_call if web_enabled else 0
            unsettled = sum(row["reserved_usd"] for row in self.state["requests"] if row["status"] != "complete")
            if self.state["spent_estimate_usd"] + unsettled + amount > cap:
                raise BudgetError(f"Budget cannot reserve ${amount:.6f} for the next request")
            request_id = uuid.uuid4().hex
            self.state["requests"].append({"request_id": request_id, "sample_id": sample_id,
                "stage": stage, "reserved_usd": amount, "status": "pending",
                "created_at": datetime.now(timezone.utc).isoformat()})
            self._save()
            return request_id

    def unknown(self, request_id, reason):
        with self._guard:
            row = self._row(request_id)
            row.update(status="unknown", error=reason, updated_at=datetime.now(timezone.utc).isoformat())
            self.blocked_samples.add(row["sample_id"])
            self._save()


def run_batch(inputs: list[Path], output: Path, settings: Settings, *,
              workers: int = 10, mode: str = "mock", credential_file: Path | None = None,
              isolate_failures: bool = False) -> dict:
    if mode not in {"mock", "live"} or isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= 32:
        raise ValueError("Research batch requires mock/live mode and 1..32 workers")
    if not inputs:
        raise ValueError("Research batch requires at least one input")
    if mode == "live" and settings.budget_usd is None:
        raise ValueError("Live research batch requires an explicit total budget")

    samples, manifest_rows = [], []
    seen = set()
    # Validate every input before credentials, workers or paid requests.
    for path in inputs:
        path = Path(path).resolve()
        sample = SampleInput.model_validate(read_json(path))
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", sample.sample_id):
            raise ValueError("Batch sample IDs must be safe directory names")
        if sample.sample_id in seen:
            raise ValueError("Batch sample IDs must be unique")
        if sample.images or sample.cp_file:
            raise ValueError("Research-only batch inputs must not supply images or CP files")
        seen.add(sample.sample_id)
        validate_agent_input(research_input(sample))
        samples.append(sample)
        manifest_rows.append({"sample_id": sample.sample_id, "input_path": str(path),
                              "research_input_sha256": digest(research_input(sample))})

    output = Path(output).resolve()
    with run_lock(output):
        freeze_json(output / "manifest.json", {"mode": mode, "settings": settings.model_dump(mode="json"),
                    "samples": manifest_rows, "second_agent_enabled": False,
                    **({"unknown_usage_scope": "compound"} if isolate_failures else {})})
        for sample in samples:
            freeze_json(output / "inputs" / f"{sample.sample_id}.json", sample.model_dump(mode="json", exclude={
                "images": True, "cp_file": True, "context": {"notes"} if not sample.context.notes else set()}))
        if mode == "live":
            if credential_file is not None:
                if settings.provider != 'openai':
                    raise ValueError('Local models use LOCAL_MODEL_API_KEY, not the OpenAI credential-file loader')
                from .research_pilot import load_authorized_key
                load_authorized_key(Path(credential_file))
            if settings.provider == "openai" and not os.environ.get(settings.api_key_env or "OPENAI_API_KEY"):
                raise ValueError("Live research batch requires OPENAI_API_KEY or --credential-file")
        ledger_type = IsolatedBudgetLedger if isolate_failures else SharedBudgetLedger
        ledger = ledger_type(output / "budget.json", settings.budget_usd, settings)
        limiter = PubChemLimiter(settings.pubchem_requests_per_second)
        started = time.monotonic()
        statuses = [{"sample_id": sample.sample_id, "name": sample.compound.name, "status": "pending"}
                    for sample in samples]

        def save_status(finished=False):
            budget = ledger.snapshot()
            summary = {"phase": "finished" if finished else "running", "workers": workers,
                       "samples": statuses, "elapsed_seconds": time.monotonic() - started,
                       "spent_estimate_usd": budget["spent_estimate_usd"],
                       "budget_cap_usd": settings.budget_usd, "blocked": budget["blocked"],
                       "second_agent_calls": 0}
            write_json(output / "status.json", summary)
            return summary

        def research_one(sample):
            began = time.monotonic()
            started_at = datetime.now(timezone.utc).isoformat()
            stage = output / "samples" / sample.sample_id / "research"
            status = {"sample_id": sample.sample_id, "name": sample.compound.name}
            if ledger.snapshot()["blocked"] and not (stage / "complete.json").is_file():
                status.update(status="blocked_unknown_usage")
            else:
                try:
                    compound_budget = ledger.for_sample(sample.sample_id) if isolate_failures else ledger
                    result = asyncio.run(_research(sample, settings, stage, mode, compound_budget,
                                                   pubchem_limiter=limiter))
                    status.update(status="complete", **research_counts(result))
                except Exception as error:
                    status.update(status="failed", error_type=type(error).__name__, error=str(error))
            status.update(started_at=started_at, duration_seconds=time.monotonic() - began,
                          completed_at=datetime.now(timezone.utc).isoformat())
            write_json(stage.parent / "status.json", status)
            return status

        save_status()
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="compound-research") as pool:
            futures = {pool.submit(research_one, sample): i for i, sample in enumerate(samples)}
            for future in as_completed(futures):
                statuses[futures[future]] = future.result()
                save_status()
        return save_status(finished=True)
