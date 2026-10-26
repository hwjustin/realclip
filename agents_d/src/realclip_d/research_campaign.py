"""Plan a dataset-wide researcher run and save unattended progress/cost telemetry.

No dataset/Arrow dependency, model prompt changes, observer, paid retry, or
automatic restart. The launcher supplies a small exported metadata JSON file.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import threading
import time

from .contracts import SampleInput, parse_research_result
from .inputs import research_input
from .language import validate_agent_input
from .storage import digest, freeze_json, read_json, verify_stage, write_json


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def identity_key(payload: dict) -> tuple | None:
    key = (payload.get("compound") or {}).get("inchikey")
    context = payload.get("context") or {}
    if not key:
        return None
    return key, context.get("cell_line"), context.get("dose"), context.get("treatment_duration")


def completed_research(runs: Path) -> tuple[dict, dict]:
    """Prefer the newest sealed live result; never mistake mocks for completion."""
    complete, attempted = {}, {}
    stages = set(path.parent for pattern in ("*/samples/*/research/input.json", "*/research/input.json")
                 for path in runs.glob(pattern))
    for stage in sorted(stages):
        binding = stage / "binding.json"
        if not binding.is_file() or read_json(binding).get("mode") != "live":
            continue
        payload = read_json(stage / "input.json")
        key = identity_key(payload)
        if key is None:
            continue
        if (stage / "complete.json").is_file():
            try:
                verify_stage(stage)
                result = parse_research_result(read_json(stage / "result.json"))
                if result.sample_id != payload["sample_id"]:
                    raise ValueError("Result sample ID does not match the saved input")
            except (ValueError, OSError, KeyError) as error:
                attempted.setdefault(key, []).append({"stage": str(stage), "reason": str(error)})
                continue
            row = {"stage": str(stage), "result_path": str(stage / "result.json"),
                   "sample_id": payload["sample_id"], "completed_mtime": (stage / "complete.json").stat().st_mtime,
                   "result_sha256": digest(result.model_dump(mode="json"))}
            if key not in complete or row["completed_mtime"] > complete[key]["completed_mtime"]:
                complete[key] = row
        elif any((stage / "requests").glob("*.json")):
            attempted.setdefault(key, []).append({"stage": str(stage),
                "reason": "Incomplete live research has recorded model attempts; do not resubmit automatically."})
    return complete, attempted


def prepare_plan(records: list[dict], runs: Path, output: Path) -> dict:
    """Build one input per full InChIKey; reuse previous completed contexts."""
    complete, attempted = completed_research(runs)
    planned, reused, held = [], [], []
    seen_ids, seen_keys = set(), set()
    context = {"cell_line": "U2OS", "dose": None, "treatment_duration": None}
    for record in records:
        compound_id, inchikey = record["compound_id"], record["inchikey"]
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", compound_id):
            raise ValueError("Unsafe dataset compound ID")
        if compound_id in seen_ids or inchikey in seen_keys:
            raise ValueError("Dataset export must contain unique compound IDs and complete InChIKeys")
        seen_ids.add(compound_id); seen_keys.add(inchikey)
        sample = SampleInput(sample_id=compound_id, compound={"inchikey": inchikey},
                             known_moa=record.get("known_moa", []), context=context)
        payload = research_input(sample)
        validate_agent_input(payload)
        key = identity_key(payload)
        info = {**record, "sample_id": compound_id}
        if key in complete:
            reused.append({**info, **complete[key]})
        elif key in attempted:
            held.append({**info, "attempts": attempted[key]})
        else:
            path = output / "prepared_inputs" / f"{compound_id}.json"
            value = sample.model_dump(mode="json", exclude={"images": True, "cp_file": True, "context": {"notes"}})
            freeze_json(path, value)
            planned.append({**info, "input_path": str(path.resolve()), "research_input_sha256": digest(payload)})
    plan = {"created_at": now(), "dataset_compounds": len(records), "scheduled": planned,
            "reused": reused, "held_for_review": held, "workers": 32,
            "context": context, "second_agent_enabled": False,
            "deduplication": "Full InChIKey and cell_line/dose/treatment_duration; newest verified sealed live research reused regardless of older tool version. Incomplete paid attempts are held, never automatically repeated."}
    freeze_json(output / "campaign_plan.json", plan)
    return plan


def cost_summary(ledger: dict) -> dict:
    known, tokens, web_calls = 0.0, Counter(), 0
    rows = ledger.get("requests", [])
    for row in rows:
        if row.get("status") != "complete":
            continue
        usage = row.get("usage") or {}
        details = usage.get("input_tokens_details") or {}
        inp, out = usage.get("input_tokens", 0), usage.get("output_tokens", 0)
        cached, writes = details.get("cached_tokens", 0), details.get("cache_write_tokens", 0)
        if cached + writes > inp:
            raise ValueError("Invalid token cache breakdown")
        long = inp > 272000
        web = row.get("web_calls", 0)
        known += ((inp - cached - writes) * .10 + cached * .01 + writes * .125) * (2 if long else 1) / 1e6
        known += out * .50 * (1.5 if long else 1) / 1e6 + web * .01
        tokens.update(input_tokens=inp, cached_tokens=cached, cache_write_tokens=writes, output_tokens=out)
        web_calls += web
    statuses = Counter(row.get("status", "unknown") for row in rows)
    return {"updated_at": now(), "known_standard_price_estimate_usd": known,
            "conservative_budget_estimate_usd": ledger.get("spent_estimate_usd", 0),
            "unsettled_reserved_usd": sum(row["reserved_usd"] for row in rows if row.get("status") != "complete"),
            "budget_cap_usd": ledger.get("budget_usd"), "blocked": ledger.get("blocked", False),
            "model_requests": len(rows), "request_statuses": dict(statuses), "tokens": dict(tokens), "web_calls": web_calls,
            "billing_reconciled": False, "price_source": "https://developers.openai.com/api/docs/models/gpt-6-luna",
            "note": "Known returned usage at standard/cache/long-context rates plus hosted web calls at $0.01 each; not an invoice. Unknown usage is excluded from the known estimate and remains reserved."}


class CampaignProgress:
    """Small local heartbeat; no agent, model call, notification or resubmission."""
    def __init__(self, output: Path, plan: dict, monitor, interval: float = 30):
        self.output, self.plan, self.monitor, self.interval = output, plan, monitor, interval
        self.started = time.monotonic()
        self.done = threading.Event()
        self.errors = []

    def checkpoint(self, phase="running"):
        status_path, budget_path = self.output / "status.json", self.output / "budget.json"
        status = read_json(status_path) if status_path.exists() else {"samples": []}
        ledger = read_json(budget_path) if budget_path.exists() else {}
        costs = cost_summary(ledger)
        counts = Counter(row["status"] for row in status["samples"])
        if phase == "finished" and costs["blocked"]:
            phase = "stopped_unknown_usage_or_budget_anomaly"
        elif phase == "finished" and any(key != "complete" and value for key, value in counts.items()):
            phase = "finished_with_errors"
        elif phase == "finished" and self.plan["held_for_review"]:
            phase = "finished_with_held_samples"
        with self.monitor._lock:
            events = list(self.monitor.http_events)
        rates = {"updated_at": now(), "http_requests": len(events),
            "statuses_by_host": dict(Counter(f'{row["host"]}:{row["status_code"]}' for row in events)),
            "http_429_by_host": dict(Counter(row["host"] for row in events if row["status_code"] == 429)),
            "http_503_by_host": dict(Counter(row["host"] for row in events if row["status_code"] == 503)),
            "transport_errors_by_host": dict(Counter(row["host"] for row in events if row["transport_error_type"])),
            "last_response_limit_headers_by_host": {}}
        for row in events:
            if row["response_limit_headers"]:
                rates["last_response_limit_headers_by_host"][row["host"]] = row["response_limit_headers"]
        progress = {"updated_at": now(), "phase": phase, "pid": os.getpid(), "workers": self.plan["workers"],
            "elapsed_seconds": time.monotonic() - self.started, "dataset_compounds": self.plan["dataset_compounds"],
            "previously_completed_reused": len(self.plan["reused"]), "scheduled": len(self.plan["scheduled"]),
            "held_for_review": len(self.plan["held_for_review"]), "counts": dict(counts),
            "completed_total": len(self.plan["reused"]) + counts["complete"],
            "known_standard_price_estimate_usd": costs["known_standard_price_estimate_usd"],
            "budget_blocked": costs["blocked"], "second_agent_calls": 0, "monitor_errors": self.errors}
        write_json(self.output / "progress.json", progress)
        write_json(self.output / "costs.json", costs)
        write_json(self.output / "rate_limits.json", rates)

    def _watch(self):
        while not self.done.wait(self.interval):
            try:
                self.checkpoint()
            except Exception as error:
                self.errors.append(f"{type(error).__name__}: {error}")
                write_json(self.output / "progress_errors.json", {"updated_at": now(), "errors": self.errors})

    def __enter__(self):
        self.checkpoint()
        self.thread = threading.Thread(target=self._watch, name="campaign-progress", daemon=True)
        self.thread.start()
        return self

    def __exit__(self, kind, error, tb):
        self.done.set(); self.thread.join()
        self.checkpoint("failed" if error else "finished")
        if error:
            write_json(self.output / "fatal_error.json", {"updated_at": now(), "error_type": type(error).__name__, "error": str(error)})
        statuses = read_json(self.output / "status.json").get("samples", []) if (self.output / "status.json").exists() else []
        write_json(self.output / "failed_samples.json", [row for row in statuses if row["status"] != "complete"])
        # A complete index also points to reused results in their original directories.
        index = [{**row, "status": "reused"} for row in self.plan["reused"]]
        index.extend({**row, "status": "held_for_review"} for row in self.plan["held_for_review"])
        for row in statuses:
            index.append({**row, "result_path": str(self.output / "samples" / row["sample_id"] / "research/result.json")})
        write_json(self.output / "result_index.json", index)
