"""Research concurrency, shared limits, isolation and resumption; no live calls."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import socket
import threading
from types import SimpleNamespace

import httpx
import pytest

from realclip_d import agent_runtime, research_batch as batch
from realclip_d.agent_runtime import BudgetError, UnknownUsageError
from realclip_d.cli import main
from realclip_d.config import Settings
from realclip_d.storage import read_json, verify_stage, write_json
from realclip_d.tools.http import CachedHTTP, PubChemLimiter


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Research batch tests must not make network/model requests")
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(agent_runtime, "_live_model", forbidden)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


def inputs(tmp_path, count=3):
    paths = []
    for i in range(count):
        path = tmp_path / f"input_{i}.json"
        write_json(path, {"sample_id": f"compound_{i}", "compound": {"name": f"Synthetic compound {i}"},
                          "known_moa": [], "context": {"cell_line": "U2OS", "dose": None, "treatment_duration": None}})
        paths.append(path)
    return paths


def test_actual_sdk_mock_workers_are_isolated_and_resume_without_model_calls(tmp_path, monkeypatch):
    def no_observer(*args, **kwargs):
        pytest.fail("A research-only batch must never run the observation agent")
    monkeypatch.setattr(agent_runtime, "run_observation", no_observer)
    paths = inputs(tmp_path)
    root = tmp_path / "batch"
    result = batch.run_batch(paths, root, batch.batch_settings(), workers=3)
    assert all(row["status"] == "complete" for row in result["samples"])
    assert result["second_agent_calls"] == 0 and result["spent_estimate_usd"] == 0
    assert not list(root.glob("samples/*/observation"))
    for i in range(3):
        stage = root / "samples" / f"compound_{i}" / "research"
        verify_stage(stage)
        output = read_json(stage / "result.json")
        assert output["sample_id"] == f"compound_{i}"
        assert read_json(stage / "findings.json") == output["compound_research_and_analysis"]
        requests = sorted((stage / "requests").glob("*.json"))
        assert len(requests) == 3
        payload = requests[0].read_text()
        for other in range(3):
            if other != i:
                assert f"Synthetic compound {other}" not in payload
        saved_input = read_json(root / "inputs" / f"compound_{i}.json")
        assert "images" not in saved_input and "cp_file" not in saved_input

    async def no_repeat(*args, **kwargs):
        pytest.fail("Completed research must resume without another Runner.run")
    monkeypatch.setattr(agent_runtime.Runner, "run", no_repeat)
    resumed = batch.run_batch(paths, root, batch.batch_settings(), workers=1)
    assert all(row["status"] == "complete" for row in resumed["samples"])


@pytest.mark.parametrize("workers,count", [(2, 5), (10, 10), (16, 16), (32, 40)])
def test_workers_overlap_but_remain_bounded_and_share_budget_and_limiter(tmp_path, monkeypatch, workers, count):
    barrier = threading.Barrier(workers)
    lock = threading.Lock()
    state = {"active": 0, "maximum": 0}
    contexts, thread_ids, budgets, limiters = set(), set(), set(), set()

    async def researcher(sample, settings, stage, mode, budget, *, pubchem_limiter):
        with lock:
            state["active"] += 1
            state["maximum"] = max(state["maximum"], state["active"])
            contexts.add(id(asyncio.get_running_loop()))
            thread_ids.add(threading.get_ident())
            budgets.add(id(budget))
            limiters.add(id(pubchem_limiter))
        if int(sample.sample_id.split("_")[-1]) < workers:
            barrier.wait(timeout=5)
        with lock:
            state["active"] -= 1
        return SimpleNamespace(hypotheses=[], facts=[])

    monkeypatch.setattr(batch, "_research", researcher)
    options = {} if workers == 10 else {"workers": workers}
    result = batch.run_batch(inputs(tmp_path, count), tmp_path / "batch", Settings(), **options)
    assert state["maximum"] == workers and state["active"] == 0
    assert len(thread_ids) == workers and len(contexts) >= workers
    assert len(budgets) == len(limiters) == 1
    assert all(row["status"] == "complete" for row in result["samples"])


def test_one_compound_failure_does_not_repeat_or_prevent_other_research(tmp_path, monkeypatch):
    called = []
    async def researcher(sample, *args, **kwargs):
        called.append(sample.sample_id)
        if sample.sample_id == "compound_0":
            raise ValueError("Synthetic unavailable evidence")
        return SimpleNamespace(hypotheses=[], facts=[])
    monkeypatch.setattr(batch, "_research", researcher)
    result = batch.run_batch(inputs(tmp_path), tmp_path / "batch", Settings(), workers=2)
    assert [row["status"] for row in result["samples"]] == ["failed", "complete", "complete"]
    assert sorted(called) == ["compound_0", "compound_1", "compound_2"]


def test_unknown_usage_blocks_queued_compounds_without_any_new_request(tmp_path, monkeypatch):
    called = []
    async def researcher(sample, settings, stage, mode, budget, **kwargs):
        called.append(sample.sample_id)
        request = budget.reserve("research", False)
        budget.unknown(request, "Synthetic timeout with unknown usage")
        raise UnknownUsageError("Synthetic unknown usage")
    monkeypatch.setattr(batch, "_research", researcher)
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-offline-key")
    result = batch.run_batch(inputs(tmp_path), tmp_path / "batch", Settings(budget_usd=10), workers=1, mode="live")
    assert called == ["compound_0"]
    assert [row["status"] for row in result["samples"]] == ["failed", "blocked_unknown_usage", "blocked_unknown_usage"]
    assert result["blocked"]


def test_inflight_request_can_finish_after_another_worker_has_unknown_usage(tmp_path, monkeypatch):
    barrier = threading.Barrier(2)
    unknown = threading.Event()
    called = []
    async def researcher(sample, settings, stage, mode, budget, **kwargs):
        called.append(sample.sample_id)
        request = budget.reserve("research", False)
        barrier.wait(timeout=5)
        if sample.sample_id == "compound_0":
            budget.unknown(request, "Synthetic unknown usage")
            unknown.set()
            raise UnknownUsageError("Synthetic unknown usage")
        assert unknown.wait(timeout=5)
        budget.complete(request, {"input_tokens": 100, "output_tokens": 10}, 0)
        return SimpleNamespace(hypotheses=[], facts=[])
    monkeypatch.setattr(batch, "_research", researcher)
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-offline-key")
    root = tmp_path / "batch"
    result = batch.run_batch(inputs(tmp_path, 4), root, Settings(budget_usd=10), workers=2, mode="live")
    assert sorted(called) == ["compound_0", "compound_1"]
    assert [row["status"] for row in result["samples"]] == ["failed", "complete", "blocked_unknown_usage", "blocked_unknown_usage"]
    saved = read_json(root / "budget.json")
    assert sorted(row["status"] for row in saved["requests"]) == ["complete", "unknown"]
    assert saved["blocked"] and saved["spent_estimate_usd"] > 0


def test_isolated_failure_stops_one_compound_without_repeating_or_blocking_others(tmp_path, monkeypatch):
    called = []
    async def researcher(sample, settings, stage, mode, budget, **kwargs):
        called.append(sample.sample_id)
        request = budget.reserve("research", False)
        if sample.sample_id == "compound_0":
            budget.unknown(request, "Synthetic provider policy rejection; do not rephrase or retry")
            with pytest.raises(UnknownUsageError):
                budget.reserve("research", False)
            raise UnknownUsageError("Synthetic isolated failure")
        budget.complete(request, {"input_tokens": 10, "output_tokens": 1}, 0)
        return SimpleNamespace(hypotheses=[], facts=[])
    monkeypatch.setattr(batch, "_research", researcher)
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-offline-key")
    result = batch.run_batch(inputs(tmp_path, 5), tmp_path / "batch", Settings(budget_usd=10),
                             workers=1, mode="live", isolate_failures=True)
    assert called == [f"compound_{i}" for i in range(5)]
    assert [r["status"] for r in result["samples"]] == ["failed"] + ["complete"] * 4
    ledger = read_json(tmp_path / "batch/budget.json")
    assert not ledger["blocked"] and ledger["unknown_usage_scope"] == "compound"
    assert len(ledger["requests"]) == 5
    assert ledger["requests"][0]["sample_id"] == "compound_0"
    assert ledger["requests"][0]["status"] == "unknown"


def test_isolated_unknown_reservations_still_enforce_one_shared_cap_and_survive_reload(tmp_path):
    settings = Settings(max_input_tokens=1000, max_output_tokens=256)
    amount = (1000 * settings.input_usd_per_million + 256 * settings.output_usd_per_million) / 1e6
    cap = amount * 2.1
    path = tmp_path / "budget.json"
    ledger = batch.IsolatedBudgetLedger(path, cap, settings)
    request = ledger.for_sample("one").reserve("research", False)
    ledger.for_sample("one").unknown(request, "Synthetic unknown usage")
    ledger.for_sample("two").reserve("research", False)
    with pytest.raises(BudgetError):
        ledger.for_sample("three").reserve("research", False)
    restored = batch.IsolatedBudgetLedger(path, cap, settings)
    with pytest.raises(UnknownUsageError):
        restored.for_sample("one").reserve("research", False)
    with pytest.raises(UnknownUsageError):
        restored.for_sample("two").reserve("research", False)
    assert len(restored.state["requests"]) == 2 and not restored.state["blocked"]


def test_isolated_pool_still_stops_globally_if_provider_exceeds_reservation(tmp_path):
    settings = Settings(max_input_tokens=1000, max_output_tokens=256)
    pool = batch.IsolatedBudgetLedger(tmp_path / "budget.json", 10, settings)
    request = pool.for_sample("one").reserve("research", False)
    pool.for_sample("one").complete(request, {"input_tokens": 1001, "output_tokens": 1}, 0)
    with pytest.raises(UnknownUsageError):
        pool.for_sample("two").reserve("research", False)


def test_shared_budget_reservations_cannot_overspend_when_threads_race(tmp_path):
    settings = Settings(max_input_tokens=1000, max_output_tokens=256)
    reservation = (1000 * settings.input_usd_per_million + 256 * settings.output_usd_per_million) / 1e6
    ledger = batch.SharedBudgetLedger(tmp_path / "budget.json", reservation * 3.1, settings)
    barrier = threading.Barrier(10)
    def reserve(_):
        barrier.wait(timeout=5)
        try:
            return ledger.reserve("research", False)
        except BudgetError:
            return None
    with ThreadPoolExecutor(max_workers=10) as pool:
        reserved = list(pool.map(reserve, range(10)))
    assert sum(value is not None for value in reserved) == 3
    assert len(read_json(tmp_path / "budget.json")["requests"]) == 3
    assert read_json(tmp_path / "budget.json") == ledger.snapshot()


def test_shared_budget_concurrent_completions_preserve_every_charge(tmp_path):
    settings = Settings(max_input_tokens=1000, max_output_tokens=256, budget_usd=10)
    ledger = batch.SharedBudgetLedger(tmp_path / "budget.json", 10, settings)
    def complete(_):
        request = ledger.reserve("research", False)
        ledger.complete(request, {"input_tokens": 500, "output_tokens": 100}, 0)
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(complete, range(40)))
    saved = read_json(tmp_path / "budget.json")
    assert len(saved["requests"]) == 40
    assert all(row["status"] == "complete" for row in saved["requests"])
    assert saved["spent_estimate_usd"] == pytest.approx(40 * (500 * .125 + 100 * .5) / 1e6)
    assert not saved["blocked"]


def test_pubchem_limit_is_shared_across_separate_http_clients(tmp_path):
    clock = [0.0]
    def sleep(seconds):
        clock[0] += seconds
    limiter = PubChemLimiter(2, sleep=sleep, monotonic=lambda: clock[0])
    request_times = []
    def respond(request):
        request_times.append(clock[0])
        return httpx.Response(200, json={"ok": True})
    with httpx.Client(transport=httpx.MockTransport(respond)) as a, httpx.Client(transport=httpx.MockTransport(respond)) as b:
        clients = [CachedHTTP(tmp_path / "a", client=a, pubchem_limiter=limiter),
                   CachedHTTP(tmp_path / "b", client=b, pubchem_limiter=limiter)]
        for i in range(4):
            clients[i % 2].get(f"https://pubchem.ncbi.nlm.nih.gov/fixture/{i}")
    assert request_times == [0, .5, 1, 1.5]


@pytest.mark.parametrize("workers", [0, 33, True, 1.5])
def test_invalid_worker_counts_fail_before_start(tmp_path, workers):
    with pytest.raises(ValueError, match="workers"):
        batch.run_batch(inputs(tmp_path), tmp_path / "batch", Settings(), workers=workers)
    assert not (tmp_path / "batch").exists()


@pytest.mark.parametrize("sample_id", ["compound_0", "../escape", "/absolute"])
def test_duplicate_or_unsafe_ids_fail_before_any_job(tmp_path, monkeypatch, sample_id):
    paths = inputs(tmp_path, 2)
    value = read_json(paths[1]); value["sample_id"] = sample_id; write_json(paths[1], value)
    with pytest.raises(ValueError):
        batch.run_batch(paths, tmp_path / "batch", Settings())
    assert not (tmp_path / "batch").exists()


def test_live_budget_is_required_before_loading_credentials(tmp_path):
    with pytest.raises(ValueError, match="explicit total budget"):
        batch.run_batch(inputs(tmp_path), tmp_path / "batch", Settings(), mode="live", credential_file=tmp_path / "missing.env")


def test_cli_exposes_research_only_batch_with_mock_default(tmp_path, capsys):
    paths = inputs(tmp_path, 2)
    result = main(["research-batch", "--inputs", *(str(p) for p in paths),
                   "--output", str(tmp_path / "batch")])
    assert result == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["workers"] == 10 and summary["second_agent_calls"] == 0
    assert all(row["status"] == "complete" for row in summary["samples"])


def test_cli_reports_a_nonzero_exit_status_if_a_compound_failed(tmp_path, monkeypatch, capsys):
    async def unavailable(*args, **kwargs):
        raise ValueError("Synthetic failure")
    monkeypatch.setattr(batch, "_research", unavailable)
    paths = inputs(tmp_path, 1)
    assert main(["research-batch", "--inputs", str(paths[0]), "--output", str(tmp_path / "batch")]) == 2
    assert json.loads(capsys.readouterr().out)["samples"][0]["status"] == "failed"
