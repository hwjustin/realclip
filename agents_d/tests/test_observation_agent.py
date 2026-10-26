"""Offline integration checks for Observation's five-module public boundary."""
from __future__ import annotations

import shutil
import socket
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from realclip_d.config import Settings
from realclip_d.inputs import load_sample
from realclip_d.observation_agent import (
    MODULES, preflight_observation_agent, read_observation_agent, run_observation_agent,
)
from realclip_d.storage import file_hash, read_json, seal_stage, verify_stage, write_json
from realclip_d.workflow import _research


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("Observation integration tests must not access the network")

    async def async_blocked(*args, **kwargs):
        blocked()

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(httpx.Client, "send", blocked)
    monkeypatch.setattr(httpx.AsyncClient, "send", async_blocked)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


@pytest.fixture
def sample(tmp_path):
    source = Path(__file__).resolve().parents[1] / "examples"
    shutil.copytree(source, tmp_path / "input")
    path = tmp_path / "input/sample.json"
    payload = read_json(path)
    for image in payload["images"]:
        image["well_id"] = "SYNTHETIC_" + image["role"].upper() + "_WELL"
    write_json(path, payload)
    return load_sample(path)


def forbid_runner(monkeypatch):
    from realclip_d.agent_runtime import Runner

    async def blocked(*args, **kwargs):
        raise AssertionError("A saved module unexpectedly invoked Runner.run")

    monkeypatch.setattr(Runner, "run", blocked)


async def run_mock(sample, tmp_path, *, settings=None):
    settings = settings or Settings()
    research = await _research(sample, settings, tmp_path / "research", "mock", None)
    directory = tmp_path / "observation"
    result = await run_observation_agent(sample, research, settings, directory, "mock", None)
    return research, directory, result


def test_preflight_uses_only_available_original_treatment_channels(sample):
    assert preflight_observation_agent(sample, Settings()) == ["DNA"]
    assert preflight_observation_agent(sample, Settings(observation_summary_channels=["ER", "DNA"])) == ["DNA"]


@pytest.mark.parametrize("case", ["missing_well", "multiple_wells", "missing_channels", "too_few_turns"])
def test_preflight_rejects_invalid_summary_inputs(sample, case):
    settings = Settings()
    if case == "missing_well":
        next(image for image in sample.images if image.role == "treatment").well_id = None
    elif case == "multiple_wells":
        next(image for image in sample.images if image.role == "treatment").well_id = "SYNTHETIC_OTHER_WELL"
    elif case == "missing_channels":
        settings = Settings(observation_summary_channels=["ER"])
    else:
        settings = Settings(observation_max_turns=4)
    with pytest.raises(ValueError):
        preflight_observation_agent(sample, settings)


@pytest.mark.asyncio
async def test_five_modules_run_offline_and_publish_plain_summary(sample, tmp_path, monkeypatch):
    from realclip_d.agent_runtime import Runner

    original = Runner.run
    seen = []

    async def observed(agent, *args, **kwargs):
        seen.append(agent.name)
        return await original(agent, *args, **kwargs)

    monkeypatch.setattr(Runner, "run", observed)
    _, directory, result = await run_mock(sample, tmp_path)
    assert seen == ["D Research Agent", *["D Observation " + module for module in MODULES]]
    assert result.schema_version == "5"
    assert result.modules == list(MODULES)
    assert result.full_description == ""
    assert result.description.startswith("SYNTHETIC MOCK OUTPUT: no biological description")
    assert read_json(directory / "result.json")["description"] == result.description
    assert (directory / "description.txt").read_text() == result.description + "\n"
    assert read_json(directory / "reviewed/result.json")["schema_version"] == "4"
    assert read_json(directory / "summarize/result.json")["schema_version"] == "2"
    assert "sentences" not in read_json(directory / "summarize/result.json")
    for stage in (directory, directory / "reviewed", directory / "summarize"):
        verify_stage(stage)
    binding = read_json(directory / "binding.json")
    assert binding["module_receipt_sha256"] == {
        name: file_hash(directory / name / "complete.json") for name in ("reviewed", "summarize")}
    usage = read_json(directory / "usage.json")
    assert usage["requests"] == sum(read_json(directory / name / "usage.json")["requests"] for name in ("reviewed", "summarize"))
    pipeline = read_json(directory / "pipeline.json")
    assert [row["phase"] for row in pipeline["phases"]] == list(MODULES)
    assert pipeline["model_turns"] == usage["requests"] <= 8
    assert read_json(directory / "reviewed/settings.json")["observation_max_turns"] == 7
    assert read_json(directory / "summarize/settings.json")["observation_max_turns"] == 8
    assert read_json(directory / "summarize/usage.json")["requests"] == 1
    assert list((directory / "requests").glob("summarize_*.json"))


@pytest.mark.asyncio
async def test_completed_agent_resumes_without_runner(sample, tmp_path, monkeypatch):
    research, directory, first = await run_mock(sample, tmp_path)
    receipt = verify_stage(directory)
    forbid_runner(monkeypatch)
    second = await run_observation_agent(sample, research, Settings(), directory, "mock", None)
    assert second == first
    assert verify_stage(directory) == receipt


@pytest.mark.asyncio
async def test_completed_agent_with_deleted_summary_fails_before_runner(sample, tmp_path, monkeypatch):
    research, directory, _ = await run_mock(sample, tmp_path)
    shutil.rmtree(directory / "summarize")
    forbid_runner(monkeypatch)
    with pytest.raises(ValueError, match="artifact missing or changed"):
        await run_observation_agent(sample, research, Settings(), directory, "live", object())


@pytest.mark.asyncio
async def test_modules_receive_same_ledger_and_unchanged_metering_bounds(sample, tmp_path, monkeypatch):
    from realclip_d import observation_agent, workflow

    settings = Settings(max_input_tokens=120000, max_output_tokens=2048,
                        input_usd_per_million=0.25, output_usd_per_million=0.75)
    research = await _research(sample, settings, tmp_path / "research", "mock", None)
    ledger = SimpleNamespace(path=tmp_path / "budget.json")
    original_core = workflow._describe
    original_summary = observation_agent.run_summarize
    received = []

    async def core(sample, research, core_settings, directory, mode, budget, handoff_path):
        received.append(("reviewed", core_settings, budget))
        return await original_core(sample, research, core_settings, directory, mode, budget, handoff_path)

    async def summary(observation_dir, manifest, output_dir, summary_settings, **kwargs):
        received.append(("summarize", summary_settings, kwargs["budget"]))
        return await original_summary(observation_dir, manifest, output_dir, summary_settings, **kwargs)

    monkeypatch.setattr(workflow, "_describe", core)
    monkeypatch.setattr(observation_agent, "run_summarize", summary)
    # The sentinel exposes a ledger location but has no reserve method, making
    # it impossible for this test to perform a paid reservation.
    await run_observation_agent(sample, research, settings, tmp_path / "observation", "mock", ledger)
    assert [name for name, _, _ in received] == ["reviewed", "summarize"]
    assert all(budget is ledger for _, _, budget in received)
    for _, module_settings, _ in received:
        for key in ("model", "max_input_tokens", "max_output_tokens",
                    "input_usd_per_million", "output_usd_per_million", "budget_usd"):
            assert getattr(module_settings, key) == getattr(settings, key)


@pytest.mark.asyncio
async def test_complete_modules_can_finish_outer_stage_without_runner(sample, tmp_path, monkeypatch):
    research, directory, first = await run_mock(sample, tmp_path)
    (directory / "complete.json").unlink()
    forbid_runner(monkeypatch)
    assert await run_observation_agent(sample, research, Settings(), directory, "mock", None) == first
    verify_stage(directory)


@pytest.mark.asyncio
async def test_saved_agent_can_be_read_after_directory_copy(sample, tmp_path, monkeypatch):
    _, directory, first = await run_mock(sample, tmp_path)
    destination = tmp_path / "replayed_observation"
    shutil.copytree(directory, destination)
    shutil.rmtree(directory)
    forbid_runner(monkeypatch)
    assert read_observation_agent(destination) == first


@pytest.mark.asyncio
@pytest.mark.parametrize("module", ["reviewed", "summarize"])
async def test_nested_completion_receipts_are_explicitly_bound(sample, tmp_path, module):
    _, directory, _ = await run_mock(sample, tmp_path)
    path = directory / module / "complete.json"
    receipt = read_json(path)
    receipt["extra_note"] = "Receipt content changed without changing any bound data file"
    write_json(path, receipt)
    # seal_stage excludes all complete.json files, so the outer receipt alone
    # would not notice this. The explicit module receipt hashes must catch it.
    verify_stage(directory)
    with pytest.raises(ValueError, match="module completion receipt changed"):
        read_observation_agent(directory)


@pytest.mark.asyncio
@pytest.mark.parametrize("module", ["reviewed", "summarize"])
async def test_changed_module_result_fails_read(sample, tmp_path, module):
    _, directory, _ = await run_mock(sample, tmp_path)
    path = directory / module / "result.json"
    value = read_json(path)
    value["description"] += " Altered text."
    write_json(path, value)
    with pytest.raises(ValueError, match="artifact missing or changed"):
        read_observation_agent(directory)


@pytest.mark.asyncio
async def test_summary_failure_preserves_reviewed_and_has_no_outer_completion(sample, tmp_path, monkeypatch):
    from realclip_d import observation_agent

    settings = Settings()
    research = await _research(sample, settings, tmp_path / "research", "mock", None)
    directory = tmp_path / "observation"

    async def fail_summary(*args, **kwargs):
        raise RuntimeError("Synthetic summary failure")

    monkeypatch.setattr(observation_agent, "run_summarize", fail_summary)
    with pytest.raises(RuntimeError, match="Synthetic summary failure"):
        await run_observation_agent(sample, research, settings, directory, "mock", None)
    verify_stage(directory / "reviewed")
    assert not (directory / "complete.json").exists()
    assert not (directory / "result.json").exists()


@pytest.mark.asyncio
async def test_incomplete_live_summary_is_not_resubmitted(sample, tmp_path, monkeypatch):
    from realclip_d import observation_agent, workflow

    research, completed, _ = await run_mock(sample, tmp_path / "original")
    directory = tmp_path / "partial"
    shutil.copytree(completed / "reviewed", directory / "reviewed")
    partial = directory / "summarize/requests/001.json"
    write_json(partial, {"synthetic": True, "attempt": "already submitted"})
    receipt = verify_stage(directory / "reviewed")

    async def sealed_core(*args, **kwargs):
        from realclip_d.observation_stages import ObservationPipelineResult
        return ObservationPipelineResult.model_validate(read_json(directory / "reviewed/result.json"))

    async def forbidden_summary(*args, **kwargs):
        raise AssertionError("An incomplete paid summary must not be resubmitted")

    monkeypatch.setattr(workflow, "_describe", sealed_core)
    monkeypatch.setattr(observation_agent, "run_summarize", forbidden_summary)
    with pytest.raises(ValueError, match="Incomplete live Summarize"):
        await run_observation_agent(sample, research, Settings(), directory, "live", object())
    assert verify_stage(directory / "reviewed") == receipt
    assert partial.exists()
    assert not (directory / "complete.json").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["settings", "image", "cp", "handoff", "research"])
async def test_resume_refuses_changed_inputs_before_runner(sample, tmp_path, monkeypatch, change):
    research, directory, _ = await run_mock(sample, tmp_path)
    settings, handoff = Settings(), None
    if change == "settings":
        settings = Settings(observation_max_turns=9)
    elif change == "image":
        sample.images[0].display_settings = "SYNTHETIC changed rendering metadata"
    elif change == "cp":
        sample.cp_file.write_text(sample.cp_file.read_text().replace(",60,pixel^2,", ",61,pixel^2,"))
    elif change == "handoff":
        handoff = tmp_path / "invalid_handoff.json"
        audit = read_json(directory / "handoff.json")["revision"]
        audit["research_sha256"] = "changed"
        write_json(handoff, audit)
    else:
        research.compound_research_and_analysis.research_gaps += " Synthetic changed research source."
    forbid_runner(monkeypatch)
    with pytest.raises(ValueError):
        await run_observation_agent(sample, research, settings, directory, "mock", None, handoff)


@pytest.mark.asyncio
async def test_summary_receipt_source_binding_is_checked(sample, tmp_path):
    _, directory, _ = await run_mock(sample, tmp_path)
    summary = directory / "summarize"
    binding = read_json(summary / "binding.json")
    binding["observation_receipt_sha256"] = "0" * 64
    write_json(summary / "binding.json", binding)
    (summary / "complete.json").unlink()
    seal_stage(summary)
    outer = read_json(directory / "binding.json")
    outer["module_receipt_sha256"]["summarize"] = file_hash(summary / "complete.json")
    write_json(directory / "binding.json", outer)
    (directory / "complete.json").unlink()
    seal_stage(directory)
    with pytest.raises(ValueError, match="Summarize source, configuration or prompt binding differs"):
        read_observation_agent(directory)
