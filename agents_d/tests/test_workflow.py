"""Offline end-to-end checks for SDK execution, isolation and resumable artifacts."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import socket

import httpx
from PIL import Image
import pytest

from realclip_d.config import Settings
from realclip_d.storage import read_json, verify_stage, write_json
from realclip_d.workflow import execute


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Fail any attempted model, tracing or public-data network request."""
    def blocked(*args, **kwargs):
        raise AssertionError("This offline workflow test attempted a network connection")

    async def async_blocked(*args, **kwargs):
        blocked()

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(httpx.Client, "send", blocked)
    monkeypatch.setattr(httpx.AsyncClient, "send", async_blocked)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


@pytest.fixture
def sample_path(tmp_path: Path) -> Path:
    examples = Path(__file__).resolve().parents[1] / "examples"
    shutil.copytree(examples, tmp_path / "inputs")
    return tmp_path / "inputs" / "sample.json"


def no_runner_calls(monkeypatch):
    from realclip_d.agent_runtime import Runner

    async def blocked(*args, **kwargs):
        raise AssertionError("A saved stage unexpectedly called Runner.run")

    monkeypatch.setattr(Runner, "run", blocked)


@pytest.mark.asyncio
async def test_mock_runs_both_real_sdk_runners_offline(sample_path: Path, tmp_path: Path, monkeypatch):
    from realclip_d.agent_runtime import Runner

    original_run = Runner.run
    calls = []

    async def observed_run(agent, *args, **kwargs):
        calls.append(agent.name)
        return await original_run(agent, *args, **kwargs)

    monkeypatch.setattr(Runner, "run", observed_run)
    output = tmp_path / "complete"
    status = await execute("run", sample_path, output, Settings())

    assert status["status"] == "complete"
    assert status["synthetic"] is True
    assert status["biological_accuracy_validated"] is False
    assert calls == ["D Research Agent", "D Observation baseline", "D Observation guided",
                     "D Observation writer", "D Observation reviewer", "D Observation summarize"]
    assert status["agents"] == ["research", "observation"]
    assert status["observation_modules"] == ["baseline", "guided", "writer", "reviewer", "summarize"]
    assert read_json(output / "research/result.json")["morphology_hypotheses"] == []
    assert read_json(output / "observation/input.json")["hypotheses"] == []
    result = read_json(output / "observation/result.json")
    assert result["schema_version"] == "5"
    assert result["full_description"] == ""
    assert "SYNTHETIC" in result["description"]
    assert (output / "observation/description.txt").read_text().strip() == result["description"]
    assert "SYNTHETIC" in result["limitations"][0]
    for name in ("research", "observation", "observation/reviewed", "observation/summarize"):
        verify_stage(output / name)
    for name in ("research", "observation/reviewed", "observation/summarize"):
        assert list((output / name / "requests").glob("*.json"))
        assert list((output / name / "responses").glob("*.json"))
        assert read_json(output / name / "usage.json")["requests"] >= 1

    research_request = read_json(output / "research/requests/001.json")
    research_text = json.dumps(research_request)
    assert '"input_image"' not in research_text
    assert "cp_file" not in research_text
    assert "control_dna_f1" not in research_text
    observation_request = read_json(output / "observation/reviewed/phases/baseline/requests/001.json")
    observation_text = json.dumps(observation_request)
    assert [tool["name"] for tool in observation_request["tools"]] == ["inspect_image"]
    assert "input_image" in observation_text
    assert "SYNTHETIC_DEMO_COMPOUND" not in observation_text
    assert "compound" not in read_json(output / "observation/input.json")
    assert "known_moa" not in read_json(output / "observation/input.json")
    cp_exposure = read_json(output / "observation/reviewed/evidence.json")["cp_feature_ids"]
    assert set(cp_exposure) == {
        "nuclear_area_control", "nuclear_area_treatment", "nuclear_texture_control", "nuclear_texture_treatment",
    }


@pytest.mark.asyncio
async def test_replay_preserves_results_with_zero_sdk_or_network_calls(sample_path: Path, tmp_path: Path, monkeypatch):
    original, replayed = tmp_path / "original", tmp_path / "replayed"
    await execute("run", sample_path, original, Settings())
    no_runner_calls(monkeypatch)

    status = await execute("run", sample_path, replayed, Settings(), mode="replay", replay_from=original)

    assert status["model_calls"] == status["network_calls"] == 0
    assert status["replayed_stages"] == ["research", "observation"]
    for stage in ("research", "observation"):
        assert (original / stage / "result.json").read_bytes() == (replayed / stage / "result.json").read_bytes()
        assert verify_stage(original / stage) == verify_stage(replayed / stage)


@pytest.mark.asyncio
async def test_complete_run_resumes_without_runner_calls(sample_path: Path, tmp_path: Path, monkeypatch):
    output = tmp_path / "resumed"
    first = await execute("run", sample_path, output, Settings())
    research_receipt = verify_stage(output / "research")
    observation_receipt = verify_stage(output / "observation")
    no_runner_calls(monkeypatch)

    second = await execute("run", sample_path, output, Settings())

    assert second == first
    assert verify_stage(output / "research") == research_receipt
    assert verify_stage(output / "observation") == observation_receipt


@pytest.mark.asyncio
@pytest.mark.parametrize("reference_file", [False, True])
async def test_describe_separately_from_saved_research(
    sample_path: Path, tmp_path: Path, monkeypatch, reference_file: bool,
):
    from realclip_d import agent_runtime

    source, output = tmp_path / "research-only", tmp_path / "description-only"
    await execute("research", sample_path, source, Settings())
    assert not (source / "observation").exists()
    receipt = verify_stage(source / "research")

    async def no_research(*args, **kwargs):
        raise AssertionError("describe must reuse saved research")

    monkeypatch.setattr(agent_runtime, "run_research", no_research)
    reference = source / "research" / "result.json" if reference_file else source / "research"
    status = await execute("describe", sample_path, output, Settings(), research_path=reference)

    assert status["status"] == "complete"
    assert verify_stage(output / "research") == receipt
    assert verify_stage(source / "research") == receipt
    verify_stage(output / "observation")


@pytest.mark.asyncio
async def test_research_stage_resumes_into_complete_run(sample_path: Path, tmp_path: Path, monkeypatch):
    from realclip_d import agent_runtime

    output = tmp_path / "two_steps"
    await execute("research", sample_path, output, Settings())

    async def no_research(*args, **kwargs):
        raise AssertionError("A successful research stage must not be paid for again")

    monkeypatch.setattr(agent_runtime, "run_research", no_research)
    status = await execute("run", sample_path, output, Settings())
    assert status["status"] == "complete"
    verify_stage(output / "observation")


@pytest.mark.asyncio
async def test_missing_optional_cp_and_no_hypotheses_are_valid(sample_path: Path, tmp_path: Path):
    sample = read_json(sample_path)
    sample["cp_file"] = None
    write_json(sample_path, sample)
    output = tmp_path / "no_cp"

    await execute("run", sample_path, output, Settings())

    view = read_json(output / "observation/input.json")
    assert view["cp_available"] is False
    assert view["hypotheses"] == []
    receipts = [json.loads(line) for line in (output / "observation/reviewed/tools.jsonl").read_text().splitlines()]
    cp = next(receipt["result"] for receipt in receipts if receipt["tool"] == "query_cp")
    assert cp["status"] == "unavailable"
    assert read_json(output / "observation/reviewed/evidence.json")["cp_feature_ids"] == []
    verify_stage(output / "observation")


@pytest.mark.asyncio
async def test_named_missing_cp_file_fails_before_observation_model(sample_path: Path, tmp_path: Path, monkeypatch):
    from realclip_d import agent_runtime

    sample = read_json(sample_path)
    sample["cp_file"] = "does-not-exist.csv"
    write_json(sample_path, sample)
    output = tmp_path / "missing_cp"

    async def no_observation(*args, **kwargs):
        raise AssertionError("Missing CP file must fail before observation model invocation")

    monkeypatch.setattr(agent_runtime, "run_observation", no_observation)
    with pytest.raises(FileNotFoundError):
        await execute("run", sample_path, output, Settings())
    assert read_json(output / "status.json")["status"] == "failed"
    assert not (output / "observation/requests").exists()


@pytest.mark.asyncio
async def test_damaged_image_fails_before_observation_model(sample_path: Path, tmp_path: Path, monkeypatch):
    from realclip_d import agent_runtime

    (sample_path.parent / "images/treatment_dna_f1.png").write_bytes(b"not an image")
    output = tmp_path / "bad_image"

    async def no_observation(*args, **kwargs):
        raise AssertionError("Damaged images must fail before observation model invocation")

    monkeypatch.setattr(agent_runtime, "run_observation", no_observation)
    with pytest.raises(ValueError, match="Missing or damaged image"):
        await execute("run", sample_path, output, Settings())
    assert not (output / "observation/requests").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["compound", "context", "image", "cp", "settings"])
async def test_input_or_configuration_change_rejects_resume(
    sample_path: Path, tmp_path: Path, monkeypatch, change: str,
):
    output = tmp_path / "resume_changed"
    settings = Settings()
    await execute("run", sample_path, output, settings)
    no_runner_calls(monkeypatch)
    sample = read_json(sample_path)
    if change == "compound":
        sample["compound"]["name"] = "ANOTHER_SYNTHETIC_IDENTITY"
        write_json(sample_path, sample)
    elif change == "context":
        sample["context"]["dose"] = "changed synthetic condition"
        write_json(sample_path, sample)
    elif change == "image":
        image_path = sample_path.parent / "images/treatment_dna_f1.png"
        with Image.open(image_path) as image:
            changed = image.copy()
        changed.putpixel((0, 0), (255, 255, 255))
        changed.save(image_path)
    elif change == "cp":
        cp = sample_path.parent / "cp.csv"
        cp.write_text(cp.read_text().replace(",60,pixel^2,", ",65,pixel^2,"))
    else:
        settings = Settings(research_max_turns=7)

    with pytest.raises(ValueError, match="differ"):
        await execute("run", sample_path, output, settings)


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["research", "observation"])
async def test_tampered_stage_is_rejected_on_replay(sample_path: Path, tmp_path: Path, monkeypatch, stage: str):
    source, output = tmp_path / "tampered", tmp_path / "attempted_replay"
    await execute("run", sample_path, source, Settings())
    result_path = source / stage / "result.json"
    result = read_json(result_path)
    if stage == "research":
        result["compound_research_and_analysis"]["research_gaps"] += " Changed after completion."
    else:
        result["limitations"].append("Changed after the stage completed")
    write_json(result_path, result)
    no_runner_calls(monkeypatch)

    with pytest.raises(ValueError, match="Stage artifact missing or changed"):
        await execute("run", sample_path, output, Settings(), mode="replay", replay_from=source)
    assert not (output / "replay.json").exists()


@pytest.mark.asyncio
async def test_live_requires_explicit_budget_and_key_without_running_models(sample_path: Path, tmp_path: Path, monkeypatch):
    no_runner_calls(monkeypatch)
    with pytest.raises(ValueError, match="OPENAI_API_KEY.*budget"):
        await execute("run", sample_path, tmp_path / "accidental_live", Settings(), mode="live")


@pytest.mark.asyncio
async def test_research_workspace_is_sealed_and_reused_without_model_calls(sample_path, tmp_path, monkeypatch):
    output = tmp_path / "research-memory"
    await execute("research", sample_path, output, Settings())
    directory = output / "research"
    receipt = verify_stage(directory)
    assert "findings.json" in receipt["files"]
    assert "tools.jsonl" in receipt["files"]
    assert "sources.json" in receipt["files"]
    assert not (directory / "research_memory").exists()
    assert not (directory / "synthesis_input.json").exists()
    no_runner_calls(monkeypatch)
    assert (await execute("research", sample_path, output, Settings()))["status"] == "complete"
    state = directory / "findings.json"
    state.write_text(state.read_text() + " ")
    with pytest.raises(ValueError, match="artifact missing or changed"):
        verify_stage(directory)


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid", ["missing_well", "too_few_turns", "no_caption_channel"])
async def test_complete_pipeline_preflights_summary_before_research(
    sample_path, tmp_path, monkeypatch, invalid,
):
    from realclip_d import agent_runtime

    sample = read_json(sample_path)
    settings = Settings()
    if invalid == "missing_well":
        for image in sample["images"]:
            image.pop("well_id", None)
    elif invalid == "too_few_turns":
        settings = Settings(observation_max_turns=4)
    else:
        for image in sample["images"]:
            image["channel"] = "Brightfield_L"
    write_json(sample_path, sample)

    async def forbidden(*args, **kwargs):
        raise AssertionError("Invalid complete-agent inputs must fail before Research")

    monkeypatch.setattr(agent_runtime, "run_research", forbidden)
    with pytest.raises(ValueError):
        await execute("run", sample_path, tmp_path / "invalid-agent", settings)
