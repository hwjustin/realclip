"""Additional isolation and recovery regressions found during integration."""
from __future__ import annotations

from pathlib import Path
import shutil

import pytest

from realclip_d.config import Settings
from realclip_d.contracts import CompoundIdentity, ResearchResult
from realclip_d.inputs import hypothesis_view, image_manifest, load_sample
from realclip_d.storage import read_json
from realclip_d.workflow import execute


def copy_example(tmp_path: Path) -> Path:
    shutil.copytree(Path(__file__).parents[1] / "examples", tmp_path / "input")
    return tmp_path / "input/sample.json"


@pytest.mark.parametrize("private_id", ["aspirin", "aspirin_625nM", "case_aspirin"])
def test_compound_names_cannot_hide_in_observer_sample_ids(tmp_path, private_id):
    sample = load_sample(copy_example(tmp_path))
    sample.sample_id = private_id
    sample.compound = CompoundIdentity(name="aspirin")
    research = ResearchResult(sample_id=private_id, identity_summary="private", sources=[], facts=[], hypotheses=[], limitations=[])
    with pytest.raises(ValueError, match="private compound"):
        hypothesis_view(sample, research, image_manifest(sample))


def test_short_private_moa_in_context_is_rejected(tmp_path):
    sample = load_sample(copy_example(tmp_path))
    sample.known_moa = ["ROS"]
    sample.context.notes = ["ROS is expected"]
    research = ResearchResult(sample_id=sample.sample_id, identity_summary="private", sources=[], facts=[], hypotheses=[], limitations=[])
    with pytest.raises(ValueError, match="private compound"):
        hypothesis_view(sample, research, image_manifest(sample))


@pytest.mark.asyncio
async def test_describe_rejects_recursive_output_before_creating_it(tmp_path):
    sample_path = copy_example(tmp_path)
    run = tmp_path / "original"
    await execute("research", sample_path, run, Settings())
    nested = run / "research/nested"
    with pytest.raises(ValueError, match="overlap"):
        await execute("describe", sample_path, nested, Settings(), research_path=run / "research")
    assert not nested.exists()


@pytest.mark.asyncio
async def test_replay_rejects_recursive_output_before_creating_it(tmp_path):
    sample_path = copy_example(tmp_path)
    run = tmp_path / "original"
    await execute("run", sample_path, run, Settings())
    nested = run / "nested"
    with pytest.raises(ValueError, match="separate"):
        await execute("run", sample_path, nested, Settings(), mode="replay", replay_from=run)
    assert not nested.exists()


@pytest.mark.asyncio
async def test_incomplete_live_attempts_never_restart_model(tmp_path, monkeypatch):
    from realclip_d import agent_runtime

    sample_path = copy_example(tmp_path)
    run = tmp_path / "attempted"
    (run / "research/requests").mkdir(parents=True)
    (run / "research/requests/001.json").write_text("{}")
    monkeypatch.setenv("OPENAI_API_KEY", "test-only-never-sent")

    async def forbidden(*args, **kwargs):
        raise AssertionError("A paid attempt was repeated")

    monkeypatch.setattr(agent_runtime, "run_research", forbidden)
    with pytest.raises(ValueError, match="will not be resubmitted"):
        await execute("run", sample_path, run, Settings(budget_usd=1), mode="live")
    assert read_json(run / "status.json")["status"] == "failed"
