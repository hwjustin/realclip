"""Run plain-text Summarize through the real SDK without network access."""
from __future__ import annotations

import socket

import pytest

from realclip_d import agent_runtime as runtime
from realclip_d import summarize
from realclip_d.storage import file_hash, read_json, verify_stage, write_json
from realclip_d.summarize_contracts import SummarizeConstraints
from realclip_d.summarize_inputs import make_provisional_manifest
from test_summarize_inputs import case


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Summarize runtime tests forbid network access")
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


async def test_sdk_run_seals_plain_description_and_never_changes_upstream(case, tmp_path):
    before = {str(path.relative_to(case)): file_hash(path) for path in case.rglob("*") if path.is_file()}
    destination = tmp_path.parent / (tmp_path.name + "-summary")
    description = "Oval profiles are present."
    result = await summarize.run_summarize(case, make_provisional_manifest(case, ["DNA", "AGP"]),
                                          destination, fixture=description)
    assert result == {"schema_version": "2", "sample_id": "SYNTHETIC_CASE",
                      "description": description, "word_count": 4, "mode": "mock"}
    assert (destination / "description.txt").read_text() == description + "\n"
    assert read_json(destination / "result.json") == result
    assert read_json(destination / "usage.json")["requests"] == 1
    request = read_json(destination / "requests" / "001.json")
    assert request["output_schema"] is None
    assert request["tools"] == []
    for filename in ("output_schema.json", "validation.json", "corrections.json", "caption.txt"):
        assert not (destination / filename).exists()
    assert not list(destination.glob("candidate_*.json"))
    verify_stage(destination)
    assert before == {str(path.relative_to(case)): file_hash(path) for path in case.rglob("*") if path.is_file()}
    with pytest.raises(ValueError, match="prior results/attempts"):
        await summarize.run_summarize(case, make_provisional_manifest(case, ["DNA"]), destination,
                                      fixture=description)


async def test_plain_output_is_not_reference_filtered_or_length_corrected(case, tmp_path):
    description = ("Control and CellProfiler hypotheses mention tBF, cDNA and unknown_image. "
                   + "Visible profiles. " * 80).strip()
    destination = tmp_path.parent / (tmp_path.name + "-unfiltered")
    result = await summarize.run_summarize(case, make_provisional_manifest(case, ["DNA"]),
                                          destination, fixture=description)
    assert result["description"] == description
    assert result["word_count"] > SummarizeConstraints().max_words
    assert read_json(destination / "usage.json")["requests"] == 1
    assert not (destination / "corrections.json").exists()
    assert not (destination / "validation.json").exists()
    verify_stage(destination)


@pytest.mark.parametrize("text,error", [("  ", "nonempty plain-text"), ("细胞呈椭圆形。", "English-only")])
async def test_basic_output_failure_is_recorded_without_retry(case, tmp_path, text, error):
    destination = tmp_path.parent / (tmp_path.name + "-invalid")
    with pytest.raises(ValueError, match=error):
        await summarize.run_summarize(case, make_provisional_manifest(case, ["DNA"]),
                                      destination, fixture=text)
    assert read_json(destination / "usage.json")["requests"] == 1
    assert read_json(destination / "failure.json")["model_calls"] == 1
    assert read_json(destination / "failure.json")["upstream_rerun"] is False
    assert not (destination / "result.json").exists()
    assert not (destination / "complete.json").exists()
    assert not (destination / "corrections.json").exists()


async def test_runtime_hard_limit_prevents_a_second_model_request(case, tmp_path, monkeypatch):
    real_run = summarize.Runner.run

    async def attempted_replay(*args, **kwargs):
        await real_run(*args, **kwargs)
        return await real_run(*args, **kwargs)

    monkeypatch.setattr(summarize.Runner, "run", attempted_replay)
    destination = tmp_path.parent / (tmp_path.name + "-one-call")
    with pytest.raises(runtime.BudgetError, match="model-call limit"):
        await summarize.run_summarize(case, make_provisional_manifest(case, ["DNA"]),
                                      destination, fixture="Oval profiles are present.")
    assert read_json(destination / "usage.json")["requests"] == 1
    assert read_json(destination / "failure.json")["model_calls"] == 1
    assert len(list((destination / "requests").glob("*.json"))) == 1


async def test_transport_failure_does_not_retry_and_blocks_shared_ledger(case, tmp_path, monkeypatch):
    class FailingModel(summarize.TextFixtureModel):
        async def get_response(self, *args, **kwargs):
            raise OSError("Synthetic transport failure; no external request")

    monkeypatch.setattr(runtime, "_live_model", lambda settings: (FailingModel(""), None))
    settings = summarize.default_settings().model_copy(update={"budget_usd": 2})
    ledger_path = tmp_path.parent / (tmp_path.name + "-budget.json")
    ledger = runtime.BudgetLedger(ledger_path, 2, settings)
    destination = tmp_path.parent / (tmp_path.name + "-transport")
    with pytest.raises(OSError, match="Synthetic transport"):
        await summarize.run_summarize(case, make_provisional_manifest(case, ["DNA"]),
                                      destination, settings, mode="live", budget=ledger)
    saved = read_json(ledger_path)
    assert saved["blocked"] is True
    assert len(saved["requests"]) == 1
    assert saved["requests"][0]["status"] == "unknown"
    assert read_json(destination / "failure.json")["model_calls"] == 1
    assert read_json(destination / "usage.json")["requests"] == 1


async def test_missing_provider_usage_blocks_ledger_without_retry(case, tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "_live_model",
                        lambda settings: (summarize.TextFixtureModel("Oval profiles are present."), None))
    settings = summarize.default_settings().model_copy(update={"budget_usd": 2})
    ledger = runtime.BudgetLedger(tmp_path.parent / (tmp_path.name + "-budget.json"), 2, settings)
    destination = tmp_path.parent / (tmp_path.name + "-unknown-usage")
    with pytest.raises(runtime.UnknownUsageError, match="raw usage"):
        await summarize.run_summarize(case, make_provisional_manifest(case, ["DNA"]),
                                      destination, settings, mode="live", budget=ledger)
    assert ledger.state["blocked"] is True
    assert len(ledger.state["requests"]) == 1
    assert ledger.state["requests"][0]["status"] == "unknown"
    assert read_json(destination / "usage.json")["requests"] == 1
    assert not (destination / "result.json").exists()


async def test_no_affordable_reservation_sends_no_model_request(case, tmp_path, monkeypatch):
    sent = []

    class TrackingModel(summarize.TextFixtureModel):
        async def get_response(self, *args, **kwargs):
            sent.append(1)
            return await super().get_response(*args, **kwargs)

    monkeypatch.setattr(runtime, "_live_model", lambda settings: (TrackingModel("Unused."), None))
    settings = summarize.default_settings().model_copy(update={"budget_usd": 0.001})
    ledger = runtime.BudgetLedger(tmp_path.parent / (tmp_path.name + "-budget.json"), 0.001, settings)
    with pytest.raises(runtime.BudgetError, match="cannot reserve"):
        await summarize.run_summarize(case, make_provisional_manifest(case, ["DNA"]),
            tmp_path.parent / (tmp_path.name + "-no-budget"), settings, mode="live", budget=ledger)
    assert sent == []
    assert ledger.state["requests"] == []


async def test_read_only_paths_and_metering_config_are_checked(case, tmp_path):
    manifest = make_provisional_manifest(case, ["DNA"])
    with pytest.raises(ValueError, match="separate"):
        await summarize.run_summarize(case, manifest, case / "forbidden", fixture="Unused.")
    settings = summarize.default_settings().model_copy(update={"budget_usd": 2})
    inside = runtime.BudgetLedger(case / "forbidden-budget.json", 2, settings)
    with pytest.raises(ValueError, match="ledger must be outside"):
        await summarize.run_summarize(case, manifest, tmp_path.parent / "bad-ledger", settings,
                                      mode="live", budget=inside)
    other = runtime.BudgetLedger(tmp_path.parent / (tmp_path.name + "-budget.json"), 2, settings)
    incompatible = settings.model_copy(update={"max_output_tokens": 4000})
    with pytest.raises(runtime.BudgetError, match="must match"):
        await summarize.run_summarize(case, manifest, tmp_path.parent / "bad-settings", incompatible,
                                      mode="live", budget=other)


async def test_preflight_fails_before_constructing_live_client(case, tmp_path, monkeypatch):
    manifest = make_provisional_manifest(case, ["DNA"])
    manifest["images"][0]["well_id"] = "OTHER_WELL"

    def forbidden(settings):
        raise AssertionError("Invalid inputs must not construct a live client")

    monkeypatch.setattr(runtime, "_live_model", forbidden)
    with pytest.raises(ValueError):
        await summarize.run_summarize(case, manifest, tmp_path.parent / (tmp_path.name + "-invalid"),
                                      mode="live")


async def test_optional_tokenizer_is_guidance_only_and_never_loaded(case, tmp_path):
    destination = tmp_path.parent / (tmp_path.name + "-no-tokenizer")
    constraints = SummarizeConstraints(tokenizer_path="/PRIVATE_TOKENIZER_PATH/DOES_NOT_EXIST",
                                        max_text_tokens=1)
    result = await summarize.run_summarize(case, make_provisional_manifest(case, ["DNA"]),
        destination, constraints=constraints, fixture="Oval profiles are present.")
    assert result["description"] == "Oval profiles are present."
    assert "PRIVATE_TOKENIZER_PATH" not in (destination / "requests" / "001.json").read_text()
    assert "PRIVATE_TOKENIZER_PATH" not in (destination / "input.json").read_text()
    assert read_json(destination / "usage.json")["requests"] == 1


def test_cli_mock_accepts_text_fixture_and_prints_only_description(case, tmp_path, capsys):
    manifest_path = tmp_path.parent / (tmp_path.name + "-manifest.json")
    fixture_path = tmp_path.parent / (tmp_path.name + "-fixture.txt")
    destination = tmp_path.parent / (tmp_path.name + "-cli-summary")
    write_json(manifest_path, make_provisional_manifest(case, ["DNA"]))
    fixture_path.write_text("Oval profiles are present.\n")
    assert summarize.main(["run", "--observation", str(case), "--manifest", str(manifest_path),
                           "--output", str(destination), "--fixture", str(fixture_path)]) == 0
    assert capsys.readouterr().out == "Oval profiles are present.\n"
    verify_stage(destination)


def test_cli_has_no_corrective_call_option():
    with pytest.raises(SystemExit):
        summarize.parser().parse_args(["run", "--observation", "/synthetic", "--manifest", "/manifest",
                                      "--output", "/output", "--max-corrections", "1"])
