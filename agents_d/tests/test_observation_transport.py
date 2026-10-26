"""Real SDK wire checks and byte-preserving transport lifecycle, entirely offline."""
import asyncio
import base64
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import socket
from types import SimpleNamespace

import httpx
import pytest
from agents import AgentOutputSchema, ModelTracing, function_tool

from realclip_d import agent_runtime as runtime
from realclip_d.config import Settings
from realclip_d.observation_contracts import ObservationOutput
from realclip_d.observation_transport import ObservationTransport, current_transport
from test_observation_pipeline import case, phase_finals


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def no_network(*args, **kwargs):
        raise AssertionError("Transport tests must not use the network")
    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setattr(socket.socket, "connect_ex", no_network)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


def image(data=b"synthetic-exact-source-bytes", **extra):
    return {"type": "input_image", "image_url": "data:image/png;base64," + base64.b64encode(data).decode(),
            "detail": "high", **extra}


class Files:
    def __init__(self, fail_upload=None, fail_delete=False):
        self.uploaded, self.deleted = [], []
        self.fail_upload, self.fail_delete = fail_upload, fail_delete

    async def create(self, *, file, purpose, expires_after):
        await asyncio.sleep(0)
        if file[1] == self.fail_upload:
            raise ValueError("fixture upload failure")
        assert purpose == "vision"
        assert expires_after == {"anchor": "created_at", "seconds": 3600}
        self.uploaded.append(file)
        return SimpleNamespace(id="file_" + hashlib.sha256(file[1]).hexdigest())

    async def delete(self, file_id):
        self.deleted.append(file_id)
        if self.fail_delete:
            raise ValueError("fixture deletion failure")
        return SimpleNamespace(deleted=True)


@pytest.mark.asyncio
async def test_bytes_order_metadata_detail_and_crops_preserved(tmp_path):
    files = Files()
    original = [{"role": "user", "content": [{"type": "input_text", "text": "metadata"},
        image(), image(b"distinct control"), image()]},
        {"type": "function_call_output", "call_id": "crop_call", "output": [
            {"type": "input_text", "text": "crop provenance"}, image(b"lossless crop")]}]
    before = deepcopy(original)
    async with ObservationTransport(tmp_path / "transport", client=SimpleNamespace(files=files)) as transport:
        first = await transport.prepare_input(original)
        again = await transport.prepare_input(original)
        assert first == again and original == before
        assert len(files.uploaded) == 3
        assert {f[1] for f in files.uploaded} == {b"synthetic-exact-source-bytes", b"distinct control", b"lossless crop"}
        reverse = {row["file_id"]: key for key, row in transport.files.items()}
        def compare(a, b):
            if isinstance(a, dict):
                if a.get("type") == "input_image":
                    data = base64.b64decode(a["image_url"].split(",", 1)[1])
                    assert reverse[b["file_id"]] == hashlib.sha256(data).hexdigest()
                    assert {k: v for k, v in a.items() if k != "image_url"} == {k: v for k, v in b.items() if k != "file_id"}
                else:
                    assert a.keys() == b.keys()
                    for k in a: compare(a[k], b[k])
            elif isinstance(a, list):
                assert len(a) == len(b)
                for aa, bb in zip(a, b): compare(aa, bb)
            else: assert a == b
        compare(original, first)
    assert len(files.deleted) == 3
    assert current_transport() is None
    assert json.loads((tmp_path / "transport/summary.json").read_text())["all_deleted"]


@pytest.mark.asyncio
async def test_simultaneous_prepare_deduplicates_and_context_is_local(tmp_path):
    async def one(name, data):
        files = Files()
        async with ObservationTransport(tmp_path / name, client=SimpleNamespace(files=files)) as session:
            results = await asyncio.gather(session.prepare_input([image(data)]), session.prepare_input([image(data)]))
            assert results[0] == results[1] and current_transport() is session
        assert len(files.uploaded) == len(files.deleted) == 1
        return results[0]
    a, b = await asyncio.gather(one("a", b"first"), one("b", b"second"))
    assert a != b and current_transport() is None


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["input", "model", "instructions", "tools", "text", "reasoning", "truncation"])
async def test_count_cache_requires_identical_complete_payload(tmp_path, change):
    calls = []
    async def counter(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(input_tokens=100 + len(calls))
    payload = {"model": "fixture", "input": ["source"], "instructions": "observe", "tools": [],
               "text": {}, "reasoning": {"effort": "high"}, "truncation": "disabled"}
    async with ObservationTransport(tmp_path / "transport", client=SimpleNamespace(files=Files())) as session:
        assert await session.count(payload, counter) == (101, False)
        assert await session.count(dict(reversed(list(payload.items()))), counter) == (101, True)
        changed = deepcopy(payload)
        changed[change] = "changed fixture"
        assert await session.count(changed, counter) == (102, False)
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_failed_count_is_never_cached(tmp_path):
    calls = []
    async def counter(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1: raise ValueError("fixture count failure")
        return SimpleNamespace(input_tokens=10)
    async with ObservationTransport(tmp_path / "transport", client=SimpleNamespace(files=Files())) as session:
        with pytest.raises(ValueError, match="fixture count failure"):
            await session.count({"model": "fixture"}, counter)
        assert await session.count({"model": "fixture"}, counter) == (10, False)


@pytest.mark.asyncio
async def test_partial_upload_failure_cleans_successes_and_expiry_is_recorded(tmp_path):
    files = Files(fail_upload=b"bad", fail_delete=True)
    with pytest.raises(ValueError, match="upload failure"):
        async with ObservationTransport(tmp_path / "transport", client=SimpleNamespace(files=files)) as session:
            await session.prepare_input([image(b"good"), image(b"bad")])
    assert len(files.uploaded) == len(files.deleted) == 1
    result = json.loads((tmp_path / "transport/summary.json").read_text())
    assert not result["all_deleted"] and result["expires_after_seconds"] == 3600
    assert result["files"][0]["cleanup_error"]
    assert current_transport() is None


@pytest.mark.asyncio
async def test_upload_failure_stops_new_upload_dispatch(tmp_path):
    files = Files(fail_upload=b"bad")
    with pytest.raises(ValueError, match="upload failure"):
        async with ObservationTransport(tmp_path / "transport", client=SimpleNamespace(files=files)) as session:
            await session.prepare_input([image(b"good"), image(b"bad"), image(b"must not upload")])
    assert [file[1] for file in files.uploaded] == [b"good"]
    assert len(files.deleted) == 1


@pytest.mark.asyncio
async def test_transient_file_errors_are_bounded_without_generation_retry(tmp_path, monkeypatch):
    calls = []
    class TransientError(Exception):
        status_code = 503
    class FlakyFiles(Files):
        async def create(self, **kwargs):
            calls.append("upload")
            if calls.count("upload") < 2: raise TransientError("fixture transient upload")
            return await super().create(**kwargs)
        async def delete(self, file_id):
            calls.append("delete")
            if calls.count("delete") < 2: raise TransientError("fixture transient cleanup")
            return await super().delete(file_id)
    original_sleep = asyncio.sleep
    async def instant(delay):
        await original_sleep(0)
    monkeypatch.setattr(asyncio, "sleep", instant)
    files = FlakyFiles()
    async with ObservationTransport(tmp_path / "transport", client=SimpleNamespace(files=files)) as session:
        await session.prepare_input([image()])
    assert calls == ["upload", "upload", "delete", "delete"]
    assert len(files.uploaded) == len(files.deleted) == 1
    assert not session.counts


@pytest.mark.asyncio
async def test_persistent_file_errors_stop_after_three_attempts(tmp_path, monkeypatch):
    calls = []
    class TransientError(Exception):
        status_code = 503
    class BrokenFiles(Files):
        async def create(self, **kwargs):
            calls.append("upload")
            raise TransientError("fixture persistent upload failure")
    async def instant(delay): pass
    monkeypatch.setattr(asyncio, "sleep", instant)
    async with ObservationTransport(tmp_path / "transport", client=SimpleNamespace(files=BrokenFiles())) as session:
        with pytest.raises(TransientError):
            await session.prepare_input([image()])
    assert calls == ["upload"] * 3
    assert not session.files


@pytest.mark.asyncio
async def test_cleanup_404_records_absence_without_claiming_successful_delete(tmp_path):
    class MissingFile(Exception):
        status_code = 404
    class MissingFiles(Files):
        async def delete(self, file_id):
            raise MissingFile("fixture no such file")
    async with ObservationTransport(tmp_path / "transport", client=SimpleNamespace(files=MissingFiles())) as session:
        await session.prepare_input([image()])
    result = json.loads((tmp_path / "transport/summary.json").read_text())
    assert result["all_removed"] and not result["all_deleted"]
    assert result["files"][0]["confirmed_absent"]


@pytest.mark.asyncio
@pytest.mark.parametrize("count", [1234, 190000, None])
async def test_real_sdk_count_and_generation_share_file_ids_and_schemas(tmp_path, count):
    requests = []
    def handler(request):
        payload = json.loads(request.content)
        requests.append((request.url.path, payload))
        if request.url.path.endswith("/input_tokens"):
            if count is None:
                return httpx.Response(400, json={"error": {"message": "fixture count failure", "type": "invalid_request_error"}})
            return httpx.Response(200, json={"object": "response.input_tokens", "input_tokens": count})
        assert count == 1234
        return httpx.Response(200, json={"id": "resp_fixture", "object": "response", "created_at": 1700000000,
            "status": "completed", "model": "gpt-6-luna", "output": [{"id": "msg_fixture", "type": "message",
            "role": "assistant", "status": "completed", "content": [{"type": "output_text", "text": "{}", "annotations": []}]}],
            "usage": {"input_tokens": count, "output_tokens": 10, "total_tokens": count + 10,
                      "input_tokens_details": {"cached_tokens": 0}, "output_tokens_details": {"reasoning_tokens": 0}},
            "parallel_tool_calls": False, "tools": [], "tool_choice": "auto"})
    client = runtime._DAsyncOpenAI(api_key="fixture", max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    files = Files()
    settings = Settings(budget_usd=1)
    ledger = runtime.BudgetLedger(tmp_path / "budget.json", 1, settings)
    inner = runtime.OpenAIResponsesModel(model=settings.model, openai_client=client)
    model = runtime.RecordedModel(inner, settings, tmp_path / "phase", "observation", ledger, True, False)
    @function_tool
    def query_cp(channel: str = "all") -> str:
        """Read supplied measurements."""
        return "{}"
    try:
        async with ObservationTransport(tmp_path / "transport", client=SimpleNamespace(files=files)):
            async def generate():
                return await model.get_response(system_instructions="Observe appearances.",
                    input=[{"role": "user", "content": [{"type": "input_text", "text": "metadata"}, image()]}],
                    model_settings=replace(runtime._model_settings(settings, False), truncation="disabled"),
                    tools=[query_cp], output_schema=AgentOutputSchema(ObservationOutput), handoffs=[], tracing=ModelTracing.DISABLED)
            if count == 1234:
                await generate()
                await generate()  # Same entire payload: count is reused, generation is not.
            else:
                with pytest.raises(Exception, match="exceeds|fixture count failure"):
                    await generate()
    finally:
        await client.close()
    assert len(files.uploaded) == len(files.deleted) == 1
    assert all("data:image/" not in json.dumps(p) for _, p in requests)
    assert all(p["input"][0]["content"][1]["detail"] == "high" for _, p in requests)
    if count == 1234:
        assert len(requests) == 3
        counter, generated = requests[0][1], requests[1][1]
        for key in ("model", "input", "instructions", "tools", "text", "truncation", "reasoning"):
            assert counter[key] == generated[key]
        assert requests[2][1] == generated
        assert len(ledger.state["requests"]) == 2
        assert all(row["status"] == "complete" for row in ledger.state["requests"])
    else:
        assert len(requests) == 1 and ledger.state["requests"] == []


@pytest.mark.asyncio
async def test_four_real_sdk_phases_with_crop_tool_output(tmp_path, monkeypatch, case):
    from realclip_d.observation_pipeline import run_pipeline
    sample, view = case
    finals = phase_finals(view)
    schema_names = {"BaselineOutput": "baseline", "GuidanceEvidenceOutput": "guided",
                    "DescriptionDraft": "writer", "DescriptionReview": "reviewer"}
    requests, generation = [], []
    files = Files()
    def handler(request):
        payload = json.loads(request.content)
        requests.append((request.url.path, payload))
        if request.url.path.endswith("/input_tokens"):
            return httpx.Response(200, json={"object": "response.input_tokens", "input_tokens": 1000})
        properties = payload["text"]["format"]["schema"]["properties"]
        phase = ("baseline" if "reviewed_image_ids" in properties else "guided" if "hypothesis_assessments" in properties
                 else "writer" if "sentences" in properties else "reviewer")
        generation.append(phase)
        if generation == ["baseline"]:
            output = [{"id": "tool_crop", "type": "function_call", "status": "completed",
                       "call_id": "crop_call", "name": "inspect_image", "arguments": json.dumps({
                           "image_id": "treatment_DNA", "x": 0, "y": 0, "width": 5, "height": 5,
                           "question": "Inspect synthetic pixels.", "selection_reason": "Offline crop transport test."})}]
        else:
            output = [{"id": "msg_fixture", "type": "message", "role": "assistant", "status": "completed",
                       "content": [{"type": "output_text", "text": json.dumps(finals[phase]), "annotations": []}]}]
        return httpx.Response(200, json={"id": "resp_" + str(len(generation)), "object": "response",
            "created_at": 1700000000, "status": "completed", "model": "gpt-6-luna", "output": output,
            "usage": {"input_tokens": 1000, "output_tokens": 100, "total_tokens": 1100,
                      "input_tokens_details": {"cached_tokens": 0}, "output_tokens_details": {"reasoning_tokens": 0}},
            "parallel_tool_calls": False, "tools": [], "tool_choice": "auto"})
    def factory(settings):
        client = runtime._DAsyncOpenAI(api_key="fixture", max_retries=0,
                    http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
        return runtime.OpenAIResponsesModel(model=settings.model, openai_client=client), client
    monkeypatch.setattr(runtime, "_live_model", factory)
    settings = Settings(budget_usd=1, max_output_tokens=2048)
    ledger = runtime.BudgetLedger(tmp_path / "budget.json", 1, settings)
    async with ObservationTransport(tmp_path / "transport", client=SimpleNamespace(files=files)):
        output = await run_pipeline(sample, view, settings, tmp_path / "pipeline", "live", ledger)
    assert output.description and generation == ["baseline", "baseline", "guided", "writer", "reviewer"]
    assert len(files.uploaded) == len(files.deleted) == 2  # Four fixture originals share exact bytes, plus one new crop.
    def images(value):
        if isinstance(value, dict):
            if value.get("type") == "input_image": yield value
            for v in value.values(): yield from images(v)
        elif isinstance(value, list):
            for v in value: yield from images(v)
    generated = [p for path, p in requests if not path.endswith("/input_tokens")]
    assert len(list(images(generated[0]["input"]))) == 4
    assert len(list(images(generated[1]["input"]))) == 5
    assert len(list(images(generated[2]["input"]))) == 5
    assert not list(images(generated[3]["input"]))
    assert len(list(images(generated[4]["input"]))) == 5
    assert all("file_id" in row and "image_url" not in row for p in generated for row in images(p))
    assert all(row["status"] == "complete" for row in ledger.state["requests"])
