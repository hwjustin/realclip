"""Vision budgeting exercises actual SDK conversion over local HTTP only."""
import asyncio
import json
from dataclasses import replace

import httpx
import pytest
from agents import ModelTracing, AgentOutputSchema, function_tool

from realclip_d import agent_runtime as runtime
from realclip_d.config import Settings
from realclip_d.observation_contracts import ObservationOutput


@pytest.mark.parametrize("count,error", [(1234, None), (190000, "oversize"), (None, "count_failure")])
def test_count_precedes_generation_and_preserves_images(tmp_path, count, error):
    requests = []

    def handler(request):
        payload = json.loads(request.content)
        requests.append((request.url.path, payload))
        if request.url.path.endswith("/input_tokens"):
            if count is None:
                return httpx.Response(400, json={"error": {"message": "fixture count failure", "type": "invalid_request_error"}})
            return httpx.Response(200, json={"object": "response.input_tokens", "input_tokens": count})
        assert count == 1234, "Generation must not follow a failed or oversized count"
        return httpx.Response(200, json={
            "id": "resp_fixture", "object": "response", "created_at": 1700000000,
            "status": "completed", "model": "gpt-6-luna",
            "output": [{"id": "msg_fixture", "type": "message", "role": "assistant",
                        "status": "completed", "content": [{"type": "output_text", "text": "{}", "annotations": []}]}],
            "usage": {"input_tokens": count, "output_tokens": 10, "total_tokens": count + 10,
                      "input_tokens_details": {"cached_tokens": 0}, "output_tokens_details": {"reasoning_tokens": 0}},
            "parallel_tool_calls": False, "tools": [], "tool_choice": "auto"})

    async def run():
        client = runtime._DAsyncOpenAI(api_key="fixture", max_retries=0,
                    http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
        inner = runtime.OpenAIResponsesModel(model="gpt-6-luna", openai_client=client)
        settings = Settings(budget_usd=1)
        ledger = runtime.BudgetLedger(tmp_path / "budget.json", 1, settings)
        recorded = runtime.RecordedModel(inner, settings, tmp_path, "observation", ledger, True, False)
        # Much larger than the text bound, even though this fixture's vision
        # count is small. The provider mock validates transport, not image pixels.
        image = "data:image/png;base64," + "AAAA" * 100000
        inputs = [{"role": "user", "content": [
            {"type": "input_text", "text": "Inspect all supplied fields."},
            {"type": "input_image", "image_url": image, "detail": "high"}]}]
        @function_tool
        def query_cp(channel: str = "all") -> str:
            """Read measurements for the supplied channel."""
            return "{}"
        try:
            await recorded.get_response(system_instructions="Observe appearances.", input=inputs,
                model_settings=replace(runtime._model_settings(settings, False), truncation="disabled"),
                tools=[query_cp], output_schema=AgentOutputSchema(ObservationOutput),
                handoffs=[], tracing=ModelTracing.DISABLED)
        finally:
            await client.close()
            if error:
                assert ledger.state["requests"] == []
                assert not ledger.state["blocked"]
            else:
                assert ledger.state["requests"][0]["status"] == "complete"
        return image

    if error:
        with pytest.raises(Exception, match="exceeds|fixture count failure"):
            asyncio.run(run())
        assert len(requests) == 1
    else:
        image = asyncio.run(run())
        assert len(requests) == 2
        counter, generated = requests[0][1], requests[1][1]
        for key in ("model", "input", "instructions", "tools", "text", "truncation", "reasoning"):
            assert counter[key] == generated[key]
        assert counter["input"][0]["content"][1]["image_url"] == image
        assert counter["truncation"] == "disabled"
    receipt = json.loads((tmp_path / "input_token_counts/001.json").read_text())
    assert receipt["status"] == ("failed" if count is None else "counted")
    assert "base64" not in json.dumps(receipt)
