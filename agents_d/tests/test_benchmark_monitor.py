"""Exercise monitoring with local HTTP transports and actual process samples."""
import asyncio
import json
import time

import httpx
import pytest

from realclip_d.benchmark_monitor import RunMonitor
from realclip_d.storage import read_json
from realclip_d.tools.http import CachedHTTP


def test_monitor_captures_retry_rate_headers_without_credentials(tmp_path):
    original = httpx.Client.send
    calls = []
    def respond(request):
        calls.append(request)
        return httpx.Response(429 if len(calls) == 1 else 200, json={"ok": True},
                              headers={"retry-after": "0", "x-ratelimit-remaining-tokens": "1234",
                                       "set-cookie": "must-not-be-recorded"})
    with RunMonitor(tmp_path / "monitor", interval=.01) as monitor:
        with httpx.Client(transport=httpx.MockTransport(respond), headers={"Authorization": "Bearer private-test-key"}) as client:
            transport = CachedHTTP(tmp_path / "cache", client=client, sleep=lambda _: None)
            assert transport.get("https://www.ebi.ac.uk/test?token=must-not-be-recorded").status_code == 200
        time.sleep(.03)
    assert httpx.Client.send is original
    assert monitor.summary["http_429_by_host"] == {"www.ebi.ac.uk": 1}
    assert monitor.summary["http_requests_observed"] == 2
    assert monitor.summary["samples"] >= 2 and monitor.summary["process_peak_rss_bytes"] > 0
    assert monitor.summary["system_mem_available_min_bytes"] > 0
    text = (tmp_path / "monitor/http_events.jsonl").read_text()
    assert "private-test-key" not in text and "must-not-be-recorded" not in text
    assert "1234" in text


def test_async_errors_and_restore_preserve_original_behavior(tmp_path):
    original = httpx.AsyncClient.send
    async def run():
        def respond(request):
            raise httpx.ConnectError("Synthetic connection failure", request=request)
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            await client.get("https://api.openai.com/v1/responses")
    with pytest.raises(httpx.ConnectError):
        with RunMonitor(tmp_path, interval=.01) as monitor:
            asyncio.run(run())
    assert httpx.AsyncClient.send is original
    assert monitor.summary["transport_errors_by_host"] == {"api.openai.com": 1}
    assert read_json(tmp_path / "resource_summary.json")["monitor_errors"] == []


def test_async_response_limit_header_is_observed(tmp_path):
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(
            200, json={"ok": True}, headers={"x-ratelimit-limit-project-tokens": "50000"}))) as client:
            await client.post("https://api.openai.com/v1/responses", json={"secret": "not-logged"})
    with RunMonitor(tmp_path) as monitor:
        asyncio.run(run())
    event = json.loads((tmp_path / "http_events.jsonl").read_text())
    assert event["response_limit_headers"]["x-ratelimit-limit-project-tokens"] == "50000"
    assert "not-logged" not in (tmp_path / "http_events.jsonl").read_text()
    assert monitor.summary["http_requests_observed"] == 1


def test_installed_openai_sdk_native_transport_captures_429_and_restores(tmp_path):
    from openai import AsyncOpenAI, RateLimitError
    import httpx2
    original = httpx2.AsyncClient.send
    async def run():
        transport = httpx2.MockTransport(lambda request: httpx2.Response(
            429, json={"error": {"message": "Synthetic limit", "type": "rate_limit_error", "code": "rate_limit_exceeded"}},
            headers={"retry-after": "1", "x-ratelimit-remaining-tokens": "0"}))
        async with AsyncOpenAI(api_key="synthetic-offline-key", max_retries=0,
                               http_client=httpx2.AsyncClient(transport=transport)) as client:
            await client.models.retrieve("gpt-6-luna")
    with RunMonitor(tmp_path) as monitor:
        with pytest.raises(RateLimitError):
            asyncio.run(run())
    assert httpx2.AsyncClient.send is original
    assert monitor.summary["http_requests_observed"] == 1
    assert monitor.summary["http_429_by_host"] == {"api.openai.com": 1}
    row = json.loads((tmp_path / "http_events.jsonl").read_text())
    assert row["response_limit_headers"]["x-ratelimit-remaining-tokens"] == "0"
    assert "synthetic-offline-key" not in (tmp_path / "http_events.jsonl").read_text()
