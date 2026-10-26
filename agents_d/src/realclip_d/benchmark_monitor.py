"""Local process/resource and HTTP telemetry; never records request secrets."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
import importlib
import os
from pathlib import Path
import resource
import threading
import time

import httpx

from .storage import write_json


def _now():
    return datetime.now(timezone.utc).isoformat()


def _proc_values(path):
    return {line.split()[0].rstrip(":"): line.split()[1:]
            for line in Path(path).read_text().splitlines() if len(line.split()) >= 2}


class RunMonitor:
    """Use in a dedicated batch process; sample resources and observe HTTP sends.

    HTTP wrappers preserve the original calls, responses, errors and retry policy.
    Only response rate-limit headers are retained; no request headers or bodies.
    """
    def __init__(self, directory: Path, interval: float = .5):
        self.directory = Path(directory)
        self.interval = interval
        self._lock = threading.Lock()
        self._done = threading.Event()
        self.samples, self.http_events, self.errors = [], [], []

    def _sample(self):
        now = time.monotonic()
        usage = resource.getrusage(resource.RUSAGE_SELF)
        status, memory = _proc_values("/proc/self/status"), _proc_values("/proc/meminfo")
        cpu = list(map(int, Path("/proc/stat").read_text().splitlines()[0].split()[1:9]))
        vm = _proc_values("/proc/vmstat")
        cpu_seconds = usage.ru_utime + usage.ru_stime
        row = {"timestamp": _now(), "elapsed_seconds": now - self._started,
               "process_cpu_seconds": cpu_seconds - self._initial_cpu,
               "process_rss_bytes": int(status["VmRSS"][0]) * 1024,
               "process_peak_rss_bytes": usage.ru_maxrss * 1024,
               "threads": int(status["Threads"][0]),
               "system_mem_total_bytes": int(memory["MemTotal"][0]) * 1024,
               "system_mem_available_bytes": int(memory["MemAvailable"][0]) * 1024,
               "system_swap_used_bytes": (int(memory["SwapTotal"][0]) - int(memory["SwapFree"][0])) * 1024,
               "system_swap_out_pages": int(vm.get("pswpout", [0])[0]),
               "system_cpu_total_ticks": sum(cpu), "system_cpu_idle_ticks": cpu[3] + cpu[4]}
        if self.samples:
            before = self.samples[-1]
            elapsed = row["elapsed_seconds"] - before["elapsed_seconds"]
            row["process_cpu_percent_one_core"] = 100 * (row["process_cpu_seconds"] - before["process_cpu_seconds"]) / elapsed
            total = row["system_cpu_total_ticks"] - before["system_cpu_total_ticks"]
            idle = row["system_cpu_idle_ticks"] - before["system_cpu_idle_ticks"]
            row["system_cpu_percent"] = 100 * (total - idle) / total if total > 0 else 0
        self.samples.append(row)
        with (self.directory / "resources.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        write_json(self.directory / "resource_latest.json", row)

    def _watch(self):
        while not self._done.wait(self.interval):
            try:
                self._sample()
            except Exception as error:
                self.errors.append(f"{type(error).__name__}: {error}")

    def _http(self, request, began, response=None, error=None):
        headers = {}
        if response is not None:
            for key, value in response.headers.items():
                if key.lower().startswith("x-ratelimit-") or key.lower() in {"retry-after", "x-request-id"}:
                    headers[key.lower()] = value
        row = {"timestamp": _now(), "elapsed_seconds": time.monotonic() - self._started,
               "duration_seconds": time.monotonic() - began, "thread": threading.current_thread().name,
               "method": request.method, "host": request.url.host, "path": request.url.path,
               "status_code": response.status_code if response is not None else None,
               "response_limit_headers": headers,
               "transport_error_type": type(error).__name__ if error is not None else None}
        with self._lock:
            self.http_events.append(row)
            with (self.directory / "http_events.jsonl").open("a") as stream:
                stream.write(json.dumps(row) + "\n")

    def __enter__(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        if any((self.directory / name).exists() for name in ("resources.jsonl", "http_events.jsonl")):
            raise ValueError("Monitoring output already exists; use a fresh benchmark directory")
        self._started = time.monotonic()
        usage = resource.getrusage(resource.RUSAGE_SELF)
        self._initial_cpu = usage.ru_utime + usage.ru_stime
        self._sample()
        monitor = self
        def sync_wrapper(original):
            def send(client, request, *args, **kwargs):
                began = time.monotonic()
                try:
                    response = original(client, request, *args, **kwargs)
                except BaseException as error:
                    monitor._http(request, began, error=error)
                    raise
                monitor._http(request, began, response=response)
                return response
            return send
        def async_wrapper(original):
            async def send(client, request, *args, **kwargs):
                began = time.monotonic()
                try:
                    response = await original(client, request, *args, **kwargs)
                except BaseException as error:
                    monitor._http(request, began, error=error)
                    raise
                monitor._http(request, began, response=response)
                return response
            return send
        modules = [httpx]
        # OpenAI 3.x may use httpx2; database tools continue to use httpx.
        try:
            modules.append(importlib.import_module("httpx2"))
        except ImportError:
            pass
        self._patches = []
        for module in modules:
            for cls, wrapper in ((module.Client, sync_wrapper), (module.AsyncClient, async_wrapper)):
                if any(old_cls is cls for old_cls, _ in self._patches):
                    continue
                original = cls.send
                self._patches.append((cls, original))
                cls.send = wrapper(original)
        self._thread = threading.Thread(target=self._watch, name="resource-monitor", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._done.set()
        self._thread.join()
        for cls, original in reversed(self._patches):
            cls.send = original
        self._sample()
        elapsed = time.monotonic() - self._started
        cpus = os.cpu_count() or 1
        usage = resource.getrusage(resource.RUSAGE_SELF)
        cpu_seconds = usage.ru_utime + usage.ru_stime - self._initial_cpu
        one_core_peak = max((row.get("process_cpu_percent_one_core", 0) for row in self.samples), default=0)
        self.summary = {"elapsed_seconds": elapsed, "logical_cpus": cpus,
                       "sampling_interval_seconds": self.interval, "samples": len(self.samples),
                       "process_cpu_seconds": cpu_seconds,
                       "process_cpu_percent_one_core_avg": 100 * cpu_seconds / elapsed,
                       "process_cpu_percent_machine_avg": 100 * cpu_seconds / elapsed / cpus,
                       "process_cpu_percent_one_core_peak": one_core_peak,
                       "process_cpu_percent_machine_peak": one_core_peak / cpus,
                       "process_peak_rss_bytes": usage.ru_maxrss * 1024,
                       "process_sampled_peak_rss_bytes": max(row["process_rss_bytes"] for row in self.samples),
                       "system_mem_available_min_bytes": min(row["system_mem_available_bytes"] for row in self.samples),
                       "system_cpu_percent_peak": max(row.get("system_cpu_percent", 0) for row in self.samples),
                       "system_swap_out_pages_delta": self.samples[-1]["system_swap_out_pages"] - self.samples[0]["system_swap_out_pages"],
                       "peak_threads": max(row["threads"] for row in self.samples),
                       "http_requests_observed": len(self.http_events),
                       "http_429_by_host": dict(Counter(row["host"] for row in self.http_events if row["status_code"] == 429)),
                       "http_503_by_host": dict(Counter(row["host"] for row in self.http_events if row["status_code"] == 503)),
                       "transport_errors_by_host": dict(Counter(row["host"] for row in self.http_events if row["transport_error_type"])),
                       "monitor_errors": self.errors,
                       "instrumented_http_clients": [cls.__module__ + "." + cls.__name__ for cls, _ in self._patches],
                       "scope": "Research Python process including all threads; system metrics describe the visible Linux/WSL environment. Peak RSS is process lifetime high-water mark. CPU one-core percent can exceed 100; machine percent divides by logical CPU count."}
        write_json(self.directory / "resource_summary.json", self.summary)
