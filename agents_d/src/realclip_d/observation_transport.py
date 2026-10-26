"""Opt-in, run-scoped image transport; no changes to prompts or image pixels.

Each distinct byte sequence is uploaded once, referenced by every phase and
counter, and deleted on exit (with one-hour expiry as a cleanup backstop).
Counts are reused only for an identical, complete converted count request.
The context variable isolates simultaneous sample threads and async runs.
"""
from __future__ import annotations

import asyncio
import base64
from contextvars import ContextVar
import hashlib
import json
from pathlib import Path
import time
from typing import Any

from .storage import write_json


_CURRENT: ContextVar[ObservationTransport | None] = ContextVar("observation_transport", default=None)


def current_transport() -> ObservationTransport | None:
    return _CURRENT.get()


class ObservationTransport:
    def __init__(self, directory: Path, *, client=None, upload_concurrency: int = 1,
                 timeout: float = 600, file_attempts: int = 3):
        if not 1 <= upload_concurrency <= 16:
            raise ValueError("Image uploads require 1..16 concurrent requests")
        if not 1 <= file_attempts <= 3:
            raise ValueError("File requests require 1..3 attempts")
        self.directory = Path(directory)
        self.client = client
        self.owns_client = client is None
        self.timeout = timeout
        self.parallelism = upload_concurrency
        self.file_attempts = file_attempts
        self.upload_error: BaseException | None = None
        self.files: dict[str, dict] = {}
        self.urls: dict[str, str] = {}
        self.counts: dict[str, int] = {}
        self.count_lock = asyncio.Lock()
        self.upload_lock = asyncio.Lock()
        self.semaphore = asyncio.Semaphore(upload_concurrency)
        self.cleanup_semaphore = asyncio.Semaphore(1)
        self.events: list[dict] = []
        self.started = time.monotonic()

    async def __aenter__(self):
        if _CURRENT.get() is not None:
            raise ValueError("Observation transport contexts cannot be nested")
        self.directory.mkdir(parents=True, exist_ok=True)
        if any(self.directory.iterdir()):
            raise ValueError("Transport requires a fresh directory; expired IDs must not be replayed")
        if self.client is None:
            from .agent_runtime import _DAsyncOpenAI
            self.client = _DAsyncOpenAI(max_retries=0, timeout=self.timeout)
        self.token = _CURRENT.set(self)
        self.save("running")
        return self

    def event(self, kind: str, **values):
        row = {"event": kind, "elapsed_seconds": time.monotonic() - self.started, **values}
        self.events.append(row)
        with (self.directory / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")

    def save(self, status: str):
        write_json(self.directory / "summary.json", {
            "status": status, "expires_after_seconds": 3600,
            "upload_concurrency": self.parallelism, "files": list(self.files.values()),
            "cleanup_concurrency": 1,
            "file_max_attempts": self.file_attempts,
            "upload_count": len(self.files),
            "uploaded_bytes": sum(row["bytes"] for row in self.files.values()),
            "all_deleted": all(row.get("deleted") is True for row in self.files.values()),
            "all_removed": all(row.get("deleted") is True or row.get("confirmed_absent") is True
                               for row in self.files.values()),
            "count_requests": sum(row["event"] == "count" and not row["cache_hit"] for row in self.events),
            "count_cache_hits": sum(row["event"] == "count" and row["cache_hit"] for row in self.events),
            "elapsed_seconds": time.monotonic() - self.started})

    async def _upload(self, url: str, url_hash: str):
        header, encoded = url.split(",", 1)
        mime = header.removeprefix("data:").removesuffix(";base64")
        extensions = {"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp", "image/gif": ".gif"}
        if mime not in extensions or not header.endswith(";base64"):
            raise ValueError("Only supported base64 image inputs can be uploaded")
        data = base64.b64decode(encoded, validate=True)
        checksum = hashlib.sha256(data).hexdigest()
        if checksum in self.files:
            self.urls[url_hash] = self.files[checksum]["file_id"]
            return
        began = time.monotonic()
        async with self.semaphore:
            if self.upload_error is not None:
                self.event("upload_skipped_after_failure", sha256=checksum)
                return
            for attempt in range(1, self.file_attempts + 1):
                try:
                    remote = await self.client.files.create(file=(checksum + extensions[mime], data, mime),
                        purpose="vision", expires_after={"anchor": "created_at", "seconds": 3600})
                    break
                except BaseException as error:
                    retry = attempt < self.file_attempts and self._transient(error)
                    self.event("upload_retry" if retry else "upload_failed", sha256=checksum,
                               attempt=attempt, error_type=type(error).__name__, error=str(error))
                    if not retry:
                        self.upload_error = error
                        raise
                    # An uncertain successful upload might leave an extra file;
                    # every attempt has one-hour expiry. No generation is retried.
                    await asyncio.sleep(attempt)
        self.files[checksum] = {"sha256": checksum, "bytes": len(data), "mime": mime,
                               "file_id": remote.id, "deleted": False}
        self.urls[url_hash] = remote.id
        self.event("upload", sha256=checksum, file_id=remote.id, bytes=len(data),
                   duration_seconds=time.monotonic() - began)
        self.save("running")

    async def prepare_input(self, inputs: Any) -> Any:
        """Replace only input_image.data URLs, preserving order, metadata and detail."""
        pending: dict[str, str] = {}

        def scan(value):
            if isinstance(value, dict):
                if value.get("type") == "input_image":
                    url = value.get("image_url")
                    if isinstance(url, str) and url.startswith("data:image/"):
                        key = hashlib.sha256(url.encode()).hexdigest()
                        if key not in self.urls:
                            pending[key] = url
                    elif value.get("file_id") and value["file_id"] not in self.urls.values():
                        raise ValueError("Image file ID is outside this transport run")
                for item in value.values():
                    scan(item)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    scan(item)

        # One sample normally runs sequentially. The lock also prevents duplicate
        # uploads if two callers share the session at once.
        began = time.monotonic()
        async with self.upload_lock:
            scan(inputs)
            if pending:
                # Deduplicate bytes even when the same bytes have MIME aliases.
                by_bytes: dict[str, list[tuple[str, str]]] = {}
                for key, url in pending.items():
                    checksum = hashlib.sha256(base64.b64decode(url.split(",", 1)[1], validate=True)).hexdigest()
                    by_bytes.setdefault(checksum, []).append((key, url))
                uploads = [self._upload(items[0][1], items[0][0]) for items in by_bytes.values()]
                results = await asyncio.gather(*uploads, return_exceptions=True)
                self.event("prepare_uploads", duration_seconds=time.monotonic() - began,
                           distinct_images=len(uploads))
                for checksum, items in by_bytes.items():
                    if checksum in self.files:
                        for key, _ in items:
                            self.urls[key] = self.files[checksum]["file_id"]
                for result in results:
                    if isinstance(result, BaseException):
                        raise result
                if self.upload_error is not None:
                    raise self.upload_error

        def replace(value):
            if isinstance(value, dict):
                if value.get("type") == "input_image" and isinstance(value.get("image_url"), str):
                    url = value["image_url"]
                    if url.startswith("data:image/"):
                        return {**{k: replace(v) for k, v in value.items() if k != "image_url"},
                                "file_id": self.urls[hashlib.sha256(url.encode()).hexdigest()]}
                return {k: replace(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [replace(v) for v in value]
            return value

        result = replace(inputs)
        self.event("input_prepared", uploaded_new=len(pending),
                   duration_seconds=time.monotonic() - began)
        return result

    async def count(self, payload: dict, counter):
        # Exact full-payload key: never add image token estimates to text counts,
        # nor reuse counts across changed history, schemas, tools or models.
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                             default=lambda value: value.model_dump(mode="json")).encode()
        checksum = hashlib.sha256(encoded).hexdigest()
        began = time.monotonic()
        async with self.count_lock:
            hit = checksum in self.counts
            if not hit:
                result = await counter(**payload)
                if type(result.input_tokens) is not int or result.input_tokens < 0:
                    raise ValueError("Token counter did not return a valid input count")
                self.counts[checksum] = result.input_tokens
            count = self.counts[checksum]
        self.event("count", payload_sha256=checksum, encoded_bytes=len(encoded),
                   cache_hit=hit, input_tokens=count, duration_seconds=time.monotonic() - began)
        return count, hit

    @staticmethod
    def _transient(error):
        from openai import APIConnectionError
        return isinstance(error, APIConnectionError) or getattr(error, "status_code", None) in {429, 500, 502, 503, 504}

    async def __aexit__(self, exc_type, exc, tb):
        _CURRENT.reset(self.token)

        async def delete(row):
            async with self.cleanup_semaphore:
                for attempt in range(1, self.file_attempts + 1):
                    try:
                        result = await self.client.files.delete(row["file_id"])
                        row["deleted"] = result.deleted is True
                        break
                    except Exception as error:
                        if getattr(error, "status_code", None) == 404:
                            row["confirmed_absent"] = True
                            break
                        if attempt < self.file_attempts and self._transient(error):
                            self.event("cleanup_retry", file_id=row["file_id"], attempt=attempt,
                                       error_type=type(error).__name__)
                            await asyncio.sleep(attempt)
                        else:
                            row["cleanup_error"] = f"{type(error).__name__}: {error}"
                            break
            self.save("cleaning_up")

        try:
            await asyncio.gather(*(delete(row) for row in self.files.values()))
        finally:
            self.save("failed" if exc_type else "complete")
            if self.owns_client:
                await self.client.close()
