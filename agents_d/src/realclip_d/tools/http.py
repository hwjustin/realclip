"""Read-only official API transport, with auditable raw-response caching."""
from __future__ import annotations

import base64
import hashlib
import json
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

import httpx


@dataclass(frozen=True)
class HTTPRecord:
    url: str
    status_code: int
    body: bytes
    retrieved_at: str
    cache_ref: str

    def json(self) -> Any:
        return json.loads(self.body)

    @property
    def text(self) -> str:
        return self.body.decode("utf-8", errors="replace")


class TransportError(RuntimeError):
    """A failed retrieval, retaining its HTTP record when available."""

    def __init__(self, message: str, record: HTTPRecord | None = None):
        super().__init__(message)
        self.record = record


class PubChemLimiter:
    """One thread-safe PubChem request rate shared by a batch's clients."""

    def __init__(self, rate: float = 2.0, *, sleep=time.sleep, monotonic=time.monotonic):
        if not 0 < rate <= 5:
            raise ValueError("PubChem request rate must be positive and at most 5")
        self.rate, self.sleep, self.monotonic = rate, sleep, monotonic
        self._last: float | None = None
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = self.monotonic()
            if self._last is not None:
                remaining = 1.0 / self.rate - (now - self._last)
                if remaining > 0:
                    self.sleep(remaining)
            self._last = self.monotonic()


class CachedHTTP:
    """GET-only transport. All PubChem users sharing this object share a limiter.

    Cache entries preserve the exact response bytes (base64) and status. A cached
    error is not replayed during live retrieval; replay/offline mode never opens
    a network connection. Only 429, 500, 502, 503 and 504 responses are retried.
    """

    ALLOWED_HOSTS = frozenset({"pubchem.ncbi.nlm.nih.gov", "www.ebi.ac.uk", "europepmc.org"})

    def __init__(
        self,
        cache_dir: Path,
        *,
        client: httpx.Client | None = None,
        allow_network: bool = True,
        pubchem_requests_per_second: float = 2.0,
        max_retries: int = 2,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
        pubchem_limiter: PubChemLimiter | None = None,
    ):
        if pubchem_requests_per_second <= 0 or not 0 <= max_retries <= 2:
            raise ValueError("Positive PubChem rate and 0..2 retries are required")
        self.cache_dir = Path(cache_dir)
        self.client = client or httpx.Client(timeout=30.0, follow_redirects=False)
        self._owns_client = client is None
        self.allow_network = allow_network
        self.rate = pubchem_requests_per_second
        self.max_retries = max_retries
        self.sleep = sleep
        self.monotonic = monotonic
        self.pubchem_limiter = pubchem_limiter or PubChemLimiter(
            pubchem_requests_per_second, sleep=sleep, monotonic=monotonic)

    def close(self) -> None:
        if self._owns_client:
            self.client.close()

    def _throttle(self, url: str) -> None:
        if urlparse(url).hostname != "pubchem.ncbi.nlm.nih.gov":
            return
        self.pubchem_limiter.wait()

    @staticmethod
    def _retry_delay(value: str | None, attempt: int) -> float:
        if not value:
            return float(2**attempt)
        try:
            return max(0.0, float(value))
        except ValueError:
            try:
                return max(0.0, (parsedate_to_datetime(value) - datetime.now(timezone.utc)).total_seconds())
            except (ValueError, TypeError):
                return float(2**attempt)

    def get(self, url: str, *, params: dict[str, Any] | None = None) -> HTTPRecord:
        request = self.client.build_request("GET", url, params=params)
        if request.url.scheme != "https" or request.url.host not in self.ALLOWED_HOSTS:
            raise TransportError("Only HTTPS requests to D's official API hosts are allowed")
        canonical_url = str(request.url)
        key = hashlib.sha256(canonical_url.encode()).hexdigest()
        index_path = self.cache_dir / f"{key}.index.json"
        if index_path.is_file():
            try:
                snapshot_name = json.loads(index_path.read_text())["snapshot"]
            except (OSError, ValueError, KeyError) as exc:
                raise TransportError("Invalid cache index: " + str(exc)) from exc
            # Never follow an injected path outside the cache directory.
            if not isinstance(snapshot_name, str) or not snapshot_name.endswith(".json") or Path(snapshot_name).name != snapshot_name:
                raise TransportError("Invalid cache snapshot reference")
            path = self.cache_dir / snapshot_name
            try:
                raw_snapshot = path.read_bytes()
                if hashlib.sha256(raw_snapshot).hexdigest() != path.stem:
                    raise ValueError("Snapshot digest mismatch")
                cached = json.loads(raw_snapshot)
                raw_body = base64.b64decode(cached["body_base64"], validate=True)
                if cached["url"] != canonical_url or hashlib.sha256(raw_body).hexdigest() != cached["body_sha256"]:
                    raise ValueError("Response digest or URL mismatch")
            except (OSError, ValueError, KeyError) as exc:
                raise TransportError("Invalid raw response cache: " + str(exc)) from exc
            record = HTTPRecord(
                url=cached["url"], status_code=cached["status_code"],
                body=raw_body,
                retrieved_at=cached["retrieved_at"], cache_ref=str(path.resolve()),
            )
            if record.status_code < 400 or not self.allow_network:
                return record
        if not self.allow_network:
            raise TransportError(f"Offline cache miss: {canonical_url}")
        for attempt in range(self.max_retries + 1):
            self._throttle(canonical_url)
            try:
                response = self.client.send(request, follow_redirects=False)
            except httpx.TransportError as exc:
                if attempt == self.max_retries:
                    raise TransportError(str(exc)) from exc
                self.sleep(float(2**attempt))
                continue
            timestamp = datetime.now(timezone.utc).isoformat()
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            serialized = {
                "url": canonical_url, "status_code": response.status_code,
                "retrieved_at": timestamp, "headers": dict(response.headers),
                "body_base64": base64.b64encode(response.content).decode("ascii"),
                "body_sha256": hashlib.sha256(response.content).hexdigest(),
            }
            serialized_text = json.dumps(serialized, ensure_ascii=False, sort_keys=True)
            snapshot_id = hashlib.sha256(serialized_text.encode()).hexdigest()
            path = self.cache_dir / f"{snapshot_id}.json"
            record = HTTPRecord(canonical_url, response.status_code, response.content, timestamp, str(path.resolve()))
            temporary = path.with_suffix(f".{threading.get_ident()}.tmp")
            temporary.write_text(serialized_text)
            temporary.replace(path)
            index_temp = index_path.with_suffix(f".{threading.get_ident()}.tmp")
            index_temp.write_text(json.dumps({"snapshot": path.name}))
            index_temp.replace(index_path)
            if response.status_code not in {429, 500, 502, 503, 504} or attempt == self.max_retries:
                return record
            delay = self._retry_delay(response.headers.get("Retry-After"), attempt)
            if delay > 60:
                return record  # Do not violate Retry-After with an early retry.
            self.sleep(delay)
        raise AssertionError("Unreachable retry state")
