"""Verified, resumable public-TIFF downloads; Python standard library only."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from contextlib import contextmanager
import gzip
import hashlib
from http.client import HTTPException
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import sys
import time
from urllib.error import HTTPError
from urllib.parse import urlsplit, unquote
from urllib.request import Request, HTTPRedirectHandler, build_opener

CHANNELS = ("DNA", "RNA", "ER", "Mito", "AGP")
HOST = "cellpainting-gallery.s3.amazonaws.com"


def stable_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def hash_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_path(root, relative):
    root = Path(root).resolve()
    if (not isinstance(relative, str) or not relative or "\\" in relative or ":" in relative
            or "//" in relative):
        raise ValueError("Unsafe relative path")
    parts = PurePosixPath(relative)
    if parts.is_absolute() or any(part in (".", "..") for part in relative.split("/")):
        raise ValueError(f"Unsafe relative path: {relative}")
    target = root.joinpath(*parts.parts)
    try:
        target.resolve().relative_to(root)
    except ValueError:
        raise ValueError(f"Path escapes data root: {relative}")
    current = root
    for part in parts.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"Symlinks are not allowed: {relative}")
    return target


def _sha(value):
    return isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value) is not None


def validate_url(url):
    if not isinstance(url, str):
        raise ValueError("Missing public image URL")
    parsed = urlsplit(url)
    if (parsed.scheme != "https" or parsed.netloc != HOST or parsed.query or parsed.fragment
            or not parsed.path.startswith("/cpg0016-jump/")
            or any(part in (".", "..") for part in unquote(parsed.path).split("/"))
            or "\\" in unquote(parsed.path)):
        raise ValueError(f"Only unsigned public Cell Painting Gallery TIFF URLs are allowed: {url}")


def _read_gzip(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        text = stream.read(128 * 1024 * 1024 + 1)
    if len(text) > 128 * 1024 * 1024:
        raise ValueError("Unexpectedly large metadata document")
    return text


def load_bundle(manifest_dir):
    """Authenticate the exact portable dataset and its complete URL mapping."""
    directory = Path(manifest_dir).resolve()
    metadata = json.loads(safe_path(directory, "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("schema_version") != 1:
        raise ValueError("Unsupported download metadata schema")
    artifacts = metadata.get("artifacts", {})
    expected_artifacts = {"dataset.json.gz", "raw_downloads.jsonl.gz"}
    if "archive" in metadata:
        expected_artifacts.add("prepared_archive.json.gz")
    if set(artifacts) != expected_artifacts:
        raise ValueError("Both authenticated metadata artifacts are required")
    for name, record in artifacts.items():
        path = safe_path(directory, name)
        if (not path.is_file() or path.stat().st_size != record.get("size_bytes")
                or not _sha(record.get("sha256")) or hash_file(path) != record["sha256"]):
            raise ValueError(f"Metadata artifact checksum/size mismatch: {name}")
    manifest = json.loads(_read_gzip(directory / "dataset.json.gz"))
    if (manifest.get("manifest_id") != stable_hash({k: v for k, v in manifest.items() if k != "manifest_id"})
            or manifest.get("manifest_id") != metadata.get("manifest_id")
            or manifest.get("schema_version") != 1
            or manifest.get("format") != "realclip-h200-well-pairs-v1"
            or manifest.get("channels") != list(CHANNELS)):
        raise ValueError("Portable manifest identity/format mismatch")
    wanted, groups, ids, splits = {}, {}, set(), {"train": 0, "val": 0, "test": 0}
    samples = manifest.get("samples", [])
    if not samples:
        raise ValueError("Empty dataset manifest")
    for sample in samples:
        split, sid = sample.get("split"), sample.get("sample_id")
        if split not in splits or not isinstance(sid, str) or not sid or sid in ids:
            raise ValueError("Invalid/duplicate sample or split")
        ids.add(sid)
        splits[split] += 1
        text = sample.get("description")
        if not isinstance(text, str) or not text.strip() or hashlib.sha256(text.encode()).hexdigest() != sample.get("description_sha256"):
            raise ValueError("Description checksum mismatch")
        for group in (("plate", sample.get("source"), sample.get("plate")), ("parent", sample.get("parent_id"))):
            if any(not isinstance(value, str) or not value for value in group):
                raise ValueError("Missing held-out group identity")
            if groups.setdefault(group, split) != split:
                raise ValueError("Plate/parent split leakage")
        seen = set()
        for image in sample.get("images", []):
            sha, relative = image.get("sha256"), image.get("path")
            if not _sha(sha) or relative != f"images/{sha}.tif":
                raise ValueError("Invalid image content identity")
            key = (str(image.get("site")), image.get("channel"))
            if not key[0].isdigit() or key[1] not in CHANNELS or key in seen:
                raise ValueError("Invalid/duplicate site/channel image")
            seen.add(key)
            if wanted.setdefault(relative, sha) != sha:
                raise ValueError("Conflicting image identity")
        if not seen or seen != {(site, channel) for site, _ in seen for channel in CHANNELS}:
            raise ValueError("Every well site requires all five channels")
    rows = [json.loads(line) for line in _read_gzip(directory / "raw_downloads.jsonl.gz").splitlines() if line.strip()]
    entries = {}
    for row in rows:
        relative, sha, size = row.get("path"), row.get("sha256"), row.get("size_bytes")
        if (relative in entries or wanted.get(relative) != sha or not _sha(sha)
                or type(size) is not int or size < 1):
            raise ValueError("Invalid, duplicate, or unrelated download entry")
        validate_url(row.get("url"))
        entries[relative] = row
    if set(entries) != set(wanted):
        raise ValueError("Download index does not exactly cover the dataset images")
    if (metadata.get("counts") != {"wells": len(samples), "images": len(entries)}
            or metadata.get("split_counts") != splits
            or metadata.get("raw_bytes") != sum(row["size_bytes"] for row in entries.values())):
        raise ValueError("Download counts/bytes differ from authenticated metadata")
    return manifest, list(entries.values()), metadata


def load_prepared_archive(manifest_dir, manifest, training_entries, metadata):
    """Validate additional image-only wells without admitting them to training."""
    if "archive" not in metadata:
        raise ValueError("This bundle has no all-prepared archive; update the repository first")
    archive = json.loads(_read_gzip(safe_path(manifest_dir, "prepared_archive.json.gz")))
    payload = {k: v for k, v in archive.items() if k != "archive_id"}
    if (archive.get("schema_version") != 1 or archive.get("format") != "realclip-prepared-archive-v1"
            or archive.get("archive_id") != stable_hash(payload)
            or archive.get("archive_id") != metadata["archive"].get("archive_id")
            or archive.get("training_manifest_id") != manifest["manifest_id"]):
        raise ValueError("Prepared archive identity mismatch")
    wells = {s["well_id"] for s in manifest["samples"]}
    ids = {s["sample_id"] for s in manifest["samples"]}
    wanted, complete, incomplete = {}, 0, 0
    for sample in archive.get("samples", []):
        well, sid = sample.get("well_id"), sample.get("sample_id")
        if (not isinstance(well, str) or not well or well in wells
                or not isinstance(sid, str) or not sid or sid in ids
                or sample.get("training_eligible") is not False):
            raise ValueError("Archive sample duplicates or enters the training cohort")
        wells.add(well)
        ids.add(sid)
        status, text = sample.get("description_status"), sample.get("description")
        if status == "complete":
            if (not isinstance(text, str) or not text.strip()
                    or hashlib.sha256(text.encode()).hexdigest() != sample.get("description_sha256")):
                raise ValueError("Archival description checksum mismatch")
            complete += 1
        elif status == "incomplete" and text is None and sample.get("description_sha256") is None:
            incomplete += 1
        else:
            raise ValueError("Incomplete archival samples must not receive fabricated descriptions")
        seen = set()
        for image in sample.get("images", []):
            relative, sha = image.get("path"), image.get("sha256")
            key = (str(image.get("site")), image.get("channel"))
            if (not _sha(sha) or relative != f"images/{sha}.tif" or not key[0].isdigit()
                    or key[1] not in CHANNELS or key in seen):
                raise ValueError("Invalid archival image identity/site/channel")
            seen.add(key)
            row = {k: image[k] for k in ("path", "sha256", "size_bytes", "url")}
            if wanted.setdefault(relative, row) != row:
                raise ValueError("Conflicting archival image mapping")
        if not seen or seen != {(site, channel) for site, _ in seen for channel in CHANNELS}:
            raise ValueError("Every archival site requires all five channels")
    extra, training = {}, {e["path"]: e for e in training_entries}
    for row in archive.get("images", []):
        relative, size = row.get("path"), row.get("size_bytes")
        if relative in extra or relative in training or wanted.get(relative) != row or type(size) is not int or size < 1:
            raise ValueError("Archive download index must cover only archival images exactly once")
        validate_url(row.get("url"))
        extra[relative] = row
    if set(extra) != set(wanted):
        raise ValueError("Archive index does not cover every archival image")
    summary = metadata["archive"]
    if (summary.get("wells") != complete + incomplete or summary.get("images") != len(extra)
            or summary.get("complete_descriptions") != complete
            or summary.get("incomplete_descriptions") != incomplete
            or summary.get("raw_bytes") != sum(row["size_bytes"] for row in extra.values())):
        raise ValueError("Archive counts differ from the metadata")
    expected_all = {"wells": len(wells), "images": len(training) + len(extra),
                    "raw_bytes": metadata["raw_bytes"] + summary["raw_bytes"]}
    if metadata.get("all_prepared") != expected_all or archive.get("total_prepared_wells") != len(wells):
        raise ValueError("Prepared universe counts differ from the training/archive union")
    return archive, sorted(list(training.values()) + list(extra.values()), key=lambda x: x["path"])


def ensure_manifest(root, manifest, metadata, allow_upgrade=False):
    """Explicit compatible expansion, retaining an authenticated old manifest."""
    path = safe_path(root, "manifest.json")
    if not path.exists():
        atomic_json(path, manifest)
        return False
    existing = json.loads(path.read_text(encoding="utf-8"))
    if existing == manifest:
        return False
    if not allow_upgrade:
        raise ValueError("Data root has an older/different manifest; use --upgrade-manifest for a compatible expansion")
    old_id = existing.get("manifest_id")
    if (old_id != stable_hash({k: v for k, v in existing.items() if k != "manifest_id"})
            or old_id not in metadata.get("compatible_previous_manifest_ids", [])):
        raise ValueError("The previous dataset is not an authenticated compatible expansion ancestor")
    for key in ("schema_version", "format", "channels", "preprocessing"):
        if existing.get(key) != manifest.get(key):
            raise ValueError("Manifest upgrade changed the image preprocessing or format")
    old = {s["sample_id"]: s for s in existing["samples"]}
    new = {s["sample_id"]: s for s in manifest["samples"]}
    if len(old) != len(existing["samples"]) or any(new.get(sid) != sample for sid, sample in old.items()):
        raise ValueError("Manifest upgrade removed or changed an existing sample, caption, split or window")
    if any(sample["split"] != "train" for sid, sample in new.items() if sid not in old):
        raise ValueError("Manifest upgrade may add only training samples; original holdouts are frozen")
    backup = safe_path(root, f"manifest_history/{old_id}.json")
    backup.parent.mkdir(parents=True, exist_ok=True)
    if backup.exists():
        if json.loads(backup.read_text()) != existing:
            raise ValueError("Existing previous-manifest backup differs")
    else:
        atomic_json(backup, existing)
    atomic_json(path, manifest)
    return True


def atomic_json(path, value):
    path = Path(path)
    safe_path(path.parent, path.name)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


@contextmanager
def dataset_lock(root):
    """Kernel-managed Linux lock: interrupted jobs do not leave stale locks."""
    import fcntl
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock_path = safe_path(root, ".raw-download.lock")
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        if os.fstat(descriptor).st_nlink != 1:
            raise RuntimeError("Download lock must not be a hardlink to another file")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another download/benchmark/verification already uses this data root") from exc
        os.ftruncate(descriptor, 0)
        os.write(descriptor, str(os.getpid()).encode())
        yield
    finally:
        os.close(descriptor)


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, newurl):
        raise HTTPError(request.full_url, code, "Public TIFF redirects are not accepted", headers, fp)


def open_public(request, timeout):
    return build_opener(_NoRedirect()).open(request, timeout=timeout)


class IntegrityError(ValueError):
    pass


class DownloadError(RuntimeError):
    def __init__(self, message, network_bytes=0):
        super().__init__(message)
        self.network_bytes = network_bytes


def download_file(entry, root, retries=3, timeout=60, opener=None, verify_only=False):
    """Reuse only verified bytes; publish downloads only after size and SHA256 pass."""
    target = safe_path(root, entry["path"])
    size, expected = entry["size_bytes"], entry["sha256"]
    if target.is_file() and target.stat().st_size == size and hash_file(target) == expected:
        return {"cached": True, "network_bytes": 0, "size_bytes": size, "path": entry["path"]}
    if verify_only:
        raise DownloadError("Missing or corrupt local image")
    if target.exists() and not target.is_file():
        raise DownloadError("Image target is not a regular file")
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = safe_path(root, entry["path"] + ".partial")
    if partial.exists() and not partial.is_file():
        raise DownloadError("Partial target is not a regular file")
    if partial.exists() and partial.stat().st_nlink != 1:
        raise DownloadError("Partial image must not be a hardlink to another file")
    open_request = opener or open_public
    network_bytes = 0
    last_error = None
    for attempt in range(retries):
        try:
            # Revalidate paths after failures and before replacing any existing data.
            safe_path(root, entry["path"])
            safe_path(root, entry["path"] + ".partial")
            offset = partial.stat().st_size if partial.exists() else 0
            if offset >= size:
                if offset == size and hash_file(partial) == expected:
                    partial.replace(target)
                    return {"cached": False, "network_bytes": network_bytes, "size_bytes": size, "path": entry["path"]}
                partial.unlink()
                offset = 0
            headers = {"Accept-Encoding": "identity", "User-Agent": "realclip-public-tiff-downloader/1"}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            request = Request(entry["url"], headers=headers)
            with open_request(request, timeout=timeout) as response:
                status = response.status
                if status == 206:
                    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
                    if (not match or tuple(map(int, match.groups())) != (offset, size - 1, size)):
                        raise ValueError("Invalid resumed Content-Range")
                    mode, expected_body = "ab" if offset else "wb", size - offset
                elif status == 200:
                    mode, expected_body = "wb", size  # Server ignored Range: safely restart.
                else:
                    raise ValueError(f"Unexpected public TIFF HTTP status {status}")
                if response.headers.get("Content-Encoding", "identity").lower() not in ("", "identity"):
                    raise ValueError("Encoded responses cannot preserve original TIFF bytes")
                declared = response.headers.get("Content-Length")
                if declared is not None and int(declared) != expected_body:
                    raise ValueError("HTTP Content-Length differs from the frozen image size")
                received = 0
                with partial.open(mode) as stream:
                    while True:
                        block = response.read(1024 * 1024)
                        if not block:
                            break
                        network_bytes += len(block)
                        received += len(block)
                        if received > expected_body:
                            raise IntegrityError("Image response exceeded its frozen size")
                        stream.write(block)
                    stream.flush()
                    os.fsync(stream.fileno())
                if received != expected_body:
                    raise OSError("Incomplete TIFF response; retained partial bytes for retry")
            if partial.stat().st_size != size or hash_file(partial) != expected:
                raise IntegrityError("Downloaded TIFF SHA256/size mismatch")
            safe_path(root, entry["path"])
            partial.replace(target)
            return {"cached": False, "network_bytes": network_bytes, "size_bytes": size, "path": entry["path"]}
        except (OSError, ValueError, HTTPError, HTTPException) as exc:
            last_error = exc
            if isinstance(exc, IntegrityError) and partial.exists():
                partial.unlink()  # Only this job's invalid temporary image is discarded.
            if attempt + 1 < retries:
                time.sleep(min(2, 0.25 * 2**attempt))
    raise DownloadError(f"{type(last_error).__name__}: {last_error}", network_bytes)


def benchmark_entries(entries, count):
    count = min(count, len(entries))
    return [entries[index * len(entries) // count] for index in range(count)]


def run_download(manifest_dir, data_root, workers=16, benchmark=None, verify_only=False,
                 retries=3, timeout=60, max_failures=20, opener=None,
                 all_prepared=False, upgrade_manifest=False):
    if type(workers) is not int or not 1 <= workers <= 128:
        raise ValueError("workers must be an integer in [1,128]")
    if benchmark is not None and (type(benchmark) is not int or benchmark < 1 or verify_only):
        raise ValueError("benchmark must be positive and cannot be combined with verify-only")
    if type(retries) is not int or not 1 <= retries <= 10:
        raise ValueError("retries must be an integer in [1,10]")
    if not math.isfinite(timeout) or timeout <= 0 or type(max_failures) is not int or max_failures < 1:
        raise ValueError("Positive finite timeout and max-failures are required")
    manifest, entries, metadata = load_bundle(manifest_dir)
    archive = None
    if all_prepared:
        archive, entries = load_prepared_archive(manifest_dir, manifest, entries, metadata)
    selected = benchmark_entries(entries, benchmark) if benchmark is not None else entries
    root = Path(data_root).resolve()
    report_name = "raw_benchmark.json" if benchmark is not None else "raw_verify_report.json" if verify_only else "raw_download_report.json"
    with dataset_lock(root):
        upgraded = ensure_manifest(root, manifest, metadata, upgrade_manifest)
        if archive is not None:
            atomic_json(safe_path(root, "archive_samples.json"), archive)
        scope = "all_prepared" if all_prepared else "training"
        total_raw_bytes = metadata["all_prepared"]["raw_bytes"] if all_prepared else metadata["raw_bytes"]
        started, last_progress = time.monotonic(), 0
        cached = downloaded = network_bytes = completed_bytes = 0
        failures = []
        iterator = iter(selected)
        stopped = False
        atomic_json(safe_path(root, report_name), {"schema_version": 1, "status": "running",
                    "manifest_id": manifest["manifest_id"], "pid": os.getpid(),
                    "total_images": len(entries), "selected_images": len(selected),
                    "download_scope": scope})
        print(json.dumps({"event": "started", "mode": "benchmark" if benchmark else "verify" if verify_only else "download",
                          "selected_images": len(selected), "total_images": len(entries), "workers": workers,
                          "manifest_id": manifest["manifest_id"], "download_scope": scope,
                          "manifest_upgraded": upgraded}), flush=True)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            pending = {}

            def submit_one():
                try:
                    entry = next(iterator)
                except StopIteration:
                    return False
                future = pool.submit(download_file, entry, root, retries, timeout, opener, verify_only)
                pending[future] = entry
                return True

            for _ in range(workers):
                if not submit_one():
                    break
            while pending:
                done, _ = wait(pending, timeout=5, return_when=FIRST_COMPLETED)
                for future in done:
                    entry = pending.pop(future)
                    try:
                        result = future.result()
                        cached += int(result["cached"])
                        downloaded += int(not result["cached"])
                        network_bytes += result["network_bytes"]
                        completed_bytes += result["size_bytes"]
                    except Exception as exc:
                        network_bytes += getattr(exc, "network_bytes", 0)
                        failures.append({"path": entry["path"], "error": str(exc)})
                    if not verify_only and len(failures) >= max_failures:
                        stopped = True
                    if not stopped:
                        submit_one()
                elapsed = time.monotonic() - started
                if elapsed - last_progress >= 5 or not pending:
                    print(json.dumps({"event": "progress", "successful_images": cached + downloaded,
                                      "selected_images": len(selected), "cached_images": cached,
                                      "downloaded_images": downloaded, "failed_images": len(failures),
                                      "network_bytes": network_bytes, "elapsed_seconds": round(elapsed, 2)}), flush=True)
                    last_progress = elapsed
        elapsed = time.monotonic() - started
        successful = cached + downloaded
        complete = not failures and successful == len(selected)
        status = ("benchmark_complete" if benchmark is not None else "verified" if verify_only else "complete") if complete else "failed"
        report = {"schema_version": 1, "status": status, "manifest_id": manifest["manifest_id"],
                  "total_images": len(entries), "selected_images": len(selected),
                  "successful_images": successful, "cached_images": cached, "downloaded_images": downloaded,
                  "failed_images": len(failures), "not_attempted_images": len(selected)-successful-len(failures),
                  "verified_image_bytes": completed_bytes, "total_raw_bytes": total_raw_bytes,
                  "download_scope": scope, "training_images": metadata["counts"]["images"],
                  "training_wells": metadata["counts"]["wells"],
                  "total_wells": metadata["all_prepared"]["wells"] if all_prepared else metadata["counts"]["wells"],
                  "manifest_upgraded": upgraded,
                  "network_bytes": network_bytes, "elapsed_seconds": round(elapsed, 4),
                  "network_MB_per_second": round(network_bytes / elapsed / 1e6, 4) if network_bytes else None,
                  "cache_only": network_bytes == 0,
                  "speed_note": "Actual received HTTP body bytes / wall time, including checksum, disk and retry overhead; cached files do not count as network traffic.",
                  "workers": workers, "failures": failures, "report_path": str(root/report_name)}
        atomic_json(safe_path(root, report_name), report)
        return report


def main(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, default=Path(__file__).resolve().parents[2]/"manifests")
    parser.add_argument("--workers", type=int, default=16)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--benchmark", type=int, metavar="IMAGES", help="Download only evenly sampled IMAGES; full download reuses them")
    mode.add_argument("--verify-only", action="store_true", help="Verify every image without any network request")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--max-failures", type=int, default=20, help="Stop scheduling new downloads after this many exhausted failures")
    parser.add_argument("--all-prepared", action="store_true", help="Also download archival prepared wells; only the approved cohort enters training")
    parser.add_argument("--upgrade-manifest", action="store_true", help="Allow an authenticated expansion that leaves all prior samples and holdouts unchanged")
    args = parser.parse_args(argv)
    try:
        report = run_download(args.manifest_dir, args.data_root, args.workers, args.benchmark,
                              args.verify_only, args.retries, args.timeout, args.max_failures,
                              all_prepared=args.all_prepared, upgrade_manifest=args.upgrade_manifest)
    except (OSError, ValueError, RuntimeError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}), file=sys.stderr, flush=True)
        return 1
    summary = {key: value for key, value in report.items() if key != "failures"}
    summary["failure_examples"] = report["failures"][:5]
    print(json.dumps(summary, indent=2), flush=True)
    return int(report["status"] == "failed")


if __name__ == "__main__":
    raise SystemExit(main())
