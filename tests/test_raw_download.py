from __future__ import annotations

import gzip
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import socket
import threading
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

import pytest

from realclip_h200 import raw_download as raw


def fixture_bundle(directory, wells=3):
    directory.mkdir()
    samples, entries, bodies = [], [], {}
    for index in range(wells):
        text = f"Original morphology description {index}."
        images = []
        for channel in raw.CHANNELS:
            body = (f"raw TIFF bytes {index} {channel} ".encode() * 1600)
            digest = hashlib.sha256(body).hexdigest()
            relative = f"images/{digest}.tif"
            url = f"https://{raw.HOST}/cpg0016-jump/source_{index}/{channel}.tif"
            entries.append({"path": relative, "sha256": digest, "size_bytes": len(body), "url": url})
            images.append({"path": relative, "sha256": digest, "site": "0", "channel": channel, "shape": [2, 2]})
            bodies[urlsplit(url).path] = body
        samples.append({"sample_id": f"well{index}", "well_id": f"source{index}|plate|A01",
                        "source": f"source{index}", "plate": f"plate{index}", "parent_id": f"parent{index}",
                        "split": ("train", "val", "test")[index % 3], "description": text,
                        "description_sha256": hashlib.sha256(text.encode()).hexdigest(), "images": images})
    payload = {"schema_version": 1, "format": "realclip-h200-well-pairs-v1",
               "channels": list(raw.CHANNELS), "preprocessing": {"fixture": True}, "samples": samples}
    manifest = {**payload, "manifest_id": raw.stable_hash(payload)}
    documents = {"dataset.json.gz": json.dumps(manifest).encode(),
                 "raw_downloads.jsonl.gz": "".join(json.dumps(row)+"\n" for row in entries).encode()}
    for name, content in documents.items():
        (directory/name).write_bytes(gzip.compress(content, mtime=0))
    metadata = {"schema_version": 1, "manifest_id": manifest["manifest_id"],
                "counts": {"wells": wells, "images": len(entries)},
                "split_counts": {key: sum(s["split"] == key for s in samples) for key in ("train", "val", "test")},
                "raw_bytes": sum(row["size_bytes"] for row in entries),
                "artifacts": {name: {"size_bytes": (directory/name).stat().st_size,
                                      "sha256": raw.hash_file(directory/name)} for name in documents}}
    (directory/"metadata.json").write_text(json.dumps(metadata))
    return manifest, entries, bodies


@pytest.fixture
def local_server():
    state = {"bodies": {}, "requests": [], "ignore_range": False, "wrong_range": False,
             "wrong_body": False, "fail": False, "truncate_once": False}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            header = self.headers.get("Range")
            state["requests"].append((self.path, header))
            if state["fail"]:
                self.send_response(503)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            body = state["bodies"][self.path]
            if state["wrong_body"]:
                body = b"X" + body[1:]
            offset = int(header.removeprefix("bytes=").removesuffix("-")) if header else 0
            ranged = bool(header) and not state["ignore_range"]
            self.send_response(206 if ranged else 200)
            if ranged:
                start = offset + int(state["wrong_range"])
                self.send_header("Content-Range", f"bytes {start}-{len(body)-1}/{len(body)}")
                body = body[offset:]
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Content-Type", "image/tiff")
            self.end_headers()
            if state["truncate_once"]:
                state["truncate_once"] = False
                self.wfile.write(body[:1000])
                self.wfile.flush()
                self.connection.shutdown(socket.SHUT_WR)
                return
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def opener(request, timeout):
        translated = Request(f"http://127.0.0.1:{server.server_port}"+urlsplit(request.full_url).path,
                             headers=dict(request.header_items()))
        return urlopen(translated, timeout=timeout)

    yield state, opener
    server.shutdown()
    server.server_close()
    thread.join()


def setup(tmp_path, local_server):
    directory, root = tmp_path/"metadata", tmp_path/"data"
    manifest, entries, bodies = fixture_bundle(directory)
    state, opener = local_server
    state["bodies"] = bodies
    return directory, root, manifest, entries, state, opener


def test_complete_bundle_authentication_and_tamper_detection(tmp_path):
    directory = tmp_path/"metadata"
    manifest, entries, _ = fixture_bundle(directory)
    loaded, rows, meta = raw.load_bundle(directory)
    assert loaded == manifest and rows == entries
    assert meta["counts"] == {"wells": 3, "images": 15}
    path = directory/"dataset.json.gz"
    path.write_bytes(path.read_bytes()+b"corrupt")
    with pytest.raises(ValueError, match="checksum/size"):
        raw.load_bundle(directory)


@pytest.mark.parametrize("url", [
    "http://cellpainting-gallery.s3.amazonaws.com/cpg0016-jump/x.tif",
    "https://user:password@cellpainting-gallery.s3.amazonaws.com/cpg0016-jump/x.tif",
    "https://cellpainting-gallery.s3.amazonaws.com/cpg0016-jump/x.tif?token=secret",
    "https://cellpainting-gallery.s3.amazonaws.com/cpg0016-jump/%2e%2e/x.tif",
    "https://other.example/cpg0016-jump/x.tif",
])
def test_public_url_allowlist(url):
    with pytest.raises(ValueError, match="public Cell Painting"):
        raw.validate_url(url)


def test_resume_range_and_ignore_range_restart(tmp_path, local_server):
    _, root, _, entries, state, opener = setup(tmp_path, local_server)
    row = entries[0]
    target = raw.safe_path(root, row["path"])
    target.parent.mkdir(parents=True)
    body = state["bodies"][urlsplit(row["url"]).path]
    partial = target.with_name(target.name+".partial")
    partial.write_bytes(body[:1234])
    result = raw.download_file(row, root, opener=opener)
    assert target.read_bytes() == body and not partial.exists()
    assert result["network_bytes"] == len(body)-1234
    assert state["requests"][0][1] == "bytes=1234-"
    target.unlink()
    partial.write_bytes(body[:321])
    state["ignore_range"] = True
    result = raw.download_file(row, root, opener=opener)
    assert target.read_bytes() == body
    assert result["network_bytes"] == len(body)


def test_hash_failure_keeps_existing_target_and_discards_bad_partial(tmp_path, local_server):
    _, root, _, entries, state, opener = setup(tmp_path, local_server)
    row = entries[0]
    target = raw.safe_path(root, row["path"])
    target.parent.mkdir(parents=True)
    target.write_bytes(b"old incomplete upload")
    state["wrong_body"] = True
    with pytest.raises(raw.DownloadError, match="SHA256"):
        raw.download_file(row, root, retries=2, opener=opener)
    assert target.read_bytes() == b"old incomplete upload"
    assert not Path(str(target)+".partial").exists()
    assert len(state["requests"]) == 2
    state["wrong_body"] = False
    raw.download_file(row, root, opener=opener)
    assert raw.hash_file(target) == row["sha256"]


def test_invalid_content_range_never_appends(tmp_path, local_server):
    _, root, _, entries, state, opener = setup(tmp_path, local_server)
    row = entries[0]
    partial = raw.safe_path(root, row["path"]+".partial")
    partial.parent.mkdir(parents=True)
    partial.write_bytes(b"prefix")
    state["wrong_range"] = True
    with pytest.raises(raw.DownloadError, match="Content-Range"):
        raw.download_file(row, root, retries=1, opener=opener)
    assert partial.read_bytes() == b"prefix"


def test_incomplete_response_resumes_on_retry(tmp_path, local_server):
    _, root, _, entries, state, opener = setup(tmp_path, local_server)
    state["truncate_once"] = True
    result = raw.download_file(entries[0], root, retries=2, opener=opener)
    assert result["network_bytes"] == entries[0]["size_bytes"]
    assert state["requests"] == [(urlsplit(entries[0]["url"]).path, None),
                                 (urlsplit(entries[0]["url"]).path, "bytes=1000-")]


def test_benchmark_full_reuse_and_verify_only_without_network(tmp_path, local_server):
    directory, root, manifest, entries, state, opener = setup(tmp_path, local_server)
    benchmark = raw.run_download(directory, root, workers=3, benchmark=4, opener=opener)
    assert benchmark["status"] == "benchmark_complete"
    assert benchmark["downloaded_images"] == 4
    assert benchmark["network_bytes"] == sum(row["size_bytes"] for row in raw.benchmark_entries(entries, 4))
    assert len(list((root/"images").glob("*.tif"))) == 4
    report = raw.run_download(directory, root, workers=3, opener=opener)
    assert report["status"] == "complete" and report["successful_images"] == 15
    assert report["cached_images"] == 4 and report["downloaded_images"] == 11
    assert len(state["requests"]) == 15
    assert json.loads((root/"manifest.json").read_text()) == manifest
    verified = raw.run_download(directory, root, workers=2, verify_only=True, opener=opener)
    assert verified["status"] == "verified" and verified["network_bytes"] == 0
    cached = raw.run_download(directory, root, workers=2, benchmark=4, opener=opener)
    assert cached["cache_only"] and cached["network_MB_per_second"] is None
    assert cached["cached_images"] == 4 and len(state["requests"]) == 15


def test_verification_reports_every_missing_image_without_network(tmp_path, local_server):
    directory, root, _, entries, state, opener = setup(tmp_path, local_server)
    report = raw.run_download(directory, root, workers=2, verify_only=True, opener=opener)
    assert report["status"] == "failed" and report["failed_images"] == len(entries)
    assert report["not_attempted_images"] == 0 and not state["requests"]


def test_failure_exit_is_nonzero_and_running_replaces_old_success(tmp_path, local_server, monkeypatch):
    directory, root, _, _, state, opener = setup(tmp_path, local_server)
    state["fail"] = True
    root.mkdir()
    (root/"raw_download_report.json").write_text('{"status":"complete"}')
    original = raw.download_file
    observed = []

    def checked(*args, **kwargs):
        observed.append(json.loads((root/"raw_download_report.json").read_text())["status"])
        return original(*args, **kwargs)

    monkeypatch.setattr(raw, "download_file", checked)
    monkeypatch.setattr(raw, "open_public", opener)
    code = raw.main(["--manifest-dir", str(directory), "--data-root", str(root),
                     "--workers", "1", "--retries", "1", "--max-failures", "1"])
    assert code == 1 and observed == ["running"]
    report = json.loads((root/"raw_download_report.json").read_text())
    assert report["status"] == "failed" and report["failed_images"] == 1
    assert report["not_attempted_images"] == 14


def test_kernel_lock_releases_and_blocks_second_task(tmp_path):
    with raw.dataset_lock(tmp_path):
        with pytest.raises(RuntimeError, match="Another download"):
            with raw.dataset_lock(tmp_path):
                pass
    with raw.dataset_lock(tmp_path):
        pass


def test_paths_and_symlinks_are_rejected_before_network(tmp_path, local_server):
    _, root, _, entries, state, opener = setup(tmp_path, local_server)
    for relative in ("../escape", "/absolute", "images/../escape", "a\\b", "images/a:b"):
        with pytest.raises(ValueError):
            raw.safe_path(root, relative)
    outside = tmp_path/"outside"
    outside.mkdir()
    root.mkdir()
    (root/"images").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="escapes|Symlinks"):
        raw.download_file(entries[0], root, opener=opener)
    assert not state["requests"]


def test_partial_hardlink_cannot_modify_another_file(tmp_path, local_server):
    _, root, _, entries, state, opener = setup(tmp_path, local_server)
    outside = tmp_path/"original"
    outside.write_bytes(b"original bytes")
    partial = raw.safe_path(root, entries[0]["path"]+".partial")
    partial.parent.mkdir(parents=True)
    os.link(outside, partial)
    with pytest.raises(raw.DownloadError, match="hardlink"):
        raw.download_file(entries[0], root, opener=opener)
    assert outside.read_bytes() == b"original bytes" and not state["requests"]


def add_archive(directory, manifest, entries, state, complete=False):
    text = "Archival description." if complete else None
    images, extra = [], []
    for channel in raw.CHANNELS:
        body = ("archival " + channel).encode() * 100
        sha = hashlib.sha256(body).hexdigest()
        url = f"https://{raw.HOST}/cpg0016-jump/archive/{channel}.tif"
        row = {"path": f"images/{sha}.tif", "sha256": sha, "size_bytes": len(body), "url": url}
        extra.append(row)
        images.append({**row, "site": "1", "channel": channel})
        state["bodies"][urlsplit(url).path] = body
    sample = {"sample_id": "archival", "well_id": "archive|plate|A01", "training_eligible": False,
              "description_status": "complete" if complete else "incomplete", "description": text,
              "description_sha256": hashlib.sha256(text.encode()).hexdigest() if text else None,
              "images": images}
    payload = {"schema_version": 1, "format": "realclip-prepared-archive-v1",
               "training_manifest_id": manifest["manifest_id"], "total_prepared_wells": 4,
               "samples": [sample], "images": extra}
    archive = {**payload, "archive_id": raw.stable_hash(payload)}
    path = directory / "prepared_archive.json.gz"
    path.write_bytes(gzip.compress(json.dumps(archive).encode(), mtime=0))
    metadata_path = directory / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["artifacts"][path.name] = {"sha256": raw.hash_file(path), "size_bytes": path.stat().st_size}
    size = sum(e["size_bytes"] for e in extra)
    metadata["archive"] = {"archive_id": archive["archive_id"], "wells": 1, "images": 5,
                           "raw_bytes": size, "complete_descriptions": int(complete),
                           "incomplete_descriptions": int(not complete)}
    metadata["all_prepared"] = {"wells": 4, "images": len(entries) + 5,
                                "raw_bytes": metadata["raw_bytes"] + size}
    metadata_path.write_text(json.dumps(metadata))
    return archive


@pytest.mark.parametrize("complete", [False, True])
def test_all_prepared_downloads_archive_without_adding_training_samples(tmp_path, local_server, complete):
    directory, root, manifest, entries, state, opener = setup(tmp_path, local_server)
    archive = add_archive(directory, manifest, entries, state, complete)
    primary = raw.run_download(directory, root, workers=2, opener=opener)
    assert primary["successful_images"] == 15 and primary["download_scope"] == "training"
    assert not (root / "archive_samples.json").exists()
    all_data = raw.run_download(directory, root, workers=2, opener=opener, all_prepared=True)
    assert all_data["status"] == "complete" and all_data["successful_images"] == 20
    assert all_data["cached_images"] == 15 and all_data["downloaded_images"] == 5
    assert all_data["training_wells"] == 3 and all_data["total_wells"] == 4
    assert json.loads((root / "manifest.json").read_text()) == manifest
    assert json.loads((root / "archive_samples.json").read_text()) == archive
    assert len(state["requests"]) == 20


def test_all_prepared_requires_a_bundled_archive(tmp_path, local_server):
    directory, root, _, _, _, opener = setup(tmp_path, local_server)
    with pytest.raises(ValueError, match="no all-prepared archive"):
        raw.run_download(directory, root, opener=opener, all_prepared=True)
    assert not root.exists()


def test_manifest_expansion_is_explicit_authenticated_and_preserves_previous_data(tmp_path):
    directory = tmp_path / "metadata"
    new, _, _ = fixture_bundle(directory, wells=4)
    old = {**new, "samples": new["samples"][:3]}
    old["manifest_id"] = raw.stable_hash({k: v for k, v in old.items() if k != "manifest_id"})
    root = tmp_path / "data"
    root.mkdir()
    raw.atomic_json(root / "manifest.json", old)
    metadata = {"compatible_previous_manifest_ids": [old["manifest_id"]]}
    with pytest.raises(ValueError, match="upgrade-manifest"):
        raw.ensure_manifest(root, new, metadata)
    with pytest.raises(ValueError, match="authenticated"):
        raw.ensure_manifest(root, new, {}, True)
    assert json.loads((root / "manifest.json").read_text()) == old
    assert raw.ensure_manifest(root, new, metadata, True)
    assert json.loads((root / "manifest.json").read_text()) == new
    assert json.loads((root / "manifest_history" / (old["manifest_id"] + ".json")).read_text()) == old
    assert raw.ensure_manifest(root, new, metadata, True) is False


@pytest.mark.parametrize("change", ["description", "holdout", "remove", "preprocessing"])
def test_manifest_expansion_refuses_changed_samples_or_new_holdouts(tmp_path, change):
    directory = tmp_path / "metadata"
    new, _, _ = fixture_bundle(directory, wells=4)
    old = json.loads(json.dumps({**new, "samples": new["samples"][:3]}))
    old["manifest_id"] = raw.stable_hash({k: v for k, v in old.items() if k != "manifest_id"})
    metadata = {"compatible_previous_manifest_ids": [old["manifest_id"]]}
    if change == "description":
        new["samples"][0]["description"] = "Different original text"
    elif change == "holdout":
        new["samples"][-1]["split"] = "val"
    elif change == "remove":
        new["samples"] = new["samples"][1:]
    else:
        new["preprocessing"] = {"changed": True}
    root = tmp_path / "data"
    root.mkdir()
    raw.atomic_json(root / "manifest.json", old)
    with pytest.raises(ValueError, match="changed|removed|only training"):
        raw.ensure_manifest(root, new, metadata, True)
    assert json.loads((root / "manifest.json").read_text()) == old


def test_archive_cannot_admit_itself_to_training(tmp_path, local_server):
    directory, _, manifest, entries, state, _ = setup(tmp_path, local_server)
    add_archive(directory, manifest, entries, state)
    path = directory / "prepared_archive.json.gz"
    archive = json.loads(gzip.decompress(path.read_bytes()))
    archive["samples"][0]["training_eligible"] = True
    archive["archive_id"] = raw.stable_hash({k: v for k, v in archive.items() if k != "archive_id"})
    path.write_bytes(gzip.compress(json.dumps(archive).encode(), mtime=0))
    metadata = json.loads((directory / "metadata.json").read_text())
    metadata["artifacts"][path.name] = {"sha256": raw.hash_file(path), "size_bytes": path.stat().st_size}
    metadata["archive"]["archive_id"] = archive["archive_id"]
    with pytest.raises(ValueError, match="enters the training"):
        raw.load_prepared_archive(directory, manifest, entries, metadata)
