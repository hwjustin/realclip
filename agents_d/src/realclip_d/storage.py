"""Local artifacts, atomic writes, and authenticated stage completion."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False,
                      separators=(",", ":"), default=str)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".write-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False, default=str) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def freeze_json(path: Path, value: Any) -> None:
    if path.exists():
        if read_json(path) != json.loads(canonical(value)):
            raise ValueError(f"Existing artifact differs: {path}; use a new output directory")
    else:
        write_json(path, value)


@contextmanager
def run_lock(directory: Path):
    """OS locks release after interruption; no stale lock recovery guesswork."""
    import fcntl

    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".run.lock").open("a+") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("Another D process is using this output directory") from error
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def seal_stage(directory: Path) -> None:
    hashes = {str(path.relative_to(directory)): file_hash(path)
              for path in sorted(directory.rglob("*"))
              if path.is_file() and path.name != "complete.json"}
    freeze_json(directory / "complete.json", {"schema_version": "1", "files": hashes})


def verify_stage(directory: Path) -> dict:
    receipt = read_json(directory / "complete.json")
    for name, checksum in receipt["files"].items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory.resolve()) or not path.is_file() or file_hash(path) != checksum:
            raise ValueError(f"Stage artifact missing or changed: {name}")
    for name in ("result.json", "input.json", "settings.json", "sources.json", "binding.json"):
        if name not in receipt["files"]:
            raise ValueError(f"Completion receipt does not bind {name}")
    return receipt
