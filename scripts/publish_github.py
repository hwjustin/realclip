#!/usr/bin/env python3
"""Publish this code-only package to a private GitHub repository.

Credentials are read exclusively from GH_TOKEN or GITHUB_TOKEN. No data,
weights, caches, environment files, or files outside the allowlist are sent.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
TOP_LEVEL = {"README.md", "VALIDATION.md", "pyproject.toml", ".gitignore", "uv.lock"}
DIRECTORIES = {"src/realclip_h200": ".py", "scripts": ".py", "tests": ".py", "configs": ".json"}


def collect_files(root: Path) -> dict[str, str]:
    root = root.resolve()
    files = {}
    candidates = [root / name for name in sorted(TOP_LEVEL)]
    for directory, extension in DIRECTORIES.items():
        base = root / directory
        if base.exists():
            candidates.extend(sorted(base.rglob("*" + extension)))
    for path in candidates:
        if not path.exists():
            continue
        relative = path.relative_to(root)
        ancestors = [path, *path.parents]
        if (any(parent.is_symlink() for parent in ancestors if parent.is_relative_to(root))
                or not path.is_file() or any(part.startswith(".") for part in relative.parts[:-1])):
            raise ValueError(f"Refusing nonregular or hidden source: {relative}")
        if not path.resolve().is_relative_to(root):
            raise ValueError(f"Source escapes package: {relative}")
        if path.stat().st_size > 2 * 1024 * 1024:
            raise ValueError(f"Unexpectedly large source file: {relative}")
        files[relative.as_posix()] = path.read_text(encoding="utf-8")
    if not {"README.md", "pyproject.toml"}.issubset(files):
        raise ValueError("README.md and pyproject.toml are required before publishing")
    return files


def blob_sha(text: str) -> str:
    content = text.encode("utf-8")
    return hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()


class GitHub:
    def __init__(self, token: str):
        self.token = token

    def request(self, method: str, path: str, data: dict | None = None):
        payload = None if data is None else json.dumps(data).encode("utf-8")
        request = Request("https://api.github.com" + path, data=payload, method=method, headers={
            "Authorization": "Bearer " + self.token,
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "Content-Type": "application/json",
            "User-Agent": "realclip-h200-code-publisher",
        })
        with urlopen(request, timeout=60) as response:
            return json.load(response)


def publish(client: GitHub, owner: str, name: str, files: dict[str, str], existing: bool = False) -> dict:
    login = client.request("GET", "/user")["login"]
    if login.casefold() != owner.casefold():
        raise ValueError(f"Authenticated account is {login}; requested owner is {owner}")
    repository = f"/repos/{quote(owner, safe='')}/{quote(name, safe='')}"
    created = False
    try:
        repo = client.request("GET", repository)
    except HTTPError as error:
        if error.code != 404:
            raise
        repo = client.request("POST", "/user/repos", {
            "name": name, "private": True, "auto_init": True,
            "description": "End-to-end DINOv2 + BiomedCLIP text training on existing Cell Painting well pairs",
        })
        created = True
    if repo.get("private") is not True:
        raise ValueError("Refusing to publish into a public repository")
    branch = repo["default_branch"]
    reference = client.request("GET", repository + "/git/ref/heads/" + quote(branch, safe=""))
    parent_sha = reference["object"]["sha"]
    commit = client.request("GET", repository + "/git/commits/" + parent_sha)
    tree_sha = commit["tree"]["sha"]
    if not created:
        tree = client.request("GET", repository + "/git/trees/" + tree_sha + "?recursive=1")
        if tree.get("truncated"):
            raise ValueError("Cannot verify a truncated remote tree")
        remote = {entry["path"]: entry["sha"] for entry in tree["tree"] if entry["type"] == "blob"}
        expected = {path: blob_sha(content) for path, content in files.items()}
        if remote == expected:
            return {"url": repo["html_url"], "commit": parent_sha, "unchanged": True}
        if not existing:
            raise ValueError("Repository already exists with different contents; use --existing only after review")
    tree = client.request("POST", repository + "/git/trees", {
        "base_tree": tree_sha,
        "tree": [{"path": path, "mode": "100644", "type": "blob", "content": content}
                 for path, content in sorted(files.items())],
    })
    commit = client.request("POST", repository + "/git/commits", {
        "message": "Package end-to-end DINOv2 and BiomedCLIP text training",
        "tree": tree["sha"], "parents": [parent_sha],
    })
    client.request("PATCH", repository + "/git/refs/heads/" + quote(branch, safe=""), {
        "sha": commit["sha"], "force": False,
    })
    return {"url": repo["html_url"], "commit": commit["sha"], "created": created, "files": len(files)}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--owner", default="hwjustin")
    parser.add_argument("--name", default="realclip-h200")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--existing", action="store_true", help="Explicitly permit updating a private existing repo")
    args = parser.parse_args()
    if not args.owner or not args.name or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_." for char in args.owner + args.name):
        parser.error("Use plain GitHub owner and repository names")
    files = collect_files(args.root)
    if args.dry_run:
        print(json.dumps({"owner": args.owner, "name": args.name, "private": True,
                          "files": sorted(files), "bytes": sum(len(v.encode()) for v in files.values())}, indent=2))
        return
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not token:
        parser.error("Set GH_TOKEN or GITHUB_TOKEN in this process; never put credentials in this package or .env")
    try:
        print(json.dumps(publish(GitHub(token), args.owner, args.name, files, args.existing), indent=2))
    except (HTTPError, ValueError) as error:
        # HTTPError's string contains status and URL, never the authorization header.
        print(f"Publication stopped: {error}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
