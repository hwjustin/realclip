from __future__ import annotations

import importlib.util
import hashlib
import io
import json
from pathlib import Path
import tarfile
from urllib.error import HTTPError

import pytest


def load_script(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / (name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


verify = load_script("verify_bundle")
publish = load_script("publish_github")


def make_tar(path, name, content=b"image", link=False):
    with tarfile.open(path, "w:gz") as archive:
        info = tarfile.TarInfo(name)
        if link:
            info.type = tarfile.SYMTYPE
            info.linkname = "../escape"
            archive.addfile(info)
        else:
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))


@pytest.mark.parametrize("name", ["../escape", "/absolute", "images/../../escape", "images\\escape", "C:/escape"])
def test_reject_unsafe_paths(tmp_path, name):
    with pytest.raises(ValueError):
        verify.safe_target(tmp_path, name)


def test_tar_cannot_escape_or_create_links(tmp_path):
    archive = tmp_path / "bad.tar.gz"
    make_tar(archive, "../escape")
    with pytest.raises(ValueError):
        verify.extract_shard(archive, tmp_path / "data")
    assert not (tmp_path / "escape").exists()
    make_tar(archive, "images/link", link=True)
    with pytest.raises(ValueError):
        verify.extract_shard(archive, tmp_path / "data")


def test_tar_refuses_existing_symlink(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    output = tmp_path / "data"
    output.mkdir()
    (output / "images").symlink_to(outside, target_is_directory=True)
    archive = tmp_path / "bad.tar.gz"
    make_tar(archive, "images/pixel.tif")
    with pytest.raises(ValueError):
        verify.extract_shard(archive, output)
    assert not (outside / "pixel.tif").exists()


def test_extraction_resumes_only_identical_files(tmp_path):
    archive = tmp_path / "image.tar.gz"
    output = tmp_path / "data"
    make_tar(archive, "images/pixel.tif")
    assert verify.extract_shard(archive, output) == 1
    assert verify.extract_shard(archive, output) == 1
    (output / "images/pixel.tif").write_bytes(b"changed")
    with pytest.raises(ValueError, match="refusing overwrite"):
        verify.extract_shard(archive, output)
    assert (output / "images/pixel.tif").read_bytes() == b"changed"


def test_preexisting_temporary_is_preserved(tmp_path):
    archive = tmp_path / "image.tar.gz"
    output = tmp_path / "data"
    output.mkdir()
    (output / "pixel.tif.extracting").write_bytes(b"existing work")
    make_tar(archive, "pixel.tif")
    with pytest.raises(FileExistsError):
        verify.extract_shard(archive, output)
    assert (output / "pixel.tif.extracting").read_bytes() == b"existing work"


def test_shard_index_and_corruption(tmp_path):
    archive = tmp_path / "metadata.tar.gz"
    make_tar(archive, "manifest.json", b"{}")
    index = {"complete": True, "files": [{"file_name": archive.name,
             "size": archive.stat().st_size, "sha256": verify.hash_file(archive)}]}
    (tmp_path / "upload_index.json").write_text(json.dumps(index))
    assert verify.verify_shards(tmp_path) == [archive]
    archive.write_bytes(b"damaged")
    with pytest.raises(ValueError, match="checksum"):
        verify.verify_shards(tmp_path)


def test_manifest_verifies_descriptions_images_and_split_integrity(tmp_path):
    (tmp_path / "images").mkdir()
    images = []
    for index, channel in enumerate(["DNA", "RNA", "ER", "Mito", "AGP"]):
        path = tmp_path / "images" / f"{index}.tif"
        path.write_bytes(f"pixels {index}".encode())
        images.append({"path": str(path.relative_to(tmp_path)), "sha256": verify.hash_file(path),
                       "site": 0, "channel": channel})
    text = "Oval nuclei."
    sample = {"sample_id": "one", "source": "s", "plate": "p", "parent_id": "parent",
              "split": "train", "description": text, "description_sha256": hashlib.sha256(text.encode()).hexdigest(),
              "images": images}
    manifest = {"channels": ["DNA", "RNA", "ER", "Mito", "AGP"], "samples": [sample]}
    manifest["manifest_id"] = verify.stable_hash(manifest)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    assert verify.verify_manifest(tmp_path)["images"] == 5
    (tmp_path / images[0]["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        verify.verify_manifest(tmp_path)


def test_publish_allowlist_excludes_data_weights_and_secrets(tmp_path):
    for path in ["README.md", "pyproject.toml", ".gitignore", "src/realclip_h200/model.py",
                 "scripts/train.py", "tests/test_model.py", "configs/h200.json",
                 ".env", "data/secret.py", "models/weights.bin", "credentials.json",
                 "src/realclip_h200/weights.pt"]:
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("source")
    files = publish.collect_files(tmp_path)
    assert set(files) == {"README.md", "pyproject.toml", ".gitignore", "src/realclip_h200/model.py",
                          "scripts/train.py", "tests/test_model.py", "configs/h200.json"}


def test_publish_rejects_symlinked_source(tmp_path):
    (tmp_path / "README.md").write_text("readme")
    (tmp_path / "pyproject.toml").write_text("project")
    directory = tmp_path / "scripts"
    directory.mkdir()
    (directory / "secret.py").symlink_to(tmp_path / "README.md")
    with pytest.raises(ValueError, match="nonregular"):
        publish.collect_files(tmp_path)


def test_publish_rejects_symlinked_directory(tmp_path):
    (tmp_path / "README.md").write_text("readme")
    (tmp_path / "pyproject.toml").write_text("project")
    source = tmp_path / "data"
    source.mkdir()
    (source / "secret.py").write_text("secret")
    (tmp_path / "scripts").symlink_to(source, target_is_directory=True)
    with pytest.raises(ValueError, match="nonregular"):
        publish.collect_files(tmp_path)


class FakeGitHub:
    def __init__(self, exists=False, private=True):
        self.exists, self.private, self.calls = exists, private, []

    def request(self, method, path, data=None):
        self.calls.append((method, path, data))
        if path == "/user":
            return {"login": "hwjustin"}
        if path == "/repos/hwjustin/realclip-h200" or path == "/user/repos":
            if method == "GET" and not self.exists:
                raise HTTPError(path, 404, "missing", {}, None)
            return {"private": self.private, "default_branch": "main", "html_url": "https://github.com/hwjustin/realclip-h200"}
        if "/git/ref/" in path:
            return {"object": {"sha": "parent"}}
        if path.endswith("/git/commits/parent"):
            return {"tree": {"sha": "oldtree"}}
        if "?recursive=1" in path:
            return {"tree": []}
        if path.endswith("/git/trees"):
            return {"sha": "newtree"}
        if path.endswith("/git/commits"):
            return {"sha": "newcommit"}
        if method == "PATCH":
            return {}
        raise AssertionError((method, path))


def test_publish_creates_private_repo_and_nonforce_commit():
    client = FakeGitHub()
    result = publish.publish(client, "hwjustin", "realclip-h200", {"README.md": "readme"})
    assert result["created"] is True
    creation = next(data for method, path, data in client.calls if path == "/user/repos")
    assert creation["private"] is True
    update = next(data for method, path, data in client.calls if method == "PATCH")
    assert update == {"sha": "newcommit", "force": False}


def test_publish_refuses_existing_and_public_repositories():
    for client in (FakeGitHub(exists=True), FakeGitHub(exists=True, private=False)):
        with pytest.raises(ValueError):
            publish.publish(client, "hwjustin", "realclip-h200", {"README.md": "readme"})
        assert all(method == "GET" for method, _, _ in client.calls)
