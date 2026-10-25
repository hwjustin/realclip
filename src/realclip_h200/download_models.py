"""Download pinned public checkpoints, or copy their already-downloaded local files."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

DINO_REPO = "facebook/dinov2-small"
DINO_REVISION = "ed25f3a31f01632728cabb09d1542f84ab7b0056"
BIOMED_REPO = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
BIOMED_REVISION = "9f341de24bfb00180f1b847274256e9b65a3a32e"
TEXT_REPO = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"
TEXT_REVISION = "d673b8835373c6fa116d6d8006b33d48734e305d"


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _copy(source, destination, names):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    for name in names:
        original, target = source / name, destination / name
        if not original.is_file():
            raise FileNotFoundError(original)
        target.parent.mkdir(parents=True, exist_ok=True)
        if original != target:
            shutil.copy2(original, target)


def prepare_models(model_root, local_dino=None, local_biomedclip=None):
    root = Path(model_root).resolve()
    receipt_path = root / "models.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        for relative, digest in receipt["files"].items():
            if _sha(root / relative) != digest:
                raise ValueError(f"Model file changed: {relative}")
        return receipt
    root.mkdir(parents=True, exist_ok=True)
    # The downloader reads no .env and never sends data or implicit HF tokens.
    os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
    from huggingface_hub import hf_hub_download

    dino_files = ["config.json", "preprocessor_config.json", "model.safetensors"]
    biomed_files = ["open_clip_config.json", "open_clip_pytorch_model.bin", "tokenizer.json",
                    "tokenizer_config.json", "special_tokens_map.json", "vocab.txt", "LICENSE.md",
                    "text_config/config.json"]
    if local_dino:
        _copy(local_dino, root / "dinov2-small", dino_files)
        for name in ("README.md", "LICENSE", "LICENSE.md"):
            if (Path(local_dino) / name).is_file():
                _copy(local_dino, root / "dinov2-small", [name])
    else:
        for name in [*dino_files, "README.md"]:
            hf_hub_download(DINO_REPO, filename=name, revision=DINO_REVISION,
                            local_dir=root / "dinov2-small", token=False)
    if local_biomedclip:
        if not (Path(local_biomedclip) / "open_clip_pytorch_model.bin").is_file():
            biomed_files[1] = "open_clip_pytorch_model.safetensors"
        _copy(local_biomedclip, root / "biomedclip", biomed_files)
    else:
        for name in biomed_files[:-1]:
            hf_hub_download(BIOMED_REPO, filename=name, revision=BIOMED_REVISION,
                            local_dir=root / "biomedclip", token=False)
        hf_hub_download(TEXT_REPO, filename="config.json", revision=TEXT_REVISION,
                        local_dir=root / "biomedclip/text_config", token=False)
    files = {str(path.relative_to(root)): _sha(path) for path in sorted(root.rglob("*"))
             if path.is_file() and not any(part.startswith(".") for part in path.relative_to(root).parts)}
    receipt = {"schema_version": 1,
               "expected_revisions": {DINO_REPO: DINO_REVISION, BIOMED_REPO: BIOMED_REVISION,
                                      TEXT_REPO: TEXT_REVISION},
               "sources": {"dinov2-small": "local_copy" if local_dino else "pinned_huggingface_download",
                           "biomedclip": "local_copy" if local_biomedclip else "pinned_huggingface_download"},
               "local_copy_note": "Hashes record supplied bytes; local copies do not independently prove upstream revision",
               "files": files}
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--local-dino", type=Path)
    parser.add_argument("--local-biomedclip", type=Path)
    args = parser.parse_args(argv)
    receipt = prepare_models(args.model_root, args.local_dino, args.local_biomedclip)
    print(json.dumps({"model_root": str(args.model_root.resolve()), "file_count": len(receipt["files"]),
                      "sources": receipt["sources"]}, indent=2))


if __name__ == "__main__":
    main()
