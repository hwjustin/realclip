# RealCLIP: full DINOv2 + BiomedCLIP text training

Train the complete DINOv2-small image encoder and the complete BiomedCLIP text
encoder together on existing Cell Painting images and unchanged English
descriptions. Both pretrained encoders, both projection heads, and temperature
are trainable. No caption generation, paid API, CP features, or MOA labels are
needed for training.

This is a new end-to-end experiment. Previous frozen-feature checkpoints are
not compatible with it. The original experiments and their data remain intact.

## Data and model

The initial portable snapshot contains **735 wells: 603 train, 66 validation,
66 test**, and 25,900 original uint16 treatment TIFFs, totalling
60,014,495,288 bytes before archive compression. It preserves the existing
plate and compound-parent separation. New descriptions still being generated
are not included in this fixed snapshot.

Each well uses all available sites and five channels in the fixed order
DNA, RNA, ER, Mito, AGP. The original per-well, per-channel intensity windows
are preserved: pooled 0.5–99.8 percentiles across all sites, with the original
fallback for degenerate channels. Full fields are letterboxed to 518×518,
replicated to RGB, and normalized with ImageNet statistics. There are no random
crops or color augmentations.

The image path is complete DINOv2 → per-channel mean over sites → L2 → shared
384→32 + GELU → concatenate five channels → dropout → 160→128 → L2.
The text path is the complete pretrained BiomedCLIP text tower, including its
official CLS pooling and 512-dimensional pretrained projection, followed by
dropout → 512→128 → L2. Descriptions over the 256-token context are rejected,
not silently truncated. All encoder outputs are recomputed during training;
only fixed pixels and intensity windows can be reused.

## Install on the training machine

Use Linux, Python 3.12, and a CUDA-enabled PyTorch build compatible with the
machine's NVIDIA driver. Dependencies reproduce the versions installed on the
source machine. Check the driver and CUDA wheel compatibility on the H200
host before the full run.

```bash
git clone https://github.com/hwjustin/realclip.git
cd realclip
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
python -c 'import torch; print(torch.__version__, torch.cuda.get_device_name(0))'
```

The dataset uses relative paths and can be stored on any local disk. For a
direct SSH transfer, stage the frozen dataset on the source machine with
`scripts/stage_ssh_data.py`. Its TIFFs are hard links to the existing source
files, so staging does not duplicate the 60 GB of images. See `--help` for the
source manifest and intensity-window arguments. Treat the staged dataset as
read-only: modifying a hard-linked image would also modify the source image.

Run this on the source machine after substituting the actual SSH target,
port, and destination. Both machines need `rsync`:

```bash
rsync -a --partial --info=progress2 -e 'ssh -p PORT' \
  /absolute/path/to/staged-data/ USER@HOST:/absolute/destination/data/
```

Re-run the same command after an interruption. Completed files are reused.
After transfer, verify every image checksum on the destination before training:

```bash
python scripts/verify_bundle.py --data-root /absolute/destination/data
```

Compressed archive transfer is also supported: download the entire completed
bundle, including `upload_index.json`, and run
`python scripts/verify_bundle.py --bundle-dir downloads --data-root data --extract`.
Archive extraction temporarily requires space for both compressed and raw data.

Public pretrained model downloads are pinned to immutable revisions:

```bash
python -m realclip_h200.download_models --help
python -m realclip_h200.download_models --model-root models
```

The downloaded directory contains `dinov2-small/` and `biomedclip/`. No
generative-model API keys or source `.env` files are needed. The two local
pretrained model directories total approximately 873 MB; they can also be
transferred separately rather than downloaded again.

## H200 training

The initial single-GPU configuration uses BF16 autocast, FP32 contrastive
logits, full-encoder gradient checkpointing, AdamW, separate backbone and head
learning rates, and gradient clipping. A well batch and an image microbatch
are different: one well contains many site/channel images. Image microbatches
limit encoder work per call while the well batch determines the actual
InfoNCE negatives. Ordinary gradient accumulation does not add cross-batch
contrastive negatives.

The default loader uses `num_workers=0`, which also works in containers with
small shared memory. Increase it only after allowing enough host/shared memory
for batches of complete 518-pixel well images.

```bash
python -m realclip_h200.train --help
python -m realclip_h200.train \
  --config configs/h200.json --data-root data --model-root models \
  --output runs/h200-smoke --smoke-steps 2

python -m realclip_h200.train \
  --config configs/h200.json --data-root data --model-root models \
  --output runs/h200-full
```

The default is a starting configuration, not an H200 throughput measurement.
Verify peak memory on the target machine before increasing the well batch or
image microbatch. Full unfreezing with only 603 training wells can overfit;
the best checkpoint is selected by validation retrieval MRR. Training never
evaluates the test split. Keep the test evaluation separate from model and
hyperparameter selection.

Exactly repeated full descriptions are treated as multiple valid positives in
the contrastive loss and retrieval metrics; other wells are batch negatives.
This rule is recorded explicitly and differs from exact-well-only metrics in
some earlier frozen experiments.

`last.pt` stores both encoders, heads, optimizer, scheduler, progress, random
states, and data/config identity for resuming. `best.pt` is selected using
validation only. Use a fresh output directory for a new configuration or data
snapshot.

When moving a partially trained run to another machine, copy the complete run
directory, including both `last.pt` and any existing `best.pt`, and resume with
that same directory as `--output`.

```bash
python -m realclip_h200.train \
  --config configs/h200.json --data-root data --model-root models \
  --output runs/h200-full --resume runs/h200-full/last.pt

python -m pytest -q
```

To evaluate the held-out test set once the model configuration is fixed:

```bash
python -m realclip_h200.train \
  --config configs/h200.json --data-root data --model-root models \
  --output runs/h200-full --evaluate test --checkpoint runs/h200-full/best.pt
```

## Packaging and provenance

`scripts/export_data.py` builds the portable manifest and compressed TIFF
shards from the original frozen snapshot. It omits Observation PNGs, controls,
Brightfield images, CP tables, caption-generation requests, credentials,
frozen embeddings, and historical experiments. Source manifest and window-file
hashes record lineage without embedding source-machine absolute paths.

Data and pretrained models belong on Drive or local storage; GitHub contains
only code, configuration, tests, and documentation. The upload tools use an
explicit file allowlist. `scripts/publish_github.py --help` provides a private
repository publishing fallback for a machine with GitHub authentication.

Source models: [DINOv2](https://github.com/facebookresearch/dinov2) and
[BiomedCLIP](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224).
Data originates from the existing JUMP Cell Painting snapshot. Upstream data
and model terms continue to apply; this package does not grant a new license
to those assets.
