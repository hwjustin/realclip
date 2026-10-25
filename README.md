# RealCLIP: full DINOv2 + BiomedCLIP text training

Train the complete DINOv2-small image encoder and the complete BiomedCLIP text
encoder together on existing Cell Painting images and unchanged English
descriptions. Both pretrained encoders, both projection heads, and temperature
are trainable. No caption generation, paid API, CP features, or MOA labels are
needed for training.

This is a new end-to-end experiment. Previous frozen-feature checkpoints are
not compatible with it. The original experiments and their data remain intact.

## H200：从公开 S3 测速、下载并训练

下载脚本只需要 Python 3.9 及以上版本的标准库，可以先测速和下载，再安装 PyTorch。它只取
清单指定的 25,900 张原图，保留全部英文描述、603/66/66 划分和原有预处理窗宽。
文件按原始 SHA256 校验，已有且正确的文件直接复用，失败下载可以重新运行续传。
不需要 AWS 账号。原图 URL、校验清单及训练元数据随本仓库提供在 `manifests/`。

以下示例使用服务器上的 `~/realclip-transfer/data` 和 `models`，因此会复用之前
通过 SSH 上传到这些位置的文件。选择有至少约 100 GB 可用空间的目录。
测速结果代表执行命令的服务器节点到 S3 的实际链路；这里还没有实测 H200 的网速。
切换前在中转电脑按 `Ctrl+C` 停止旧的 rsync 上传；完整上传的图像会继续复用，
不要让两种传输同时写入同一个数据目录。

先获取代码。使用新的 `realclip-s3` 目录，可以保留此前从 U 盘解压的 `realclip`：

```bash
mkdir -p "$HOME/realclip-transfer"
cd "$HOME/realclip-transfer"
git clone https://github.com/hwjustin/realclip.git realclip-s3
cd realclip-s3
mkdir -p logs
```

如果 `realclip-s3` 已经是 Git 仓库，在该目录执行 `git pull --ff-only` 更新。
以下命令都从 `realclip-s3/` 执行。

先并发下载 100 张图测速；这些图会保留给完整下载复用：

```bash
python3 scripts/download_raw.py \
  --data-root ../data --workers 16 --benchmark 100
cat ../data/raw_benchmark.json
```

报告区分缓存文件与实际网络流量，速度包含下载、校验和写盘开销。如果实际网络
字节数为零，说明样本已全部缓存，这次运行不能作为网络测速。

测速正常结束后，后台下载剩余原图。SSH 断开后任务继续运行：

```bash
nohup python3 -u scripts/download_raw.py \
  --data-root ../data --workers 16 \
  > logs/raw-download.log 2>&1 < /dev/null &

tail -f logs/raw-download.log
```

`Ctrl+C` 只退出日志查看。下载中断后重新运行同一条下载命令；同一数据目录
只运行一个下载任务。等待完整下载任务正常结束，再检查报告：

```bash
python3 - <<'PY'
import json
from pathlib import Path
report = json.loads(Path("../data/raw_download_report.json").read_text())
manifest = json.loads(Path("../data/manifest.json").read_text())
assert report["status"] == "complete", report
assert report["failed_images"] == 0, report
assert report["successful_images"] == 25900, report
assert report["manifest_id"] == manifest["manifest_id"]
print("25,900 张原图已下载或复用，并通过校验")
PY
```

如需单独重新校验全部原图，可执行：

```bash
python3 scripts/download_raw.py --data-root ../data --verify-only
```

训练需要 Python 3.12、兼容驱动的 CUDA PyTorch，以及已经分配到的 H200。
在 Slurm 等集群中，训练命令应在 GPU 作业内执行。安装依赖需要可访问的软件源
或已经准备好的离线依赖；本仓库和 U 盘包不包含 Python 环境。

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e .

nvidia-smi
python - <<'PY'
import torch
print("PyTorch:", torch.__version__)
assert torch.cuda.is_available(), "CUDA unavailable"
print("GPU:", torch.cuda.get_device_name(0))
assert torch.cuda.is_bf16_supported(), "BF16 unavailable"
PY
```

模型来自固定版本的公开 Hugging Face 仓库。模型尚未上传时执行：

```bash
python -m realclip_h200.download_models --model-root ../models
```

如果两个完整模型目录已从 U 盘上传并通过校验，则用本地文件登记：

```bash
python -m realclip_h200.download_models --model-root ../models \
  --local-dino ../models/dinov2-small \
  --local-biomedclip ../models/biomedclip
```

先运行两步完整双塔训练，使用独立输出目录：

```bash
python -u -m realclip_h200.train \
  --config configs/h200.json --data-root ../data --model-root ../models \
  --output runs/h200-smoke-s3 --smoke-steps 2
```

测试正常结束后正式训练。默认两边编码器全部解冻、BF16、20 epochs：

```bash
nohup .venv/bin/python -u -m realclip_h200.train \
  --config configs/h200.json --data-root ../data --model-root ../models \
  --output runs/h200-full \
  > logs/h200-full.log 2>&1 < /dev/null &

tail -f logs/h200-full.log
```

如果 `runs/h200-full` 已有训练断点，按下面的续训命令恢复，而不是重新初始化。
`best.pt` 按验证集选出；测试集只在训练完成并固定配置后显式评估。

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

## Models and installation / 模型和安装

Use Linux, Python 3.12, and a CUDA-enabled PyTorch build compatible with the
machine's NVIDIA driver. Dependencies reproduce the versions installed on the
source machine. Check the driver and CUDA wheel compatibility on the H200
host before the full run.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
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
python -m realclip_h200.download_models --model-root ../models
```

The downloaded directory contains `dinov2-small/` and `biomedclip/`. No
generative-model API keys or source `.env` files are needed. The two local
pretrained model directories total approximately 873 MB; they can also be
transferred separately rather than downloaded again.

When the complete model directories already exist, register their local bytes
without a model download (from the `realclip-s3/` layout above):

```bash
python -m realclip_h200.download_models --model-root ../models \
  --local-dino ../models/dinov2-small \
  --local-biomedclip ../models/biomedclip
```

Use this only after those local files have passed the USB transfer's checksums.

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
  --config configs/h200.json --data-root ../data --model-root ../models \
  --output runs/h200-smoke --smoke-steps 2

python -m realclip_h200.train \
  --config configs/h200.json --data-root ../data --model-root ../models \
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
  --config configs/h200.json --data-root ../data --model-root ../models \
  --output runs/h200-full --resume runs/h200-full/last.pt

python -m pytest -q
```

To evaluate the held-out test set once the model configuration is fixed:

```bash
python -m realclip_h200.train \
  --config configs/h200.json --data-root ../data --model-root ../models \
  --output runs/h200-full --evaluate test --checkpoint runs/h200-full/best.pt
```

## Packaging and provenance

`scripts/export_data.py` builds the portable manifest and compressed TIFF
shards from the original frozen snapshot. It omits Observation PNGs, controls,
Brightfield images, CP tables, caption-generation requests, credentials,
frozen embeddings, and historical experiments. Source manifest and window-file
hashes record lineage without embedding source-machine absolute paths.

Raw images and pretrained weights stay on object storage or local disks.
GitHub contains code, configuration, tests, documentation, and the compressed
frozen dataset metadata and raw-download URL index in `manifests/`. These
metadata preserve captions, splits, image checksums, and intensity windows;
they contain no source-machine absolute paths or API credentials. The upload
tools use an explicit file allowlist. `scripts/publish_github.py --help`
provides a private-repository publishing fallback for a machine with GitHub
authentication.

Source models: [DINOv2](https://github.com/facebookresearch/dinov2) and
[BiomedCLIP](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224).
Data originates from the existing JUMP Cell Painting snapshot. Upstream data
and model terms continue to apply; this package does not grant a new license
to those assets.
