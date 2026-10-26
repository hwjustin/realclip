# RealCLIP: full DINOv2 + BiomedCLIP text training

Train the complete DINOv2-small image encoder and the complete BiomedCLIP text
encoder together on existing Cell Painting images and unchanged English
descriptions. Both pretrained encoders, both projection heads, and temperature
are trainable. No caption generation, paid API, CP features, or MOA labels are
needed for training.

This is a new end-to-end experiment. Previous frozen-feature checkpoints are
not compatible with it. The original experiments and their data remain intact.

## H200：下载全部 1,806 个已准备样本

最新数据分为两个范围：

| 范围 | well 数 | TIFF 数 | 原图大小 |
| --- | ---: | ---: | ---: |
| 当前训练/验证/测试队列 | 1,206（1,074/66/66） | 42,545 | 98.70 GB |
| 额外归档：316 个完整描述、284 个描述未完成 | 600 | 21,065 | 48.85 GB |
| `--all-prepared` 全量下载 | 1,806 | 63,610 | 147.55 GB |

沿用全部 treatment sites 的 DNA/RNA/ER/Mito/AGP 五通道；control、Brightfield
不属于本训练输入。归档样本保存在 `archive_samples.json`，不会进入训练 manifest。
原验证和测试各 66 个 well 不变，旧 735 个样本的描述、窗宽、图像及划分也不变。
284 个尚未完成的描述不补写、不伪造。

服务器建议至少预留 200 GB。下载脚本仅用 Python 标准库，不需要先安装 PyTorch
或 AWS 凭据。它复用 SHA256 正确的旧图片，只下载缺失/损坏的原图，支持 Range 续传。
切换前停止旧下载/rsync 任务；同一数据目录只运行一个下载任务。

已有本仓库的服务器直接更新：

```bash
cd "$HOME/realclip-transfer/realclip-s3"
git pull --ff-only
mkdir -p logs
```

首次使用则执行：

```bash
mkdir -p "$HOME/realclip-transfer"
cd "$HOME/realclip-transfer"
git clone https://github.com/hwjustin/realclip.git realclip-s3
cd realclip-s3
mkdir -p logs
```

以下命令均从 `realclip-s3/` 执行。`--upgrade-manifest` 只允许清单声明的兼容扩量：
它验证所有旧样本完全不变、仅增加训练样本，并将旧清单备份到
`../data/manifest_history/`。旧图像不会被删除，也无需重传。

先下载 100 张样本测速，这些图片会保留给后续全量下载：

```bash
python3 scripts/download_raw.py --data-root ../data \
  --all-prepared --upgrade-manifest --workers 16 --benchmark 100
cat ../data/raw_benchmark.json
```

`network_MB_per_second` 是实际收到的 HTTP 数据量除以耗时，包含校验/写盘开销。
`cache_only: true` 表示全命中缓存，本次不能作为网络测速。H200 的速度需在该节点实测。

后台补齐全部 63,610 张原图：

```bash
nohup python3 -u scripts/download_raw.py --data-root ../data \
  --all-prepared --upgrade-manifest --workers 16 \
  > logs/raw-download-all.log 2>&1 < /dev/null &

tail -f logs/raw-download-all.log
```

按 `Ctrl+C` 只退出日志查看；下载继续。中断后重新运行相同命令。
完整任务结束后检查报告（成功下载即已逐图核对大小和 SHA256）：

```bash
python3 - <<'PYCODE'
import json
from pathlib import Path
r = json.loads(Path("../data/raw_download_report.json").read_text())
m = json.loads(Path("manifests/metadata.json").read_text())
assert r["status"] == "complete", r
assert r["download_scope"] == "all_prepared", r
assert r["failed_images"] == 0, r
assert r["successful_images"] == m["all_prepared"]["images"] == 63610, r
assert r["manifest_id"] == m["manifest_id"]
print("1,806 个样本的原图已全部下载/复用并通过校验；训练队列仍为 1,206 个")
PYCODE
```

单独重新校验全部图片（不联网）可执行：

```bash
python3 scripts/download_raw.py --data-root ../data --all-prepared --verify-only
```

不加 `--all-prepared` 时，仅下载训练/验证/测试队列的 42,545 张图。

## 在扩量数据上训练

训练仍是完整 DINOv2 + BiomedCLIP 文本塔双侧全解冻。需要 Python 3.12、兼容驱动的
CUDA PyTorch 和已分配的 H200；Slurm 集群在 GPU 作业内启动训练。
安装需要可访问的软件源或已备好的离线依赖。

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
nvidia-smi
python - <<'PYCODE'
import torch
assert torch.cuda.is_available(), "CUDA unavailable"
print("GPU:", torch.cuda.get_device_name(0))
assert torch.cuda.is_bf16_supported(), "BF16 unavailable"
PYCODE
```

模型尚未上传时，从固定的公开 Hugging Face 版本下载：

```bash
python -m realclip_h200.download_models --model-root ../models
```

已有完整且校验过的两个模型目录时，可使用下文的本地登记命令。
扩量后的数据身份已改变，使用新的运行目录，不直接续训旧 735 样本的断点：

```bash
python -u -m realclip_h200.train \
  --config configs/h200.json --data-root ../data --model-root ../models \
  --output runs/h200-smoke-v2 --smoke-steps 2

nohup .venv/bin/python -u -m realclip_h200.train \
  --config configs/h200.json --data-root ../data --model-root ../models \
  --output runs/h200-expanded-v2 \
  > logs/h200-expanded-v2.log 2>&1 < /dev/null &

tail -f logs/h200-expanded-v2.log
```

归档的 600 个样本不会被训练读取。默认 BF16、20 epochs；best.pt 仍由原 66 验证 well 选择。

## Data and model

The current portable training snapshot contains **1,206 wells: 1,074 train,
66 validation, 66 test**, and 42,545 original uint16 treatment TIFFs, totalling
98,697,986,392 bytes. The optional prepared archive adds 600 wells and 21,065
images, so all 1,806 prepared wells total 147,548,160,295 bytes. Archival wells
never become contrastive training examples. The previous 735-well snapshot is
an authenticated expansion ancestor; its samples and holdouts are unchanged.

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

## MoA prediction

[MoA 代码与使用说明](moa/README.md)包含既有多标签线性探针、监督 adapter 对照、
扩量/维度评估、统计与报告代码，以及独立的 embedding → MoA 概率预测入口。
MoA 标签只用于下游任务；H200 全解冻 CLIP 训练仍只使用图像和已有描述。
新 checkpoint 的 embedding 需要重新拟合相应 MoA readout。
