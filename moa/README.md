# MoA prediction

现有 RealCLIP 多标签 MoA 预测、对照实验与评估代码。代码与 H200 CLIP 训练放在同一仓库，
MoA 是 CLIP 训练完成后的下游任务：用带 MoA 标签的训练集拟合分类器，在验证集选参数，
最后独立评估测试集。原图下载清单不包含 MoA 标签、已训练 MoA 分类器或历史特征缓存。

## 安装

以下命令从 GitHub 仓库根目录执行，使用 Python 3.12。只拟合线性分类器或预测已有
embedding 时用 CPU 即可，不需要 H200，也不需要 PyTorch：

```bash
python3.12 -m venv moa/.venv
source moa/.venv/bin/activate
python -m pip install -e 'moa/CLIP[dev]'
python moa/predict.py --help
python -m pytest -q moa/tests moa/CLIP/tests --ignore=moa/CLIP/tests/test_moa_capacity_probe.py
```

提取图像特征、使用监督 adapter 对照模型或运行全部测试时需要编码器依赖。
可在已经安装仓库根目录 H200 包的环境中执行：

```bash
python -m pip install -e 'moa/CLIP[encoders,dev]'
python -m pytest -q moa/tests moa/CLIP/tests
```

## 用已有 MoA 分类器预测新样本

`predict.py` 读取现有脚本拟合的 `.joblib` 字典模型，保留训练时的 scaler/PCA、
特征维度和标签顺序。它只做预测，不读取标签或重新拟合。joblib 模型应使用自己训练、
来源可信且依赖版本一致的文件。

特征 `.npz` 必须包含 `sample_ids: [N]` 的唯一字符串，以及模型需要的二维浮点特征。
例如模型以 `matched_seed42` 为特征名，而新文件把对应特征保存在 `image_embeddings`：

```bash
python moa/predict.py   --model /path/to/moa/models/primary__full__matched_seed42.joblib   --features /path/to/new_well_embeddings.npz   --feature-map matched_seed42=image_embeddings   --output /path/to/predictions/new_wells.npz
```

若 NPZ 的特征名与模型一致，省略 `--feature-map`；多路融合可重复该参数。
`source_only` 对照还需要与样本顺序一致的 `sources: [N]`。DINO 对照通常需要
`dino: [N,1920]`，原 128 维 CLIP 需要对应的 `[N,128]` 特征；PCA 使用保存的变换。
新特征必须由同一编码器 checkpoint、图像处理和聚合方式生成，改名不会使不同模型兼容。

输出 `.npz` 包含 `sample_ids`、按列排列的 `labels` 和 `probabilities: [N,K]`；
同名 `.json` 记录模型、输入和输出 SHA256。概率是每类独立的 sigmoid 概率，不要求
加和为 1；脚本不自动选阈值。已有输出会保留，需选择新输出路径。

监督 adapter 的 `.pt` 模型使用原脚本：

```bash
python moa/CLIP/scripts/predict_moa_capacity_supplemental.py   --capacity /path/to/moa_probe_v1/capacity   --features /path/to/supplemental/features/features.npz
```

此命令需要 `sample_ids` 和 `image_features: [N,5,384]` 的原冻结 DINO 特征，
以及原模型的 results、selection seal、complete 收据；它按原设计写入 capacity 目录的
`supplemental_probabilities.npz`，请为新预测使用该目录的独立副本。

## 在自己的 CLIP embedding 上拟合 MoA 分类器

复用原 `run_moa_linear_probe.py` 的 `fit_and_select` 和 `predict_saved`。
训练/验证标签均为 `[N,K]` 的 0/1 多标签矩阵；每列有固定的机制名称。
下面假设特征、标签和 source 已按 sample ID 对齐，且训练/验证/测试的 canonical parent
与 `(source, plate)` 已检查没有交叉。`train_embeddings`、`val_embeddings` 为
`[N,D]`，`train_sources`、`val_sources` 为 `[N]` 字符串；这些变量需要由自己的数据读取。

```python
import sys
import joblib
sys.path.insert(0, "moa/CLIP/scripts")
from run_moa_linear_probe import fit_and_select

model, selection = fit_and_select(
    "clip", {"clip": train_embeddings}, train_sources, y_train,
    {"clip": val_embeddings}, val_sources, y_val,
    c_grid=[0.01, 0.1, 1.0, 10.0],
)
model["labels"] = label_names  # 与 y 的列顺序完全一致
joblib.dump(model, "moa_readout.joblib")
```

scaler/PCA 只拟合训练特征；正则强度 C 按验证集 macro AP 选择，完全相同时保留较小 C。
拟合时每个标签需要训练集正负样本；验证集也应具有可用的正负支持。测试集不传给
`fit_and_select`。独立测试评分可使用 `run_expanded_moa_probe.safe_metrics`：
缺少某类正/负样本时保留固定词表，并将该类和完整 macro 指标标为不可评估。

H200 全解冻 CLIP 训练完成后，需要先从新 checkpoint 导出 embedding、对齐真实 MoA
标签，再拟合一个新 readout。本次上传的是现有 MoA 实验代码和通用特征预测入口；
没有把历史冻结骨干分类器接到新 checkpoint 上，也没有在 H200 上运行新 MoA 实验。

## 原实验脚本

保留 `moa/CLIP/` 布局，`SOURCE_FILES.json` 列出原文件及 SHA256；原脚本和测试没有改写。

| 用途 | `moa/CLIP/scripts/` 下的入口 |
| --- | --- |
| 标签身份关联、canonical parent 支持、划分审计 | `prepare_moa_probe.py`, `audit_moa_lineage.py` |
| 原线性对照、DINO/PCA、matched/shuffled、融合、source/prevalence | `run_moa_linear_probe.py` |
| MoA 监督 image adapter 容量对照 | `run_moa_capacity_probe.py` |
| 补充图像测试队列及预测 | `prepare_supplemental_moa_probe.py`, `extract_moa_supplemental.py`, `predict_moa_capacity_supplemental.py` |
| 首次扩量、固定 306 与扩量 603 标签对照 | `prepare_expanded_moa_probe.py`, `precompute_expanded_moa.py`, `run_expanded_moa_probe.py`, `finalize_expanded_moa_probe.py` |
| 最近 1,206 样本扩量后的 MoA | `run_sample_expansion_moa.py`, `diagnose_expansion_moa_cross.py` |
| 部分解冻、维度、direct256、联合维度、完整 BiomedCLIP 对照 | `run_finetune_moa_probe.py`, `run_dimension_moa_probe.py`, `run_direct256_moa_probe.py`, `run_joint_dimension_moa_probe.py`, `run_full_biomed_moa_probe.py` |
| 分 plate 配对 bootstrap、报告、独立结果核验 | `summarize*_moa_probe.py`, `render*_moa_report.py`, `verify*_moa_probe.py` |

这些是历史实验复现入口，部分脚本固定 438/735/1,206 样本或原 66 个验证/测试样本，
并核对原 manifest、特征、协议与模型收据。不能把当前可迁移 `../data/manifest.json`
直接替换进旧实验。原始两类 primary 和八类 exploratory 词表在
`run_finetune_moa_probe.py` 中，保留原设计。

复现原线性实验需要原 `prototype_v1/` 的 manifest、`features/`、`evaluation/`，
以及 `moa_probe_v1/` 的 `protocol.json`、`cohort_arrays.npz`：

```bash
python moa/CLIP/scripts/run_moa_linear_probe.py   --work /path/to/prototype_v1   --parent /path/to/moa_probe_v1   --output /path/to/new_linear_run
```

从上游重新生成标签关联还需要项目原目录中的
`data/processed/dataset.parquet`、对应 `manifest.json`、`data/raw/moa_labels.csv`，
以及原 caption provenance/selection/pairing 文件；可用 `--root`、`--work`、`--output`
指定原项目位置。历史结果核验/报告也需要其对应完整产物。

`run_sample_expansion_moa.py --stage all` 等原调度器会按旧目录寻找两套 `.venv`；
迁移时建议分别以 `--stage prepare`、`--stage export`、`--stage fit` 运行对应阶段，
按各入口 `--help` 指定完整输入路径。`export` 需要编码器环境，其余线性拟合和统计可用
CPU 环境。本目录只打包源码、依赖和测试，不包含历史数据、特征、权重或本地运行记录。
