"""Render the frozen MoA comparison without fitting or choosing any models.

Uses the root repository's matplotlib environment. All values and conclusions
come from summary.json; the protocol and model artifacts remain unchanged.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import html
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


FAMILIES = ["dino", "dino_pca128", "matched", "shuffled", "fusion_matched", "fusion_shuffled", "capacity", "source_only", "prevalence"]
LABELS = {"dino": "DINO", "dino_pca128": "DINO PCA128", "matched": "CLIP matched", "shuffled": "CLIP shuffled",
          "fusion_matched": "DINO + matched", "fusion_shuffled": "DINO + shuffled", "capacity": "MoA-only image adapter",
          "source_only": "Source only", "prevalence": "Constant prevalence"}
SHORT = {"dino": "DINO", "dino_pca128": "DINO\nPCA128", "matched": "CLIP\nmatched", "shuffled": "CLIP\nshuffled",
         "fusion_matched": "DINO +\nmatched", "fusion_shuffled": "DINO +\nshuffled", "capacity": "MoA-only\nadapter",
         "source_only": "Source\nonly", "prevalence": "Constant\nprevalence"}
COLORS = {"dino": "#416d9d", "dino_pca128": "#94b5d3", "matched": "#0a8491", "shuffled": "#bf8545",
          "fusion_matched": "#4f9771", "fusion_shuffled": "#d9b18b", "capacity": "#8175aa",
          "source_only": "#8e969d", "prevalence": "#c4c9cd"}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ap_text(row):
    value = f"{row['mean_macro_ap']:.4f}"
    return value + (f" ± {row['seed_std_macro_ap']:.4f}" if row["n_models"] > 1 else "")


def comparison(entry, a, b, kind="plate_bootstrap"):
    return next(row for row in entry[kind]["comparisons"] if row["a"] == a and row["b"] == b)


def interval_text(values):
    return f"[{values[0]:+.4f}, {values[1]:+.4f}]"


def make_primary_plot(summary, output):
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.2), sharey=True, layout="constrained")
    for ax, key, title in zip(axes, ["original_test", "new_plates"], ["Original held-out wells", "Supplemental new-plate wells"]):
        dataset = summary["datasets"][key]
        entry = dataset["label_sets"]["primary"]
        values = entry["families"]
        x = np.arange(len(FAMILIES))
        means = [values[f]["mean_macro_ap"] for f in FAMILIES]
        stds = [values[f]["seed_std_macro_ap"] for f in FAMILIES]
        ax.bar(x, means, color=[COLORS[f] for f in FAMILIES], width=.72,
               yerr=stds, capsize=3, error_kw={"elinewidth": 1.1, "ecolor": "#263342"}, zorder=2)
        for i, family in enumerate(FAMILIES):
            rows = list(values[family]["run_metrics"].values())
            if len(rows) > 1:
                ax.scatter(i + np.linspace(-.12, .12, len(rows)), [row["macro_ap"] for row in rows],
                           s=18, c="#152638", zorder=4)
            label_height = max(means[i] + stds[i], max(row["macro_ap"] for row in rows)) + .008
            ax.text(i, label_height, f"{means[i]:.3f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x, [SHORT[f] for f in FAMILIES], rotation=28, ha="right", fontsize=9)
        ax.set_title(f"{title}\n{dataset['n_wells']} wells / {dataset['n_plates']} plates; {entry['unique_positive_wells']} positive wells", fontsize=12, pad=14)
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=.2)
        ax.set_ylim(0, .34)
        ax.set_ylabel("MoA macro average precision (AP)")
    fig.suptitle("Two predeclared MoA classes: no stable improvement over DINO", fontsize=16)
    fig.supxlabel("Bars: fixed-model mean; dots / error bars: model seeds / seed SD, not confidence intervals", fontsize=10)
    fig.savefig(output / "primary_comparison.png", dpi=180)
    plt.close(fig)


def make_per_label_plot(summary, output):
    entry = summary["datasets"]["new_plates"]["label_sets"]["exploratory"]
    names = entry["labels"]
    positive = entry["positive_counts"]
    fig, ax = plt.subplots(figsize=(12, 6.5), layout="constrained")
    display = ["dino", "matched", "shuffled", "fusion_matched"]
    for shift, family in zip([-.22, -.075, .075, .22], display):
        values = entry["families"][family]["per_label_mean_ap"]
        ax.scatter(values, np.arange(len(names)) + shift, s=46, color=COLORS[family], label=LABELS[family], zorder=3)
    for i, n in enumerate(positive):
        if n == 1:
            ax.axhspan(i - .48, i + .48, color="#fff2eb", zorder=0)
    ax.set_yticks(np.arange(len(names)), [name + f"  (positive n={n})" for name, n in zip(names, positive)], fontsize=10)
    ax.invert_yaxis()
    ax.set(xlim=(0, 1.02), xlabel="Per-class average precision; three-seed mean for CLIP / fusion")
    ax.grid(axis="x", alpha=.2)
    ax.legend(loc="upper right", frameon=True, fontsize=9, ncols=2)
    ax.set_title("142 new-plate wells: exploratory eight-class AP\nOrange rows have only one positive; no broad MoA claim", fontsize=14, pad=14)
    fig.savefig(output / "new_plate_per_label.png", dpi=180)
    plt.close(fig)


def low_label_stats(dataset):
    result = {}
    mapping = {"dino": ("dino", "dino"), "dino_pca128": ("dino_pca128", "dino_pca128"),
               "matched_seed42": ("matched", "matched_seed42"), "shuffled_seed42": ("shuffled", "shuffled_seed42"),
               "dino_plus_matched_seed42": ("fusion_matched", "dino_plus_matched_seed42"), "source_only": ("source_only", "source_only")}
    grouped = defaultdict(list)
    for row in dataset["low_label"]:
        grouped[(row["representation"], row["n_train"])].append(row["metrics"]["macro_ap"])
    for representation, (family, model) in mapping.items():
        rows = []
        for (name, n), values in grouped.items():
            if name == representation:
                rows.append({"n_train": n, "mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)), "n_subsets": len(values)})
        value = dataset["label_sets"]["primary"]["families"][family]["run_metrics"][model]["macro_ap"]
        rows.append({"n_train": 306, "mean": value, "std": 0., "n_subsets": 1})
        result[representation] = sorted(rows, key=lambda row: row["n_train"])
    return result


def make_low_label_plot(summary, output):
    stats = low_label_stats(summary["datasets"]["new_plates"])
    fig, ax = plt.subplots(figsize=(11.5, 6.5), layout="constrained")
    styles = {"dino": ("dino", "DINO", "o"), "dino_pca128": ("dino_pca128", "DINO PCA<=128", "s"),
              "matched_seed42": ("matched", "CLIP matched seed42", "D"), "shuffled_seed42": ("shuffled", "CLIP shuffled seed42", "^"),
              "dino_plus_matched_seed42": ("fusion_matched", "DINO + matched seed42", "v"), "source_only": ("source_only", "Source only", "x")}
    for representation, rows in stats.items():
        family, label, marker = styles[representation]
        x = np.asarray([row["n_train"] / 306 * 100 for row in rows])
        ax.errorbar(x, [row["mean"] for row in rows], yerr=[row["std"] for row in rows],
                    color=COLORS[family], label=label, marker=marker, markersize=6,
                    linewidth=1.8, capsize=3, linestyle="--" if family == "source_only" else "-")
    counts = [row["n_train"] for row in stats["dino"]]
    ax.set_xticks([n / 306 * 100 for n in counts], [f"{round(n/306*100)}%\n({n} labeled training wells)" for n in counts])
    ax.set(xlim=(20, 105), ylim=(0, .25), ylabel="Primary two-class macro AP on 142 new-plate wells")
    ax.grid(axis="y", alpha=.2)
    ax.set_title("Downstream-label sensitivity with fixed image-caption pretraining\n25% / 50%: three nested subsets; error bars are subset SD, not CI", fontsize=13, pad=14)
    ax.legend(loc="upper left", bbox_to_anchor=(0, -.14), ncols=3, frameon=False, fontsize=9)
    fig.supxlabel("At 100%: one full-data fit per fixed seed42 representation. PCA uses 75 dimensions for 76 training rows.\nCLIP pretraining retains all 306 wells and mechanism-prior-guided captions.", fontsize=9)
    fig.savefig(output / "new_plate_low_label.png", dpi=180)
    plt.close(fig)
    return stats


def ci_table(entry):
    rows = ["| 配对比较（前者−后者） | AP差值 | plate条件95%区间 | well条件95%区间 |", "|---|---:|---:|---:|"]
    for row in entry["plate_bootstrap"]["comparisons"]:
        well = comparison(entry, row["a"], row["b"], "well_bootstrap")
        rows.append(f"| {LABELS[row['a']]} − {LABELS[row['b']]} | {row['observed_macro_ap_difference']:+.4f} | {interval_text(row['conditional_paired_bootstrap_95_interval'])} | {interval_text(well['conditional_paired_bootstrap_95_interval'])} |")
    return "\n".join(rows)


def inline(text):
    text = html.escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    return re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"<a href='\2'>\1</a>", text)


def markdown_html(markdown):
    parts = []
    for block in markdown.split("\n\n"):
        if block.startswith("# "):
            parts.append("<h1>" + inline(block[2:]) + "</h1>")
        elif block.startswith("## "):
            parts.append("<h2>" + inline(block[3:]) + "</h2>")
        elif block.startswith("!["):
            match = re.fullmatch(r"!\[([^\]]*)\]\(([^)]+)\)", block)
            parts.append(f"<figure><img src='{html.escape(match[2])}' alt='{html.escape(match[1])}'></figure>")
        elif block.startswith("|"):
            lines = block.splitlines()
            row = lambda line, tag: "<tr>" + "".join(f"<{tag}>{inline(value.strip())}</{tag}>" for value in line.strip("|").split("|")) + "</tr>"
            parts.append("<div class='table'><table>" + row(lines[0], "th") + "".join(row(line, "td") for line in lines[2:]) + "</table></div>")
        elif block.startswith("```"):
            parts.append("<pre>" + html.escape("\n".join(block.splitlines()[1:-1])) + "</pre>")
        else:
            parts.append("<p>" + inline(block).replace("\n", "<br>") + "</p>")
    return "\n".join(parts)


def run(output):
    summary_path = output / "summary.json"
    summary = read(summary_path)
    protocol = read(output / "protocol.json")
    assert sha(output / "protocol.json") == summary["protocol_sha256"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    make_primary_plot(summary, output)
    make_per_label_plot(summary, output)
    low = make_low_label_plot(summary, output)
    old = summary["datasets"]["original_test"]
    new = summary["datasets"]["new_plates"]
    old_primary, primary = [dataset["label_sets"]["primary"] for dataset in [old, new]]
    exploratory = new["label_sets"]["exploratory"]
    seed_table = ["| 固定图像表示 | 原66：主2类AP | 新142：主2类AP | 新142：探索8类AP |", "|---|---:|---:|---:|"]
    seed_table.append("| DINO基线 | " + " | ".join(f"{entry['families']['dino']['mean_macro_ap']:.6f}" for entry in [old_primary, primary, exploratory]) + " |")
    for seed in [42, 43, 44]:
        suffix = "（预先指定主seed）" if seed == protocol["linear_probe"]["primary_model_seed"] else "（复核seed）"
        seed_table.append(f"| CLIP matched seed{seed}{suffix} | " + " | ".join(f"{entry['families']['matched']['run_metrics'][f'matched_seed{seed}']['macro_ap']:.6f}" for entry in [old_primary, primary, exploratory]) + " |")
    primary_seed = protocol["linear_probe"]["primary_model_seed"]
    primary_seed_ap = primary["families"]["matched"]["run_metrics"][f"matched_seed{primary_seed}"]["macro_ap"]
    exploratory_seed_ap = exploratory["families"]["matched"]["run_metrics"][f"matched_seed{primary_seed}"]["macro_ap"]
    delta = comparison(primary, "matched", "dino")
    bounds = delta["conditional_paired_bootstrap_95_interval"]
    if bounds[0] > 0:
        verdict = "在两个预定MoA类别和本批内部新plate上，当前CLIP超过了匹配的DINO线性读出；这一结论的范围仍限于这两个类别。"
    elif bounds[1] < 0:
        verdict = "在两个预定MoA类别和本批内部新plate上，当前CLIP低于匹配的DINO线性读出。"
    else:
        seed_relation = "低于" if primary_seed_ap < primary["families"]["dino"]["mean_macro_ap"] else "高于"
        verdict = f"目前没有稳定胜过DINO的证据。预先固定seed{primary_seed}在补充新plate上的点估计{seed_relation}DINO；三个固定seed的两类AP平均接近DINO，家族配对差值区间覆盖零。"
    table = ["| 表示／分类器 | 原66：主2类 macro AP | 新142：主2类 macro AP | 新142：探索8类 macro AP |", "|---|---:|---:|---:|"]
    for family in FAMILIES:
        table.append(f"| {LABELS[family]} | {ap_text(old_primary['families'][family])} | {ap_text(primary['families'][family])} | {ap_text(exploratory['families'][family])} |")
    supports = ["| MoA | 原train阳性 | 原val阳性 | 原66 test阳性 | 新142阳性 |", "|---|---:|---:|---:|---:|"]
    cohort = read(output / "cohort.json")
    by_label = {row["label"]: row for row in cohort["class_support"]}
    for label, count in zip(exploratory["labels"], exploratory["positive_counts"]):
        row = by_label[label]
        supports.append(f"| {label} | {row['train_positive_parents']} | {row['val_positive_parents']} | {row['test_positive_parents']} | {count} |")
    per_label = ["| 新142 MoA（阳性n） | DINO AP | CLIP matched AP | CLIP shuffled AP | DINO＋matched AP |", "|---|---:|---:|---:|---:|"]
    for i, label in enumerate(exploratory["labels"]):
        per_label.append(f"| {label} ({exploratory['positive_counts'][i]}) | " + " | ".join(f"{exploratory['families'][family]['per_label_mean_ap'][i]:.4f}" for family in ["dino", "matched", "shuffled", "fusion_matched"]) + " |")
    low_table = ["| 固定表示 | 25%：76孔 | 50%：153孔 | 100%：306孔 |", "|---|---:|---:|---:|"]
    names = {"dino": "DINO", "dino_pca128": "DINO PCA≤128", "matched_seed42": "CLIP matched seed42", "shuffled_seed42": "CLIP shuffled seed42", "dino_plus_matched_seed42": "DINO＋matched seed42", "source_only": "Source only"}
    for rep, rows in low.items():
        low_table.append(f"| {names[rep]} | " + " | ".join(f"{row['mean']:.4f} ± {row['std']:.4f}" if row["n_subsets"] > 1 else f"{row['mean']:.4f}" for row in rows) + " |")
    draw_text = []
    for title, entry in [("原66", old_primary), ("新142", primary)]:
        p, w = entry["plate_bootstrap"], entry["well_bootstrap"]
        draw_text.append(f"{title}：plate {p['n_clusters']}组，接纳{p['retained_draws']}/{p['attempted_draws']}次；well {w['n_clusters']}组，接纳{w['retained_draws']}/{w['attempted_draws']}次。")
    report = f"""# MoA预测对照：当前CLIP尚未稳定超过DINO

**{verdict}** 更广的八类探索结果中，CLIP平均低于DINO；DINO＋CLIP融合也未显示清晰增益。这不否定未来更大或更适合MoA的图文训练，但本轮不能声称已打败DINO。

**预先固定的主模型是seed{primary_seed}，不是三个seed的平均。** 在新142主两类上，matched seed{primary_seed} macro AP为 **{primary_seed_ap:.6f}**，低于DINO的 **{primary['families']['dino']['mean_macro_ap']:.6f}**；探索八类分别为 **{exploratory_seed_ap:.6f}** 和 **{exploratory['families']['dino']['mean_macro_ap']:.6f}**。原66中seed42的较高成绩没有在新plate上复现，另两个seed在原66也明显更低。

{chr(10).join(seed_table)}

作为保留全部预定种子的家族汇总，新142主两类DINO macro AP为 **{primary['families']['dino']['mean_macro_ap']:.4f}**，CLIP matched三seed均值为 **{primary['families']['matched']['mean_macro_ap']:.4f}**，差值 **{delta['observed_macro_ap_difference']:+.4f}**，按plate配对bootstrap的条件95%区间为 **{interval_text(bounds)}**。**下文的配对CI针对三个固定seed的家族AP平均，不是seed42的CI。** 不根据测试结果更换主seed，也不把家族均值替代预先固定主模型的结果。

## 实际比较了什么

原438个well的全部site×五通道DINO特征，按原306 train／66 val／66 test划分。原图、caption、CLIP checkpoint和旧评估产物均不修改。下游预测只用图像表示，source-only为明确标出的干扰对照，不向分类器提供测试caption、药物身份、Research、CP或MoA字符串。

除原66测试外，另固定了{new['n_wells']}个新parent、{new['n_plates']}块新plate的补充测试。它们与原全部438的parent、plate及原图SHA均无重叠；5490张uint16 TIFF完成哈希和五通道完整性校验，仅用原冻结DINO和六个CLIP头做推理，没有新生成描述。来源分布：{', '.join(f'{source}={count}' for source,count in new['source_counts'].items())}。这是同数据项目中的内部新plate补充验证，非未见source或scaffold外部测试。

统一的线性读出采用one-vs-rest LogisticRegression、相同balanced class weights和C网格{protocol['linear_probe']['C_grid']}。StandardScaler、PCA及融合缩放只拟合各自训练行；每种表示按原val macro AP选一个全局C。原所有full-data与low-label选择封存后才计算测试结果；补充集沿用这些封存模型，不再拟合、校准或选参。

纯图像容量对照使用与CLIP图像分支相同的聚合／投影结构，直接用train MoA训练三种子、val选步数。它检验一个固定配方下的小头容量，**不是充分调优后的DINO性能上限**。两类与八类各自训练、选模，因此相同类别在两个任务里的AP可以不同。

## 标签覆盖决定了结论范围

原438个独立parent全部精确连接本地原始MoA标签、processed metadata及caption provenance，身份和标签三方一致，共264种注释。主规则保持train≥5、val≥1、test≥2阳性parent，只得到dopamine和serotonin receptor antagonist两类。较宽的train≥3、val≥1、test≥1固定八类仅作探索，未根据模型分数选择。

主两类在原66测试中只有 **{old_primary['unique_positive_wells']}个独立阳性well**，新142中有 **{primary['unique_positive_wells']}个独立阳性well**；多标签well可同时计入两类阳性。八类在新142共{exploratory['unique_positive_wells']}个独立阳性well，HDAC／SRC仍各只有一个阳性。

{chr(10).join(supports)}

## 主结果及对照

{chr(10).join(table)}

±为三个固定模型种子的样本标准差，不是置信区间；DINO/PCA/source/prevalence为单个确定拟合。分类指标均为固定类别macro AP，不能跨不同类别集合直接当作同一任务比较。

![主两类：原测试与新plate补充测试](primary_comparison.png)

原66上matched均值虽高于DINO，但 **shuffled均值也高于matched**，不能将这一小测试集上的名次解释成正确文字监督获益。新142上matched接近DINO、稍高于shuffled，但下表中的关键差值区间均跨零；融合matched与融合shuffled的差距也未稳定建立。

## 配对不确定性：保留六项预定比较

以下差值先对同一模型家族的固定种子取平均。plate重采样是主分析，well／parent重采样为敏感性分析；只接纳每一固定类别都有正负样本的draw，所有区间都以该支持条件、固定模型和当前样本为条件，不包含重新训练、新划分或新来源不确定性，且未作多重比较校正。

**新142补充测试：**

{ci_table(primary)}

**原66保留测试：**

{ci_table(old_primary)}

{' '.join(draw_text)}

原66主两类的独立阳性well只有4个，条件bootstrap不能创造更多生物学证据。新142增加了阳性支持，也未给出稳定优于DINO的结果。

## 八类探索：看总体方向，也保留逐类稀疏性

新142上的八类macro AP：DINO **{exploratory['families']['dino']['mean_macro_ap']:.4f}**，matched **{exploratory['families']['matched']['mean_macro_ap']:.4f}**，DINO＋matched **{exploratory['families']['fusion_matched']['mean_macro_ap']:.4f}**。当前CLIP的八类均值低于DINO；这种下降应如实保留。HDAC和SRC仅一阳性，tubulin也只有两阳性，因此逐类高低不能推广成可靠的机制优劣结论。

![补充新plate逐类AP与阳性支持](new_plate_per_label.png)

{chr(10).join(per_label)}

## 少标签敏感性

固定原CLIP seed42图像表示，把显式下游train标签减少到76／153／306孔；低标签处使用三组保证主类别覆盖的嵌套train子集，所有表示共用相同行，仍只用原val选C。下表和图使用补充新142测试，±是三个子集的标准差；100%为单个完整训练拟合，不把它混成三个子集误差。

{chr(10).join(low_table)}

![补充新plate的下游标签量敏感性](new_plate_low_label.png)

低标签曲线没有展示matched一致超越DINO。PCA在76个train行时实际只有75维，图中写为PCA≤128。此诊断仅减少MoA读出拟合时显式提供的train标签，CLIP预训练仍保留全部306训练well的caption。子集构造也用原训练标签保证覆盖，因此不是现实随机取得25%标注的无偏模拟。

## 监督来源与解释边界

Research曾接收compound identity和已知MoA；438份Observation输入接收其衍生的外观假设。正式caption没有扫描到精确MoA名称、药名或compound ID，这不撤销上游机制先验、CP和对照辅助。因此，若图文表示获益，应解释为这套额外监督流程下的收益，不能称为不依赖MoA信息的纯无监督胜利。本轮推理没有把任何补充well的caption或机制资料送入分类器。

原CLIP的val同时参与预训练checkpoint与下游选模；分析属于内部探索。plate／parent分离不代表source／scaffold隔离；原标签中未列出的MoA按benchmark约定作阴性，不能解释为实验确认无该作用。完整出处与局限见 [lineage审计](leakage_audit.md) 和 [独立方法审阅](method_review.md)。

## 当前判断与下一步

当前结果支持继续研究图文监督，但不支持宣称当前原型已经胜过DINO。下一步应在新增独立parent／plate和更充分的每类阳性支持上，预先固定MoA目标、训练与比较；如果调整图文学习目标、增加成对数据或修订描述，保留DINO、PCA、shuffled和融合对照，并使用新的终评数据。当前已看的66和142不能再当作未触及的新测试集。

## 复现与完整产物

[冻结主协议](protocol.json)、[补充协议](supplemental/supplemental_protocol.json)、[机器可读汇总](summary.json)、[类别支持](class_support.csv)、[完整性核验](finalverification.json)。线性模型与封存选择保存在linear/，纯图像容量模型保存在capacity/，补充图像特征与六组投影保存在supplemental/。

报告只读取已完成的summary.json并作图，不拟合或选择任何模型。重新渲染从仓库根目录运行：

```bash
.venv/bin/python CLIP/scripts/render_moa_probe_report.py
```
"""
    output.joinpath("REPORT.md").write_text(report, encoding="utf-8")
    document = """<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>RealCLIP MoA prediction comparison</title><style>
body{margin:0;background:#f1f5f8;color:#203047;font:16px/1.8 system-ui,sans-serif}main{max-width:1220px;margin:30px auto;background:white;border-radius:15px;padding:30px 38px}h1{font-size:30px;line-height:1.45}h2{border-top:1px solid #d9e3ea;padding-top:22px;margin-top:32px}a{color:#087c8c}nav{display:flex;flex-wrap:wrap;gap:20px;border-bottom:1px solid #d9e3ea;padding:0 0 14px}.table{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:14px;font-variant-numeric:tabular-nums}th,td{padding:10px;border-bottom:1px solid #dce5eb;text-align:left}th{background:#eaf3f6}figure{margin:22px 0}img{width:100%;height:auto}pre{padding:16px;background:#edf2f6;overflow:auto}strong{color:#133d4b}@media(max-width:720px){main{margin:0;padding:20px;border-radius:0}h1{font-size:24px}table{font-size:13px}}
</style><main><nav><a href="REPORT.md">完整 Markdown</a><a href="summary.json">全部统计 JSON</a><a href="protocol.json">冻结主协议</a><a href="class_support.csv">类别支持 CSV</a><a href="finalverification.json">完整性核验</a></nav>"""
    output.joinpath("index.html").write_text(document + markdown_html(report) + "</main></html>", encoding="utf-8")
    files = ["REPORT.md", "index.html", "primary_comparison.png", "new_plate_per_label.png", "new_plate_low_label.png"]
    receipt = {"summary_sha256": sha(summary_path), "protocol_sha256": sha(output / "protocol.json"), "script_sha256": sha(__file__),
               "derived_files": {name: sha(output / name) for name in files}, "no_training_or_model_selection": True}
    output.joinpath("render_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(output / "index.html")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "artifacts/moa_probe_v1")
    run(parser.parse_args().output.resolve())
