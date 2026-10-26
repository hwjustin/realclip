"""Render the larger CLIP review; no fitting or model selection occurs here."""
from pathlib import Path
import hashlib
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_moa_probe_report import markdown_html

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/expanded_moa_probe_v1"
NAMES = {"dino": "DINO", "pca": "DINO PCA128", "old_matched": "Old CLIP matched",
         "new_matched": "Expanded CLIP matched", "old_shuffled": "Old CLIP shuffled",
         "new_shuffled": "Expanded CLIP shuffled", "fusion_old_matched": "DINO + old CLIP",
         "fusion_new_matched": "DINO + expanded CLIP", "source": "Source only", "prevalence": "Prevalence"}


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def entry(s, stage, dataset, label_set="primary"):
    return s["stages"][stage]["datasets"][dataset]["label_sets"][label_set]


def comparison(p, a, b):
    return next(r for r in p["plate_bootstrap"]["comparisons"] if r["a"] == a and r["b"] == b)


def bounds(x):
    return f"[{x[0]:+.4f}, {x[1]:+.4f}]"


def plot(s):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout="constrained", sharey=True)
    keys = ["dino", "old_matched", "new_matched", "new_shuffled", "fusion_new_matched"]
    colors = ["#4677a6", "#b18a5b", "#058a93", "#aab1b7", "#49996a"]
    upper = max(max(v["mean_macro_ap"] + v["seed_std_macro_ap"],
                    max(r["macro_ap"] for r in v["run_metrics"].values()))
                for stage in ["fixed306", "expanded603"] for dataset in ["original66", "retained67"]
                for key, v in entry(s, stage, dataset)["families"].items() if key in keys) * 1.15
    for row, stage in enumerate(["fixed306", "expanded603"]):
        for col, dataset in enumerate(["original66", "retained67"]):
            ax = axes[row, col]
            p = entry(s, stage, dataset)
            for i, (key, color) in enumerate(zip(keys, colors)):
                v = p["families"][key]
                ax.bar(i, v["mean_macro_ap"], yerr=v["seed_std_macro_ap"], color=color, capsize=3)
                runs = [r["macro_ap"] for r in v["run_metrics"].values()]
                ax.scatter(np.full(len(runs), i), runs, s=15, color="#172638", zorder=4)
            ax.set(xticks=np.arange(len(keys)), xticklabels=[NAMES[k].replace(" ", "\n", 1) for k in keys],
                   ylabel="MoA macro average precision", ylim=(0, upper),
                   title=f"MoA training: {s['stages'][stage]['n_moa_train']} wells | {dataset}")
            ax.grid(axis="y", alpha=.15)
    fig.suptitle("Pretraining expansion (306 to 603 pairs) and downstream MoA labels are separate factors")
    fig.supxlabel("Two fixed MoA classes. Bars: model means; dots: individual runs; error bars: seed SD, not confidence intervals.", fontsize=9)
    fig.savefig(OUT / "moa_comparison.png", dpi=170)
    plt.close(fig)


def run():
    s = read(OUT / "summary.json")
    retrieval = read(OUT / "retrieval_gain.json")
    cohort = read(OUT / "cohort.json")
    with np.load(OUT / "cohort_arrays.npz", allow_pickle=False) as a:
        vocabulary = a["label_names"].tolist()
        original_ids = set(cohort["original306_train_sample_ids"])
        original_rows = np.asarray([sid in original_ids for sid in a["sample_ids"]])
        support_lines = []
        for label in cohort["label_sets"]["exploratory"]["labels"]:
            j = vocabulary.index(label)
            counts = [int(a["y_all"][original_rows, j].sum())] + [
                int(a["y_all"][a["splits"] == split, j].sum()) for split in ["train", "val", "test"]]
            support_lines.append("| " + label + " | " + " | ".join(map(str, counts)) + " |")
    plot(s)
    columns = [(stage, dataset) for stage in ["fixed306", "expanded603"] for dataset in ["original66", "retained67"]]
    headline = []
    for stage in ["fixed306", "expanded603"]:
        p = entry(s, stage, "retained67")
        r = comparison(p, "new_matched", "old_matched")
        q = comparison(p, "new_matched", "dino")
        headline.append(f"{s['stages'][stage]['n_moa_train']}个MoA训练标签：67孔上新CLIP−旧CLIP的三种子平均AP差 {r['observed_macro_ap_difference']:+.4f}，plate条件95%区间 {bounds(r['conditional_paired_bootstrap_95_interval'])}；新CLIP−DINO为 {q['observed_macro_ap_difference']:+.4f}，区间 {bounds(q['conditional_paired_bootstrap_95_interval'])}。")
    principal = entry(s, "expanded603", "retained67")["families"]
    verdict = ("**结论：有小幅图文检索改善的点估计，但尚未稳定转化为MoA收益。** "
        f"所有方法都使用603个MoA训练标签、评估保留67孔时，新CLIP三种子平均AP为{principal['new_matched']['mean_macro_ap']:.4f}，"
        f"旧CLIP为{principal['old_matched']['mean_macro_ap']:.4f}，DINO为{principal['dino']['mean_macro_ap']:.4f}。"
        "配对区间仍宽且覆盖零，不能宣称普遍退化，当前也不支持在MoA上获胜。")
    lines = ["# 扩量CLIP收益复核：图文检索与MoA分开看", "", verdict, "",
             "图文训练由306对增至603对；原66验证和66测试孔、候选顺序、图像、描述、骨干、训练配方与预定初始化保持一致。新增297对只进入训练。", "",
             "## 图文检索：正向点估计，来源表现不一致", "",
             f"相同66候选上，matched双向MRR从 {retrieval['arms']['matched']['baseline_mean_mrr']:.4f} 提高到 {retrieval['arms']['matched']['expanded_mean_mrr']:.4f}，约提高9.4%，三个种子均上升。图→文R@1为6.06%→7.07%，文→图为3.54%→5.05%。但图→文R@5为18.69%→18.18%，验证MRR平均也略降。", "",
             "同来源检索中，source_2均值0.1705→0.1822；source_7为0.1703→0.1524，三个种子均下降，后者接近该来源随机期望0.1526。它不是跨来源都稳定改善。固定gallery的配对plate bootstrap对整体MRR增益给出约[-0.0181,+0.0368]区间，覆盖零。三初始化同方向不能替代新的生物样本。", "",
             "## MoA：两个训练标签规模分别比较", "",
             "A固定原306个显式下游MoA训练标签，只比较图文预训练扩量后的表示。B给DINO、旧CLIP和新CLIP相同603个MoA训练标签，检验扩大下游训练覆盖后的实际结果。新增caption仍属机制先验引导的描述监督，A不能称完全无新增MoA信息的无标签扩量。", "",
             *headline, "",
             "原142补充测试中75个孔与新训练集的parent或plate关联，已剔除；46个parent本身进入新训练。保留67孔、50块板，与新train/val parent及plate无重叠。两类主评估有10和6个阳性，共14个独立阳性well。两个测试集都已在历史分析中看过，属于内部探索性复核。", "",
             "主类别继续固定为dopamine和serotonin receptor antagonist。仍用原66验证集在相同六个C候选中选全局正则化；所有标准化／PCA仅拟合对应训练行。88个MoA读出全部选定封存后才计算测试预测。分类时只使用图像特征；source-only为明确标出的干扰对照。", "",
             "## 预先指定seed42的结果", "",
             "| 表示 | 306标签／原66 | 306标签／保留67 | 603标签／原66 | 603标签／保留67 |", "|---|---:|---:|---:|---:|"]
    for key in ["dino", "old_matched", "new_matched", "new_shuffled", "fusion_new_matched"]:
        vals = [entry(s, stage, dataset)["seed42"][key]["macro_ap"] for stage, dataset in columns]
        lines.append(f"| {NAMES[key]} | " + " | ".join(f"{v:.4f}" for v in vals) + " |")
    lines += ["", "## 全部三个预定种子的汇总", "",
              "| 表示 | 306标签／原66 | 306标签／保留67 | 603标签／原66 | 603标签／保留67 |", "|---|---:|---:|---:|---:|"]
    for key in NAMES:
        values = []
        for stage, dataset in columns:
            v = entry(s, stage, dataset)["families"][key]
            values.append(f"{v['mean_macro_ap']:.4f}" + (f" ± {v['seed_std_macro_ap']:.4f}" if v["n_models"] > 1 else ""))
        lines.append(f"| {NAMES[key]} | " + " | ".join(values) + " |")
    lines += ["", "±为同一数据上的模型种子标准差，非置信区间；以下配对CI针对三种子AP的平均，seed42仍单独保留，不能根据测试表现改选seed。不同测试集合阳性率不同，不能将跨集合AP差直接当泛化下降。", "",
              "![两个MoA训练规模和两个保留评估集合](moa_comparison.png)"]
    lines += ["", "## 固定类别的标签支持", "",
              "| MoA | 原306训练阳性 | 新603训练阳性 | 原66验证阳性 | 原66测试阳性 |", "|---|---:|---:|---:|---:|",
              *support_lines, "",
              "总孔数扩量并不等于所有MoA同步扩量；多数验证类别依旧只有一个阳性，因此MoA读出选参仍有较大波动。"]
    for stage, dataset in columns:
        p = entry(s, stage, dataset)
        lines += ["", f"## 配对区间：{stage}／{dataset}", "",
                  "| 比较（前者−后者） | 平均AP差 | plate条件95%区间 | seed42差 | seed42区间 |", "|---|---:|---:|---:|---:|"]
        for r in p["plate_bootstrap"]["comparisons"]:
            lines.append(f"| {NAMES[r['a']]} − {NAMES[r['b']]} | {r['observed_macro_ap_difference']:+.4f} | {bounds(r['conditional_paired_bootstrap_95_interval'])} | {r.get('seed42_observed_difference', float('nan')):+.4f} | {bounds(r.get('seed42_conditional_paired_bootstrap_95_interval', [float('nan')]*2))} |")
        b = p["plate_bootstrap"]
        lines.append(f"\n整群抽样单位为source+plate，有效draw {b['retained_draws']}/{b['attempted_draws']}；仅接纳固定两类均有正负支持的draw。区间以当前模型、数据和支持条件为前提，不包含重新训练／新来源不确定性，未作多重比较校正。well／parent重采样敏感性结果保存在summary.json。")
    lines += ["", "## 八类探索结果", "",
              "原66测试集八类均有阳性，但其中六类各仅一个；两类集合和八类集合各自按验证macro AP选C，因此相同MoA在两套任务中的AP可以不同。保留67孔缺少HDAC和tubulin阳性，不能报告完整八类macro AP，也不把剩余六类平均伪装成八类。", "",
              "| 表示 | 306标签／原66八类 | 603标签／原66八类 |", "|---|---:|---:|"]
    for key in ["dino", "pca", "old_matched", "new_matched", "new_shuffled", "fusion_new_matched"]:
        vals = [entry(s, stage, "original66", "exploratory")["families"][key]["mean_macro_ap"] for stage in ["fixed306", "expanded603"]]
        lines.append(f"| {NAMES[key]} | {vals[0]:.4f} | {vals[1]:.4f} |")
    lines += ["", "## 解释与完整性", "",
              "检索改善与MoA收益是不同结论。应分别检查固定MoA标签下新旧CLIP差异，以及相同603标签下新CLIP对DINO和打乱文字的差异；仅新增标签让所有分类器变好，不能归因于图文预训练。", "",
              "图文骨干仍冻结，所有matched图文checkpoint依旧由验证检索选模，三个新模型都选择第40步；MoA验证仍稀疏。全batch扩量同时增加负例数量和样本曝光，本次估计的是整个扩量配方的实际效果，不是分离其他因素后的纯数量效应。caption生成曾使用机制／身份先验，正式预测仍是图像only；当前也不是未见来源或scaffold隔离外部测试。", "",
              "来源基线另有数值修正留痕：重复设计行曾因BLAS批形状出现1 ulp伪差。四个来源控制统一用唯一设计行预测后广播、按同一验证集重验C；84个DINO／CLIP图像模型保持原封存权重。原v1选择及seal保留，未根据测试结果调整模型。", "",
              "[机器可读汇总](summary.json) · [检索配对不确定性](retrieval_gain.json) · [补充集重新隔离审计](supplemental_eligible.json) · [原扩量检索报告](../expanded_samples_v1/comparison/REPORT.md)", "",
              "[独立数值核验](verification.json) · [本轮冻结协议](protocol.json)", "",
              "复现报告（仓库根目录）：", "", "```bash\n.venv/bin/python CLIP/scripts/render_expanded_moa_report.py\n```"]
    report = "\n".join(lines) + "\n"
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")
    style = "body{font:16px/1.75 system-ui;color:#203046;background:#f2f5f8}main{max-width:1200px;margin:30px auto;padding:30px;background:white}img{max-width:100%}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}th,td{border-bottom:1px solid #ddd;padding:8px;text-align:left}.table{overflow:auto}h2{margin-top:32px;border-top:1px solid #ddd;padding-top:20px}a{color:#087c8d}pre{background:#eef3f5;padding:15px;overflow:auto}"
    (OUT / "index.html").write_text("<!doctype html><html lang='zh'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>Expanded CLIP review</title><style>" + style + "</style><main>" + markdown_html(report) + "</main></html>", encoding="utf-8")
    (OUT / "render_receipt.json").write_text(json.dumps({"renderer_sha256": sha(Path(__file__)), "summary_sha256": sha(OUT / "summary.json"),
        "outputs": {name: sha(OUT / name) for name in ["REPORT.md", "index.html", "moa_comparison.png"]}}, indent=2) + "\n")
    print(OUT / "index.html")


if __name__ == "__main__":
    run()
