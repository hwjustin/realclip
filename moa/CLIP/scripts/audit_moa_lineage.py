"""Read-only prior-exposure and split audit for the fixed WellCLIP MoA probe.

Does not infer absence of prior information from clean final caption wording.
No training, network requests, paid calls or writes to upstream artifacts.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "CLIP/artifacts/moa_probe_v1"


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def contains(text, phrase):
    if not phrase or len(phrase.strip()) < 5:
        return False
    return bool(re.search(r"(?<!\w)" + re.escape(phrase.strip()) + r"(?!\w)", text, re.I))


def run():
    manifest_path = ROOT / "CLIP/artifacts/prototype_v1/manifest.json"
    selection_path = ROOT / "D/runs/observation_campaign_500_20261023/recovery/selection.json"
    dataset_path = ROOT / "data/processed/dataset.parquet"
    old_split_path = ROOT / "results/cpg_moa/split_scaffold.parquet"
    manifest, selection = read(manifest_path), read(selection_path)
    selected = {row["compound_id"]: row for row in selection["samples"]}
    data = pd.read_parquet(dataset_path).set_index("sample_id")
    assert data.index.is_unique
    old = pd.read_parquet(old_split_path).set_index("sample_id")["split"]
    assert old.index.is_unique
    rows = []
    mechanisms = re.compile(r"\b(?:inhibitor|agonist|antagonist|kinase|apoptosis|mitosis|receptor|mechanism)\b", re.I)
    for sample in manifest["samples"]:
        record = data.loc[sample["well_id"]]
        compound_id = sample["provenance"]["compound_id"]
        assert record["compound_id"] == compound_id
        assert record["compound_group"] == sample["parent_id"]
        assert str(record["source"]) == sample["source"] and str(record["plate"]) == sample["plate"] and str(record["well"]) == sample["well"]
        source = selected[compound_id]
        research_path = Path(source["research_input"])
        assert sha(research_path) == source["research_input_sha256"]
        research = read(research_path)
        observation_path = Path(sample["provenance"]["observation_result"]).parent / "input.json"
        observation = read(observation_path)
        summarize_path = observation_path.parent / "summarize/input.json"
        summary = read(summarize_path)
        known = research.get("known_moa", [])
        drug_name = research.get("compound", {}).get("name")
        guidance = json.dumps(observation.get("hypotheses", []), ensure_ascii=False)
        caption = sample["description"]
        rows.append({
            "sample_id": sample["sample_id"], "well_id": sample["well_id"], "compound_id": compound_id,
            "new_split": sample["split"], "old_scaffold_benchmark_split": old.get(sample["well_id"], "absent"),
            "source": sample["source"], "plate": sample["plate"], "parent": sample["parent_id"],
            "scaffold": record["scaffold"], "known_moa_supplied_to_research": known,
            "dataset_moa_labels": list(record["moa_labels"]),
            "compound_identity_supplied_to_research": bool(research.get("compound")),
            "observation_direct_compound_field": "compound" in observation,
            "observation_direct_known_moa_field": "known_moa" in observation,
            "observation_hypothesis_count": len(observation.get("hypotheses", [])),
            "summary_direct_hypotheses_field": "hypotheses" in summary,
            "summary_has_reviewed_description_context": "reviewed_description_context" in summary,
            "summary_has_eligible_observations": "eligible_observations" in summary,
            "final_caption_exact_known_moa_hits": [label for label in known if contains(caption, label)],
            "final_caption_exact_drug_name_hit": bool(contains(caption, drug_name)),
            "final_caption_exact_compound_id_hit": compound_id in caption,
            "final_caption_mechanism_keyword_hits": sorted(set(m.group().lower() for m in mechanisms.finditer(caption))),
            "guidance_exact_known_moa_hits": [label for label in known if contains(guidance, label)],
            "lineage_files_sha256": {str(p): sha(p) for p in [research_path, observation_path, summarize_path]},
        })
    pair_rows = []
    for first, second in [("train", "val"), ("train", "test"), ("val", "test")]:
        a = [row for row in rows if row["new_split"] == first]
        b = [row for row in rows if row["new_split"] == second]
        shared = {row["scaffold"] for row in a} & {row["scaffold"] for row in b}
        nonempty = shared - {""}
        pair_rows.append({
            "splits": [first, second],
            "shared_plate_groups": len({(r["source"], r["plate"]) for r in a} & {(r["source"], r["plate"]) for r in b}),
            "shared_parents": len({r["parent"] for r in a} & {r["parent"] for r in b}),
            "shared_scaffolds_including_acyclic_empty": len(shared),
            "shared_nonempty_scaffolds": len(nonempty), "empty_acyclic_group_shared": "" in shared,
            "first_split_wells_with_shared_scaffold": sum(r["scaffold"] in shared for r in a),
            "second_split_wells_with_shared_scaffold": sum(r["scaffold"] in shared for r in b),
            "first_split_wells_with_shared_nonempty_scaffold": sum(r["scaffold"] in nonempty for r in a),
            "second_split_wells_with_shared_nonempty_scaffold": sum(r["scaffold"] in nonempty for r in b),
        })
    prior_count = sum(bool(r["known_moa_supplied_to_research"]) for r in rows)
    counts = Counter(r["old_scaffold_benchmark_split"] for r in rows)
    train_scaffolds = {r["scaffold"] for r in rows if r["new_split"] == "train"}
    label_support = {}
    for label in sorted({label for r in rows for label in r["dataset_moa_labels"]}):
        positives = {split: [r for r in rows if r["new_split"] == split and label in r["dataset_moa_labels"]]
                     for split in ["train", "val", "test"]}
        label_support[label] = {
            "positive_counts": {split: len(group) for split, group in positives.items()},
            "test_positive_ids": [r["sample_id"] for r in positives["test"]],
            "test_positives_with_train_shared_scaffold": sum(r["scaffold"] in train_scaffolds for r in positives["test"]),
        }
    result = {
        "schema_version": 1, "script_sha256": sha(__file__),
        "inputs_sha256": {str(p.relative_to(ROOT)): sha(p) for p in [manifest_path, selection_path, dataset_path, old_split_path]},
        "n_wells": len(rows), "current_split_counts": dict(Counter(r["new_split"] for r in rows)),
        "n_research_with_compound_identity": sum(r["compound_identity_supplied_to_research"] for r in rows),
        "n_research_with_nonempty_known_moa": prior_count,
        "n_observation_with_direct_identity_or_moa_field": sum(r["observation_direct_compound_field"] or r["observation_direct_known_moa_field"] for r in rows),
        "n_observation_with_research_derived_guidance": sum(r["observation_hypothesis_count"] > 0 for r in rows),
        "n_summary_with_reviewed_context_and_observations": sum(r["summary_has_reviewed_description_context"] and r["summary_has_eligible_observations"] for r in rows),
        "n_final_caption_exact_known_moa_hits": sum(bool(r["final_caption_exact_known_moa_hits"]) for r in rows),
        "n_final_caption_exact_drug_name_hits": sum(r["final_caption_exact_drug_name_hit"] for r in rows),
        "n_final_caption_exact_compound_id_hits": sum(r["final_caption_exact_compound_id_hit"] for r in rows),
        "n_final_caption_mechanism_keyword_hits": sum(bool(r["final_caption_mechanism_keyword_hits"]) for r in rows),
        "old_scaffold_benchmark_membership": dict(counts),
        "new_by_old_split": {split: dict(Counter(r["old_scaffold_benchmark_split"] for r in rows if r["new_split"] == split)) for split in ["train", "val", "test"]},
        "split_overlap": pair_rows,
        "label_support_and_scaffold_audit": label_support,
        "interpretation": [
            "Research used compound identity and known MoA, and supplied appearance guidance to Observation. This is prior-informed caption supervision, not a claim of MoA-label-free representation training.",
            "Observation direct inputs omit compound and known_moa fields; Summarize compresses reviewed observations. This narrows direct wording exposure but cannot remove upstream guidance effects.",
            "Exact phrase and limited keyword scans are technical checks, not a semantic proof that captions contain no mechanism cues.",
            "The current split holds out plates and chemical parents, but not chemical scaffolds or sources. Do not claim scaffold or unseen-source generalization.",
            "The new 438-well cohort mixes all old benchmark partitions and excluded compounds. Scores cannot be compared to the old benchmark as if the original holdout protocol were retained.",
            "The intended downstream comparison uses only image features at inference; held-out caption embeddings must not be input features for MoA probes.",
            "All diagnostic testing reuses already inspected internal test data. Model selection must remain validation-only and results are exploratory.",
        ],
        "rows": rows,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "leakage_audit.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    train_test = next(r for r in pair_rows if r["splits"] == ["train", "test"])
    prose = f"""# MoA 对照的先验与划分审计

对原438个 well 的 Research→Observation→Summarize lineage 作本地核验，不修改原文或产物。

- {prior_count}/438 的 Research 输入有已知 MoA；{result['n_research_with_compound_identity']}/438 有 compound identity。
- Observation 直接含 compound／known_moa 字段的数量为 {result['n_observation_with_direct_identity_or_moa_field']}，但 {result['n_observation_with_research_derived_guidance']} 份输入接收 Research 衍生外观假设。因此 caption 是身份／机制先验引导的视觉描述监督，不能称为不使用 MoA 信息的纯无监督训练。
- Summarize 的 {result['n_summary_with_reviewed_context_and_observations']} 份输入使用最终审核文字和合格图像观察；不直接读取完整 Research 或原始 MoA 字段。此隔离不能撤销上游引导。
- 最终 caption 中精确已知MoA、药名和compound ID 命中分别为 {result['n_final_caption_exact_known_moa_hits']}、{result['n_final_caption_exact_drug_name_hits']}、{result['n_final_caption_exact_compound_id_hits']}。有限机制关键词命中为 {result['n_final_caption_mechanism_keyword_hits']}。这些扫描不是语义泄漏或生物准确性的证明。

新split为306/66/66，并且 parent 与(source,plate)两两隔离。train/test 仍共享 {train_test['shared_scaffolds_including_acyclic_empty']} 个 scaffold 组，其中非空 scaffold {train_test['shared_nonempty_scaffolds']} 个；共享组覆盖 train {train_test['first_split_wells_with_shared_scaffold']}、test {train_test['second_split_wells_with_shared_scaffold']} 孔。因此是新plate／parent的内部预测，不能解释为 scaffold holdout 或未见source泛化。

438例在旧scaffold基线中的身份：train {counts.get('train',0)}、val {counts.get('val',0)}、test {counts.get('test',0)}、excluded {counts.get('excluded',0)}。新CLIP train包含旧test {result['new_by_old_split']['train'].get('test',0)} 例、旧val {result['new_by_old_split']['train'].get('val',0)} 例，不能再用旧划分成绩声称外部测试获胜。

公平主对照应在当前固定split上，只从 train 特征/标签拟合变换和 MoA 读出、以 val 选模，test 仅终评。推理只用图像，不向MoA模型提供 held-out caption、Research、identity、CP或MoA字符串。固定相同可评估类别，报告标签支持与稀疏性，并避免将不同well的重复视野当独立样本。CLIP train 的 caption 监督包含额外机制先验和CP/对照辅助，若优于DINO，应表述为本流程额外监督下的图像表示收益。

所有输入路径、逐例字段、来源SHA256及split交叉计数见 leakage_audit.json。复现：从仓库根目录运行 `.venv/bin/python CLIP/scripts/audit_moa_lineage.py`。
"""
    (OUT / "leakage_audit.md").write_text(prose)
    print({k: v for k, v in result.items() if k not in ["rows", "inputs_sha256", "script_sha256", "interpretation", "label_support_and_scaffold_audit"]})


if __name__ == "__main__":
    run()
