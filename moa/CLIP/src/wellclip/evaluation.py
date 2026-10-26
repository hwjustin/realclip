"""Sealed paired retrieval evaluation, galleries and inspectable well examples."""
from __future__ import annotations

import csv
import html
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from .common import CHANNELS, PREPROCESS, hash_file, model_identity, read_json, utc_now, write_json
from .diagnostics import diagnostic_texts, score_diagnostics
from .training import (load_features, load_run_model, random_expectation, retrieval_metrics,
                       split_indices, validate_training)


def topk_cases(similarities: np.ndarray, samples: list[dict], direction: str, k: int = 5) -> list[dict]:
    if direction not in ("image_to_text", "text_to_image"):
        raise ValueError("Unknown retrieval direction")
    scores = similarities if direction == "image_to_text" else similarities.T
    if scores.shape != (len(samples), len(samples)):
        raise ValueError("Top-k samples do not match paired gallery")
    order = np.argsort(-scores, axis=1, kind="stable")
    cases = []
    for query_index, query in enumerate(samples):
        ranking = order[query_index]
        rank = int(np.flatnonzero(ranking == query_index)[0]) + 1
        cases.append({
            "query_sample_id": query["sample_id"], "query_well_id": query["well_id"],
            "query_description": query["description"], "correct_pair_rank": rank,
            "direction": direction,
            "top_k": [{"sample_id": samples[int(index)]["sample_id"],
                       "well_id": samples[int(index)]["well_id"],
                       "description": samples[int(index)]["description"],
                       "cosine_similarity": float(scores[query_index, index]),
                       "is_exact_pair": int(index) == query_index}
                      for index in ranking[:k]],
        })
    return cases


def source_metrics(similarities: np.ndarray, samples: list[dict]) -> dict:
    result = {}
    for source in sorted({sample["source"] for sample in samples}):
        indices = [i for i, sample in enumerate(samples) if sample["source"] == source]
        if len(indices) < 2:
            result[source] = {"candidate_count": len(indices), "skipped": True,
                              "reason": "fewer_than_two_candidates"}
        else:
            result[source] = {"skipped": False,
                              **retrieval_metrics(similarities[np.ix_(indices, indices)]),
                              "random_expectation": random_expectation(len(indices))}
    return result


def _well_preview(sample: dict, destination: Path) -> None:
    """A full five-channel treatment field after the exact model transform."""
    from .preprocessing import compute_windows, load_raw, preprocess_array

    windows = compute_windows(sample)
    first_site = min({str(image["site"]) for image in sample["images"]}, key=lambda value: int(value))
    selected = {image["channel"]: image for image in sample["images"] if str(image["site"]) == first_site}
    cell = 160
    preview = Image.new("RGB", (cell * 5, cell + 48), "white")
    draw = ImageDraw.Draw(preview)
    draw.text((6, 3), f"{sample['sample_id']} | site {first_site} | first site shown, all sites encoded", fill="black")
    mean = np.asarray(PREPROCESS["mean"], dtype=np.float32)[:, None, None]
    std = np.asarray(PREPROCESS["std"], dtype=np.float32)[:, None, None]
    for column, channel in enumerate(CHANNELS):
        transformed = preprocess_array(load_raw(selected[channel]), windows[channel]).numpy()
        unit = np.clip(transformed * std + mean, 0, 1)
        pixels = np.rint(unit.transpose(1, 2, 0) * 255).astype(np.uint8)
        image = Image.fromarray(pixels).resize((cell, cell), Image.Resampling.LANCZOS)
        draw.text((column * cell + 5, 24), channel, fill="black")
        preview.paste(image, (column * cell, 48))
    destination.parent.mkdir(parents=True, exist_ok=True)
    preview.save(destination)


def _summary(runs: list[dict]) -> dict:
    summary = {}
    for arm in ("matched", "shuffled"):
        arm_runs = [run for run in runs if run["arm"] == arm]
        values = {}
        for direction in ("image_to_text", "text_to_image"):
            values[direction] = {}
            for metric in ("R@1", "R@5", "R@10", "MRR"):
                numbers = [run["test"][direction][metric] for run in arm_runs]
                values[direction][metric] = {"mean": float(np.mean(numbers)),
                                             "std": float(np.std(numbers, ddof=1)) if len(numbers) > 1 else 0.0}
        numbers = [run["test"]["mean_mrr"] for run in arm_runs]
        values["mean_mrr"] = {"mean": float(np.mean(numbers)),
                              "std": float(np.std(numbers, ddof=1)) if len(numbers) > 1 else 0.0}
        summary[arm] = values
    seeds = sorted({run["seed"] for run in runs})
    differences = []
    for seed in seeds:
        matched = next(run for run in runs if run["arm"] == "matched" and run["seed"] == seed)
        shuffled = next(run for run in runs if run["arm"] == "shuffled" and run["seed"] == seed)
        differences.append({"seed": seed,
                            "mean_mrr_difference": matched["test"]["mean_mrr"] - shuffled["test"]["mean_mrr"]})
    summary["paired_seed_differences"] = differences
    summary["matched_exceeds_shuffled_mean_mrr"] = summary["matched"]["mean_mrr"]["mean"] > summary["shuffled"]["mean_mrr"]["mean"]
    return summary


def _write_html(output_dir: Path, evaluation: dict, all_cases: dict) -> None:
    esc = html.escape
    rows = []
    for run in evaluation["runs"]:
        i2t, t2i = run["test"]["image_to_text"], run["test"]["text_to_image"]
        cells = [run["name"], str(run["best_step"]), f"{run['validation_mrr']:.4f}"]
        cells += [f"{result[metric]:.4f}" for result in (i2t, t2i) for metric in ("R@1", "R@5", "R@10", "MRR")]
        rows.append("<tr>" + "".join(f"<td>{esc(value)}</td>" for value in cells) + "</tr>")
    header = ["Run", "Selected step", "Val MRR", "I→T R@1", "R@5", "R@10", "MRR", "T→I R@1", "R@5", "R@10", "MRR"]
    parts = ["<!doctype html><html lang='en'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>",
             "<title>Well-level Cell Painting CLIP</title><style>body{font:16px system-ui;max-width:1280px;margin:32px auto;padding:0 18px;color:#17212b;background:#f7f8fb}table{border-collapse:collapse;width:100%;background:white}th,td{padding:8px;border:1px solid #d5dce5;text-align:right}td:first-child,th:first-child{text-align:left}details{background:white;border:1px solid #d5dce5;margin:12px 0;padding:12px}summary{cursor:pointer;font-weight:600}img{max-width:100%;height:auto;border:1px solid #d5dce5}.case{border-top:1px solid #ddd;padding:14px 0}.candidate{margin:8px 0;padding:8px;background:#f0f3f8}.exact{border-left:4px solid #26834a}.score{font-family:monospace}.muted{color:#536272}a{color:#175ca4}</style>",
             "<h1>Well-level Cell Painting image–text contrastive learning</h1>",
             "<p>One treatment well, all sites × five fluorescence channels, paired with one unchanged English description. Frozen DINOv2 and BiomedCLIP; learned 128-dimensional projections.</p>",
             f"<p>{evaluation['split_counts']['train']} training / {evaluation['split_counts']['val']} validation / {evaluation['split_counts']['test']} held-out test wells. Plates and chemical parents do not overlap. Test retrieval uses only the {evaluation['split_counts']['test']} test candidates.</p>",
             "<p class='muted'>New-plate internal evaluation, not an unseen-source benchmark. Similar valid morphology descriptions can refer to different wells. Seed standard deviations are not confidence intervals. First-site previews show all five channels; every site contributed to the model.</p>",
             "<p><a href='evaluation.json'>Full evaluation JSON</a> · <a href='metrics.csv'>Metrics CSV</a> · <a href='top5.json'>All top-5 cases</a></p>",
             "<table><thead><tr>" + "".join(f"<th>{esc(cell)}</th>" for cell in header) + "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"]
    chance = evaluation["random_expectation"]
    parts.append(f"<p>Random expectation: R@1={chance['R@1']:.4f}, R@5={chance['R@5']:.4f}, R@10={chance['R@10']:.4f}, MRR={chance['MRR']:.4f}.</p>")
    for arm in ("matched", "shuffled"):
        summary = evaluation["summary"][arm]["mean_mrr"]
        parts.append(f"<p>{esc(arm)} bidirectional mean MRR: {summary['mean']:.4f} ± {summary['std']:.4f} (seed standard deviation).</p>")
    conclusion = ("Correct pair training exceeds shuffled-pair training in mean test MRR across the predetermined seeds."
                  if evaluation["summary"]["matched_exceeds_shuffled_mean_mrr"] else
                  "Correct pair training does not exceed shuffled-pair training in mean test MRR; this prototype has not established useful held-out alignment.")
    parts.append(f"<p><strong>{esc(conclusion)}</strong></p>")
    if evaluation.get("diagnostics"):
        diagnostics = evaluation["diagnostics"]
        parts.append(f"<h2>Predetermined treatment-only morphology diagnostics</h2><p>Frozen text representation prefers the paraphrase in {diagnostics['frozen']['n_paraphrase_preferred']}/8 controls. These controls do not select checkpoints.</p>")
        for name, result in diagnostics["projected"].items():
            parts.append(f"<p>{esc(name)}: {result['n_paraphrase_preferred']}/8 paraphrases preferred; mean margin {result['mean_margin']:.4f}.</p>")
    parts.append("<h2>Held-out top-5 retrieval examples</h2>")
    for name, directions in all_cases.items():
        parts.append(f"<details><summary>{esc(name)}</summary>")
        for direction, cases in directions.items():
            parts.append(f"<details><summary>{esc(direction)} — all {len(cases)} test queries</summary>")
            for case in cases:
                identifier = esc(case["query_sample_id"])
                parts.append(f"<div class='case'><h3>{identifier} · correct pair rank {case['correct_pair_rank']}</h3><img loading='lazy' src='previews/{identifier}.png' alt='Query well first-site five-channel preview'><p>{esc(case['query_description'])}</p><ol>")
                for candidate in case["top_k"]:
                    exact = " exact" if candidate["is_exact_pair"] else ""
                    candidate_id = esc(candidate["sample_id"])
                    parts.append(f"<li class='candidate{exact}'><strong>{candidate_id}</strong> <span class='score'>cos={candidate['cosine_similarity']:.4f}</span><p>{esc(candidate['description'])}</p><img loading='lazy' src='previews/{candidate_id}.png' alt='Retrieved well five-channel preview'></li>")
                parts.append("</ol></div>")
            parts.append("</details>")
        parts.append("</details>")
    parts.append("</html>")
    (output_dir / "index.html").write_text("\n".join(parts), encoding="utf-8")


def evaluate(manifest_path: str | Path, features_dir: str | Path, training_dir: str | Path,
             output_dir: str | Path, device: str = "cuda",
             biomedclip_path: str | Path | None = None) -> dict:
    manifest, arrays, metadata = load_features(manifest_path, features_dir)
    protocol, receipts = validate_training(training_dir, manifest["manifest_id"], metadata["feature_id"])
    splits = split_indices(manifest)
    output_dir, training_dir = Path(output_dir), Path(training_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostic_model_hash = None
    if biomedclip_path is not None:
        diagnostic_model_hash = model_identity(biomedclip_path)["sha256"]
        if diagnostic_model_hash != metadata["model_identities"]["biomedclip"]["sha256"]:
            raise ValueError("Diagnostic text model differs from frozen training text encoder")
    evaluation_binding = {"manifest_id": manifest["manifest_id"], "feature_id": metadata["feature_id"],
                          "protocol_id": protocol["protocol_id"],
                          "training_complete_sha256": hash_file(training_dir / "complete.json"),
                          "evaluation_code": hash_file(__file__),
                          "diagnostics_code": hash_file(Path(__file__).with_name("diagnostics.py")),
                          "diagnostic_model_sha256": diagnostic_model_hash,
                          "diagnostics": bool(biomedclip_path)}
    binding_path = output_dir / "binding.json"
    if binding_path.exists() and read_json(binding_path) != evaluation_binding:
        raise ValueError("Evaluation directory belongs to different frozen inputs or code")
    write_json(binding_path, evaluation_binding)
    images = torch.from_numpy(arrays["image_features"]).to(device)
    texts = torch.from_numpy(arrays["text_features"]).to(device)
    samples = manifest["samples"]
    test_samples = [samples[i] for i in splits["test"]]
    test_idx = splits["test"]
    diagnostics = None
    diagnostic_features = None
    if biomedclip_path is not None:
        from .encoders import FrozenBiomedCLIP
        encoder = FrozenBiomedCLIP(biomedclip_path, device=device)
        diagnostic_features = encoder.encode(diagnostic_texts()).to(device)
        diagnostics = {"frozen": score_diagnostics(diagnostic_features.cpu().numpy()), "projected": {}}
        del encoder
    runs, all_cases, artifacts = [], {}, {}
    for receipt in receipts:
        name = receipt["name"]
        model, checkpoint = load_run_model(training_dir / name / "best.pt", device)
        if (checkpoint["manifest_id"] != manifest["manifest_id"] or
                checkpoint["feature_id"] != metadata["feature_id"]):
            raise ValueError("Checkpoint does not match evaluated features")
        with torch.no_grad():
            image_embeddings = model.encode_image(images).cpu().numpy()
            text_embeddings = model.encode_text(texts).cpu().numpy()
            if diagnostic_features is not None:
                diagnostics["projected"][name] = score_diagnostics(model.encode_text(diagnostic_features).cpu().numpy())
        if not np.isfinite(image_embeddings).all() or not np.isfinite(text_embeddings).all():
            raise FloatingPointError("Nonfinite exported embeddings")
        for embeddings in (image_embeddings, text_embeddings):
            if not np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-5):
                raise ValueError("Exported embeddings are not unit normalized")
        gallery_name = f"{name}_gallery.npz"
        temporary = output_dir / (gallery_name + ".tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, sample_ids=arrays["sample_ids"],
                                well_ids=np.asarray([sample["well_id"] for sample in samples]),
                                splits=np.asarray([sample["split"] for sample in samples]),
                                sources=np.asarray([sample["source"] for sample in samples]),
                                descriptions=np.asarray([sample["description"] for sample in samples]),
                                image_embeddings=image_embeddings, text_embeddings=text_embeddings)
        temporary.replace(output_dir / gallery_name)
        artifacts[gallery_name] = hash_file(output_dir / gallery_name)
        similarities = image_embeddings[test_idx] @ text_embeddings[test_idx].T
        metrics = retrieval_metrics(similarities)
        runs.append({"name": name, "arm": receipt["arm"], "seed": receipt["seed"],
                     "best_step": receipt["best_step"], "validation_mrr": receipt["validation_mrr"],
                     "checkpoint_path": str((training_dir / name / "best.pt").resolve()),
                     "gallery_path": gallery_name,
                     "checkpoint_sha256": receipt["artifacts"]["best.pt"], "test": metrics,
                     "by_source": source_metrics(similarities, test_samples),
                     "embedding_diagnostics": {
                         "image_mean_dimension_std": float(image_embeddings[test_idx].std(axis=0).mean()),
                         "text_mean_dimension_std": float(text_embeddings[test_idx].std(axis=0).mean()),
                         "test_cosine_std": float(similarities.std()),
                     }})
        all_cases[name] = {direction: topk_cases(similarities, test_samples, direction)
                           for direction in ("image_to_text", "text_to_image")}
        print(f"{name} TEST mean_MRR={metrics['mean_mrr']:.5f} I2T_R1={metrics['image_to_text']['R@1']:.5f} T2I_R1={metrics['text_to_image']['R@1']:.5f}", flush=True)
    write_json(output_dir / "top5.json", all_cases)
    artifacts["top5.json"] = hash_file(output_dir / "top5.json")
    for sample in test_samples:
        _well_preview(sample, output_dir / "previews" / f"{sample['sample_id']}.png")
    artifacts.update({str(path.relative_to(output_dir)): hash_file(path)
                      for path in sorted((output_dir / "previews").glob("*.png"))})
    fields = ["run", "arm", "seed", "best_step", "validation_mrr", "direction", "candidate_count", "R@1", "R@5", "R@10", "MRR"]
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for run in runs:
            for direction in ("image_to_text", "text_to_image"):
                values = run["test"][direction]
                writer.writerow({"run": run["name"], "arm": run["arm"], "seed": run["seed"],
                                 "best_step": run["best_step"], "validation_mrr": run["validation_mrr"],
                                 "direction": direction,
                                 **{key: values[key] for key in fields if key in values}})
    artifacts["metrics.csv"] = hash_file(output_dir / "metrics.csv")
    evaluation = {"schema_version": 1, **evaluation_binding, "completed_at": utc_now(),
                  "split_counts": {key: len(value) for key, value in splits.items()},
                  "tie_policy": "stable_manifest_order", "random_expectation": random_expectation(len(test_samples)),
                  "runs": runs, "summary": _summary(runs), "diagnostics": diagnostics,
                  "artifacts": artifacts,
                  "interpretation": "New-plate internal paired retrieval. Similar valid captions need not identify a unique well. Seed std is not a confidence interval."}
    write_json(output_dir / "evaluation.json", evaluation)
    _write_html(output_dir, evaluation, all_cases)
    write_json(output_dir / "complete.json", {"manifest_id": manifest["manifest_id"],
               "feature_id": metadata["feature_id"], "protocol_id": protocol["protocol_id"],
               "completed_at": utc_now(), "evaluation_sha256": hash_file(output_dir / "evaluation.json"),
               "report_sha256": hash_file(output_dir / "index.html")})
    return evaluation
