"""Predetermined treatment-only morphology paraphrase/opposite diagnostics."""
from __future__ import annotations

import numpy as np


MORPHOLOGY_CONTROLS = (
    ("sparse_dense", "Cells are sparsely distributed with large empty gaps.",
     "Widely spaced cells leave broad unoccupied regions.", "Cells are densely packed with few empty gaps."),
    ("round_elongated", "Nuclei have predominantly round outlines.",
     "Most nuclear profiles are circular.", "Nuclei have predominantly elongated outlines."),
    ("dispersed_clustered", "Bright puncta are dispersed throughout cell regions.",
     "Bright dots are scattered across cellular areas.", "Bright puncta are concentrated in compact clusters."),
    ("fine_coarse", "Cell regions contain a fine granular texture.",
     "Tiny granules produce a finely textured cellular pattern.", "Cell regions contain coarse granular clumps."),
    ("smooth_irregular", "Nuclei have smooth and regular outlines.",
     "Nuclear boundaries are even and smoothly contoured.", "Nuclei have jagged and irregular outlines."),
    ("continuous_fragmented", "Filamentous staining forms long continuous strands.",
     "Elongated stained threads remain connected along their lengths.", "Filamentous staining forms short disconnected fragments."),
    ("bright_dim", "Nuclear profiles are bright against the surrounding background.",
     "Nuclei show conspicuously luminous staining over a darker background.", "Nuclear profiles are dim and barely visible against the background."),
    ("filamentous_punctate", "Mito staining forms elongated filamentous structures.",
     "The mitochondrial channel shows long thread-like stained patterns.", "Mito staining forms isolated round puncta."),
)


def diagnostic_texts() -> list[str]:
    return [text for _, *sentences in MORPHOLOGY_CONTROLS for text in sentences]


def score_diagnostics(embeddings: np.ndarray) -> dict:
    embeddings = np.asarray(embeddings, dtype=np.float64)
    if embeddings.ndim != 2 or len(embeddings) != len(MORPHOLOGY_CONTROLS) * 3:
        raise ValueError("Unexpected morphology diagnostic embedding shape")
    if not np.isfinite(embeddings).all():
        raise ValueError("Nonfinite diagnostic embeddings")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("Zero diagnostic embedding")
    embeddings = embeddings / norms
    rows = []
    for i, (name, anchor, paraphrase, opposite) in enumerate(MORPHOLOGY_CONTROLS):
        a, p, o = embeddings[3 * i:3 * i + 3]
        positive, negative = float(a @ p), float(a @ o)
        rows.append({"name": name, "anchor": anchor, "paraphrase": paraphrase,
                     "opposite": opposite, "paraphrase_cosine": positive,
                     "opposite_cosine": negative, "margin": positive - negative,
                     "paraphrase_preferred": positive > negative})
    return {"n_controls": len(rows), "n_paraphrase_preferred": sum(row["paraphrase_preferred"] for row in rows),
            "mean_margin": float(np.mean([row["margin"] for row in rows])), "controls": rows}
