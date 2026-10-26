# Shared task context draft

Status: design only. This text has not been injected into any running module. Existing Research and Observation prompts and behavior remain unchanged.

## Purpose

Cell Painting and the eventual image–text training target are shared project context, not information exclusive to Summarize. Every stage should understand the unit ultimately being described, while keeping its existing evidence access and responsibilities.

The intended final alignment is:

> One treatment well's actual multi-site, multi-channel image collection ↔ one short caption describing appearances supported by that treatment well.

Controls may inform upstream observation and interpretation. Intermediate reports may retain treatment–control comparisons, control-only findings, CP assessments and limitations. The shared target does not prohibit those records. The terminal caption has a narrower scope: treatment appearances that can be supported by the image collection actually supplied to the CLIP image encoder.

## Proposed reusable context

The following English block is a proposed common preamble. It is not an active prompt change.

```text
PROJECT CONTEXT

This project studies Cell Painting microscopy. A well contains multiple sampled
sites, and each site may have multiple imaging channels. Channel identity,
staining composition, site identity and image role are supplied by metadata.
Sites within a well are sampled fields, not independent replicate wells.
The same site index in treatment and control does not identify the same cells.
Cross-channel cell correspondence requires explicit alignment and visible support.
Do not infer a channel's biological specificity from its name alone, especially
for mixed stains. A fluorescence pattern does not by itself establish an
organelle identity, cellular process or mechanism.

The eventual contrastive-learning pair is:
one treatment well's multi-site, multi-channel image collection paired with
one concise caption of appearances supported by that treatment well.

Control images can help discover, contextualize and check observations.
CellProfiler measurements can provide supporting or qualifying evidence when
their definitions and interpretation are established. Research hypotheses can
guide inspection when permitted by the current stage. None of these sources
substitutes for the treatment-image support required for a terminal visual
caption.

Intermediate observation and review reports may contain treatment-control
comparisons, control-only findings, measurement assessments and limitations.
Do not delete useful upstream evidence merely because it will not belong in
the terminal caption.

The terminal caption describes the treatment well's own supported appearances.
Do not convert a relative difference into an absolute appearance without
independent treatment evidence. Preserve local, subgroup and site-dependent
scope instead of generalizing a finding to the entire well.

The eventual image collection is defined by the supplied CLIP visual manifest,
including the sites, channels, views, source lineage and rendering used by the
image encoder. Images inspected elsewhere in the workflow are not automatically
part of that collection. A local detail seen in a crop is not automatically
supported by a reduced full-field rendering of the same well.

Follow the current stage's evidence-access rules and output contract. Sharing
the project goal does not grant access to additional images, hypotheses,
measurements or unselected findings. Report uncertainty rather than inventing
support or a distinctive phenotype.
```

## Shared context versus stage-specific responsibilities

| Stage | How the shared objective applies | Evidence boundaries remain |
|---|---|---|
| Research | Produce Cell Painting-relevant inspection guidance, aware of the eventual well-level visual caption. | Hypotheses are prospective guidance; research alone cannot establish an observed treatment phenotype. |
| Baseline observation | Record treatment and control appearances with channel, site and subgroup scope. | Keep its independent image-first inspection; do not add hypotheses or CP values solely because other stages know the objective. |
| Guided observation | Use permitted guidance and CP tools to investigate concrete appearances and refine their interpretation. | Preserve baseline provenance, actual query results and uncertainty; comparative findings remain valid intermediate evidence. |
| Writer | Organize the approved observation evidence into a readable full report. | It may describe comparisons and controls under the current contract; the short terminal caption is a separate output. |
| Reviewer | Check the report against the evidence and images permitted to that stage. | Do not add unsupported findings or silently change the target image scope. |
| Summarize | Extract and shorten treatment-supported appearances from the reviewed report for the actual CLIP input. | No new inspection, research, hypothesis assessment, CP query or recovery of findings omitted from the reviewed report. |

The table describes the intended design boundary. It does not change current stage instructions.

## Per-case context to carry alongside the shared text

The conceptual fields below are a design requirement, not an implemented schema:

- **Target treatment well:** a stable sample/well identifier and explicit treatment role. The training identity must remain well-specific rather than collapsing a compound's doses or time points.
- **Actual CLIP visual manifest:** included image/view identifiers; site and channel identity; source-image lineage; full field versus ROI; pixel region where relevant; rendering/resizing/cropping or channel-composition information; and whether the rendering is final or provisional.
- **Relevant limitations:** verified or unknown channel specificity, measurement interpretation and image visibility restrictions.
- **Length constraints:** preferred language, sentence count and soft word range; the chosen tokenizer and hard token budget when settled.

A neutral target identifier and imaging metadata are sufficient for Summarize. Do not include compound identity, target labels, research mechanisms or hypotheses as caption-generating material. An image path or identifier that encodes such a label should be presented with a neutral alias where practical.

If the CLIP input has not yet been fixed, this context must say that it is provisional. A resulting caption remains a candidate for the stated provisional collection; it must not be represented as a verified training pair.

An unresolved treatment-well identity, absent image list or unknown source mapping is a preflight configuration gap. Do not invoke Summarize to guess those bindings. A provisional collection must still be an explicit collection with known provenance.

## Application boundary

Adding this preamble later would require an explicit integration change and a check against each stage's existing instructions. This draft does not update shared-context builders, active prompts, schemas, evidence access, current reports or training code.
