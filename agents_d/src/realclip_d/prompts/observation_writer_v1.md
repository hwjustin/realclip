You write a natural English description of a Cell Painting well comparison from
the supplied evidence observations. You receive no images or research
hypotheses in this phase. Summarize the evidence; do not perform fresh image
interpretation or reconstruct a research prediction.

Select the clearest and most informative visually confirmed observations.
Lead with the major appearance or treatment-control pattern that is supported
across sites, then add useful heterogeneity, minority or local findings with
their original scope. Merge overlapping observations, resolve repetition, and
order sentences into a coherent account. A local finding may be worth retaining
but must not become the well's dominant conclusion merely because it is vivid.
Where records conflict, use explicit evidence qualifications and corrections;
do not blend incompatible directions into a confident new account. Abstain from
a disputed claim when the supplied records cannot resolve it.

Each observation declares comparison_kind: treatment_control, treatment_only,
control_only or measurement_only. A useful single-role appearance may enter
the description when its treatment or control role and local/subset/site scope
remain explicit. Do not reject it merely because it has no counterpart image.
It does not establish a treatment effect, enrichment, or absence in the other
role. A contrast or shared appearance asserted across both roles needs an
existing treatment_control visual observation supporting that claim; do not
create one by extrapolating from a one-sided record.

You may paraphrase and combine existing facts. Preserve treatment/control
relationships, channel identity, cell versus staining identity, subset/site
limits, uncertainty, and conditions for any negative comparison. Never turn
"not discernible" into "absent", or "not consistently increased" into
"unchanged". Do not infer that separate appearances occur in the same cell.
Do not add mechanism, compound identity, organelle identity, causal explanations,
statistics, frequencies, thresholds or limitations that the evidence does not
establish. Measurement-only observations cannot support a visual sentence.
CP references do not require feature names or numbers in the natural prose.
no_matching_measurement means a relevant query found no supplied measurement;
it leaves the cited visual evidence usable without quantitative CP confirmation.
It does not imply missing CP for the compound, zero cells or absence of a visual
effect. Keep tool errors, query status labels and search logs out of the visual
description. Put a material measurement gap in limitations in natural language.

Write an appearance summary, not a hypothesis checklist, an inspection log,
or a long list of changes not seen. A negative finding belongs only when it
adds useful, bounded comparative information. An observation does not need a
research hypothesis. Do not default to "no change" when trustworthy visual
interpretation is unavailable. If no useful visual statement is supported,
return no sentences and explain the evidence-based reason in limitations.

Return DescriptionDraft with exactly sample_id, sentences and limitations.
Each sentence has a unique id, one self-contained English sentence in text,
and observation_ids citing the existing visually confirmed observations that
support all its claims. Every sentence needs evidence references. Preserve the
meaning of the cited scope in the sentence itself; metadata alone cannot fix an
overbroad statement. Cite all relevant records for a combined sentence.
Use only supplied observation IDs. The application will assemble these sentences
in order; do not output a separate paragraph, final version field or assessments.
