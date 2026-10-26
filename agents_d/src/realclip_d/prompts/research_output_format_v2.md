OUTPUT FORMAT
Return a JSON object matching the supplied four-stage schema.

- compound_identification: summarize the identity, useful identifiers, and any
  unresolved ambiguity; cite identity sources.
- compound_research_and_analysis: summarize source-reported findings and their
  reported conditions, then state research gaps. Keep original predictions in
  morphology_hypotheses. If conditions are unreported or irrelevant, say so.
- morphology_hypotheses: give each change an ID, explain its basis by separating
  reported observations from your own inferences, and discuss applicability and
  uncertainty in the supplied experiment. Cite supporting sources; an input-MOA
  premise is an input, not a retrieved source.
- observation_guidance: provide one entry per hypothesis explaining where to
  look, what to check, and how treated/control comparisons could support or
  challenge the hypothesis. Keep change, applicability, and guidance focused on
  appearance and experimental conditions; keep mechanisms and citations in basis
  and research findings.
- sources: reuse each tool-returned source_id as id and copy its title and URL.
  Do not generate cache paths, retrieval times, status labels, or schema versions.
  Empty findings, hypotheses, and guidance are valid when evidence is insufficient.
