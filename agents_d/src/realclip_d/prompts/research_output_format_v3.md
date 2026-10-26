OUTPUT FORMAT
Return a JSON object matching the supplied four-stage schema.

- compound_identification: summarize the compound identity and remaining ambiguity.
- compound_research_and_analysis: summarize source-reported findings and reported
  conditions, then state research gaps. Keep original predictions in morphology_hypotheses.
- morphology_hypotheses: give each change an ID, explain its basis by separating
  reported observations from your own inferences, and discuss applicability and uncertainty.
- observation_guidance: provide one entry per hypothesis explaining where to look,
  what to check and how treated/control comparisons could support or challenge it.
- sample_id: retain the supplied sample ID.

Sources and retrieval records are saved separately by the application. Do not
produce source IDs, URLs or a bibliography. The program saves the returned findings
without a second model call. Empty findings, hypotheses and guidance are valid.
