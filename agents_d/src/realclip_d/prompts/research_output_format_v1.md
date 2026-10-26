OUTPUT FORMAT
Return a JSON object matching the supplied schema.

- Include compound identity and context, sources, findings, morphology hypotheses,
  observation guidance, and limitations.
- Copy source records from tool results and use consistent source, fact, and
  hypothesis IDs. Label findings by origin, compound scope, and actual access.
- Preserve each supplied known_moa entry verbatim as a provided_moa fact with
  access=input and no source IDs. Keep source findings separate from predictions.
- For each hypothesis, include its basis, supporting fact IDs, rationale,
  applicable conditions, uncertainty, and guidance on where to look, what to
  examine, and how to compare treated and control images.
- Keep compound names, mechanisms, and citations in the research fields and
  rationale; keep observation-facing fields focused on visible appearance,
  experimental conditions, and comparisons.
- List all literature-origin fact IDs in literature_review and inference-supporting
  fact IDs in mechanism_review. Match review statuses to the returned evidence
  and hypotheses; hypotheses_proposed requires a moa_inference hypothesis.
