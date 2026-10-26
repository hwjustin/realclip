OUTPUT FORMAT DRAFT

This is a proposed contract, not the currently active DescriptionResult schema.
The synthetic example illustrates the proposed model output only.

Return exactly these five top-level fields:

- sample_id: copy the supplied anonymous sample ID.
- observations: a list of records with id, text, scope, image_ids,
  cp_feature_ids, and visually_confirmed. text is one self-contained English
  sentence describing a bounded visual comparison or a measurement-only finding.
  scope specifies field/well/subset/local extent and any relevant heterogeneity.
- hypothesis_assessments: exactly one record for every supplied hypothesis_id,
  with hypothesis_id, assessment, checks, and reason. Each check has check,
  assessment, observation_ids, and reason. Every hypothesis has at least one
  check. Use concise appearance components rather than inventing new hypotheses.
- description_observation_ids: the unique IDs of selected visually confirmed
  observations, in reading order. This list may be empty.
- limitations: a list of concise English limitations, or an empty list.

Overall assessment is supported, partially_supported, not_supported, or uncertain.
Check assessment is supported, not_supported, or uncertain. Give every check a
reason; an unavailable check may have no observation IDs. A supported visual check
must reference relevant visual observations. CP alone cannot support it visually.
Explain conflicts, partial support, missing checks, and the scope of conclusions.

All references must exist. Observation IDs and hypothesis assessment IDs must be
unique; a check may reuse observations referenced by another hypothesis. A visual
comparison cites its treatment and control images, with any auxiliary references
kept appropriate to the comparison. CP IDs must have been returned to this agent.

Do not write a separate description paragraph. The application generates
description by joining the selected observation sentences with one space. It
retains this selection as the sentence-to-evidence mapping. An empty selection
produces an empty description and requires a reason in limitations. The
application adds contract/version metadata and run records separately.

Interpretation of images, scientific validity, and semantic leakage cannot be
established by ID checks alone. This contract does not make the illustrative
synthetic example a validated research or observation result.
