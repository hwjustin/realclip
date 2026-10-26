# Historical four-stage research output (V3)

Current research uses the [simple source-free V4 flow](RESEARCH_AGENT_SIMPLE.md).
This document describes historical `ResearchResultV3` files. The model returns exactly six top-level
fields: `sample_id`, four research stages, and a shared `sources` list.

The [research prompt](../src/realclip_d/prompts/research_v3.md) is unchanged.
The short [output note](../src/realclip_d/prompts/research_output_format_v2.md)
is a separate developer message. Historical runs may also contain a saved
`runtime_instructions.md`; current research does not use it. The SDK supplies the
[strict JSON schema](research_output_schema_v3.json).

| Stage | Fields | Purpose |
|---|---|---|
| `compound_identification` | `summary`, `source_ids` | Identity, useful identifiers, synonyms, and remaining ambiguity |
| `compound_research_and_analysis` | `findings`, `research_gaps` | Source-reported properties, mechanisms, cellular effects, and morphological observations |
| `morphology_hypotheses` | `id`, `change`, `basis`, `applicability`, `source_ids` | Predicted appearance, evidence and inference, applicability and uncertainty |
| `observation_guidance` | `hypothesis_id`, `where_to_look`, `what_to_check`, `how_to_compare` | Inspection locations, features, and treatment/control comparisons that support or challenge the prediction |

Each source finding contains `finding`, `reported_conditions`, and `source_ids`.
Describe unreported or irrelevant conditions explicitly. `basis` is prose that
separates what a source directly reports from the agent's predictions, including
extrapolation from related compounds or different experimental conditions.
`applicability` discusses the supplied experiment and its uncertainty. There are
no forced evidence categories or review-status enums. Input MOA can support an
inference without becoming a retrieved source or a mandatory duplicated fact.

The [article-reading tool](READ_ARTICLE.md) supports overview, agent-selected
sections, and full-text modes. It exposes the complete outline and sequential
pages with continuation arguments; no keyword selection is applied.

`search_literature` searches Europe PMC for compound properties, mechanisms of
action, molecular targets, and cellular effects, including reported morphology.
Its tool description asks the agent to consider experimental context where
relevant and select papers for further reading. The query is chosen by the agent;
it need not include morphology terms. The parameter descriptions explain the
query scope and the requested result count (default 10, range 1–100).

Each source has only `id`, `title`, and `url`. Copy the tool-returned `source_id`
as `id`, preserving its title (including null) and URL. Do not invent local source
aliases. Retrieval time, cache path, source kind, raw receipts, and access metadata
remain in local tool records, outside the model-generated result.

## Validation

All declared fields are required and extra fields are forbidden. Narrative
strings must be nonempty. Findings and hypotheses can be empty, as can guidance
when no hypotheses are returned. A finding requires at least one source; a
hypothesis may have no retrieved source when its premise is supplied input MOA.
Every source reference must exist. Source IDs and hypothesis IDs must be unique.
Each hypothesis requires exactly one guidance entry; extra or duplicate guidance
is rejected. Sources must match authentic tool/web receipts in ID, title, and URL.

The workflow also checks the sample ID, an actual identity-tool attempt, identity
source provenance, and successful retrieval receipts for cited findings and
hypotheses. It does not restore the old classification/status requirements or
try to validate scientific claims by reading prose labels. Structural and source
checks do not establish biological accuracy, source fidelity of a claim, or
whether the text adequately distinguishes observation from inference; these
still require scientific review. Existing English input/output guards remain.

See the [synthetic example](../examples/research_v3_synthetic_example.json).
It is a manually constructed format illustration, not a model run or scientific
evidence, and cannot replace a sealed research directory for `describe`.

## Saved runs and compatibility

The result contains no schema-version field. New stages save version information
in `output_contract.json`, the exact schema in `output_schema.json`, and the
separate format note in `output_format.md`; these are bound by the stage seal.
Use a fresh output directory for the new contract. V1/V2 results and prompts remain
readable without conversion or changes to saved runs. Saved-evidence synthesis
selects the original request's output contract and saved note, rather than
silently upgrading it or repeating retrieval.

The existing observation adapter can project the new appearance and guidance
fields without exposing research findings, source citations, or the private
basis. This migration does not redesign or execute the observation agent.

See [research continuity](RESEARCH_CONTINUITY.md) for local evidence paging,
reading progress, scientific notes, context checkpoints, and offline tests.
