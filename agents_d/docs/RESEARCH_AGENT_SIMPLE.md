# Current research agent: one complete research run

The approved scientific [research prompt](../src/realclip_d/prompts/research_v3.md)
is unchanged, including all four stages. Its SHA-256 is
`2ee97f9ea07e849090b30fad70db9ac0a8b3ac2c4a59b821b843a5c463175fad`.

`run_research` defines one researcher and calls `Runner.run` once. That run includes
the SDK's ordinary model/tool loop and returns the complete four-stage result.
There is no second findings pass, synthesis pass, intermediate summarizer,
checkpoint conversation, or automatic context replacement.

## Tools and context

The researcher receives compound identity, supplied MoA and experimental context.
It uses `resolve_compound`, `get_pubchem_annotations`, `get_chembl_evidence`,
`search_literature` and `read_article`, plus hosted web search when configured.
Existing article overview/sections/full_text modes and their continuation
arguments remain available.

Search presentation preserves all retrieved candidate titles and identifiers,
removing bulky author/affiliation/index metadata. Complete abstracts are returned
without string clipping; large batches expose explicit abstract pagination.
`next_request` continues abstracts within the same cached page using
`result_offset`; `next_page_request` uses a Europe PMC `cursor` to retrieve more
database hits. A single oversized abstract is explicitly accessible through
`read_article` overview. Access flags are provider metadata, not proof of a
successful full-text read. Original responses remain in backend receipts.

Research article sections/full_text expand locally up to 96,000 characters per
response, with an exact continuation for longer selections. This is tool-side
text assembly, not another model pass, a memory tool, or a reading-completion
gate. The scientific prompt, result schema and single `Runner.run` are unchanged.

The paired live check is saved in
`runs/research_retrieval_fix_2_20261021_005700/`. Both JNJ-7706621 and parbendazole
completed using inputs identical to the earlier sixteen-worker run. The report
records fuller mechanism/cellular evidence, the complete English outputs, costs,
and remaining scientific limitations. Full-dataset generation was not started.

The separate notes, memory-index reader, source-registry tool and reading-stop
tool have been removed from the active agent. Tool receipts and scientific
content use the existing presentation views. Source envelopes are retained in
backend records rather than passed to the model for copying into its answer.

## Output

New runs use `ResearchResultV4`, with the same four scientific stages:

| Field | Contents |
|---|---|
| `sample_id` | Supplied sample ID |
| `compound_identification` | `summary` |
| `compound_research_and_analysis` | `findings`, `research_gaps` |
| Each finding | `finding`, `reported_conditions` |
| Each `morphology_hypotheses` item | `id`, `change`, `basis`, `applicability` |
| Each `observation_guidance` item | `hypothesis_id`, `where_to_look`, `what_to_check`, `how_to_compare` |

There are no model-authored `source_ids`, `sources`, URLs or bibliography fields.
The short [output-format note](../src/realclip_d/prompts/research_output_format_v3.md)
is separate from the unchanged scientific prompt. Empty findings and hypotheses
are valid when evidence is insufficient.

## Findings and provenance

After the research run completes, Python copies the returned
`compound_research_and_analysis` unchanged into `findings.json`. That file contains
the exact findings, reported conditions and research gaps from `result.json`.
Saving it makes no additional model request. Findings are not a parallel notes
schema and are not rewritten by another model.

`sources.json`, `tools.jsonl`, raw API caches, requests and responses retain
retrieval provenance. `tools.jsonl.result` includes the complete tool receipt and
its source records; `model_result` shows the presentation returned to the agent.
The model does not need to reproduce backend source IDs, titles or URLs.

New runs check output structure, sample ID, and the association between hypotheses
and observation guidance. They do not run citation-ID, title, URL, identity-source
classification or literature-reading-completion gates. `validation.json` records
`reference_validation: "not_required"`; these checks do not assess biological truth.

Existing financial reservations, no automatic paid retry, unknown-usage stops,
English input/output boundaries, artifact seals and explicit resumption remain.
The final permitted model turn disables tools so the agent can return the complete
result. There is no new context-management behavior. The retired
`research_context_bytes` setting is accepted for old configuration compatibility
and is ignored by the current research run.

Historical V1/V2/V3 files remain readable under their saved contracts. They are
not rewritten, upgraded or regenerated. The observation adapter accepts the new
source-free result using the existing appearance/guidance projection.

The ChEMBL model view defers only bulky PDB/PDBe target-component structure-index
cross-references to the full backend receipt, with exact per-component/database
counts. Target identities, all components, functional annotations such as GO and
Reactome, mechanism/activity records, assay conditions and documents remain.
For large repeated functional annotation lists, a shared `target_annotation_records`
table within the same response retains every complete annotation once; each
component keeps ordered zero-based `record_indexes`, including repeated entries.
This is lossless representation, requires no extra tool call, and has no state
outside the current response.
This changes neither the research prompt nor the single SDK research run.
External English presentation is documented in [article reading](READ_ARTICLE.md).
`research_retry.prepare_technical_rerun` prepares only an explicit technical-failure
allowlist in a separate directory, preserves original inputs/cached source bytes,
excludes completed research, and rejects provider-policy failure cases. It does
not invoke models or release historical unknown-charge reservations.
