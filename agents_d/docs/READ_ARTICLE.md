# Agent-directed article reading

`read_article` supports three modes. It performs no compound-name, morphology,
mechanism, or other keyword filtering. The agent chooses its reading scope.

| Mode | Content shown to the model |
|---|---|
| `overview` (default) | Abstract page, complete section outline, full-text access status, and linked-supplement information |
| `sections` | Text of the selected section IDs, including subsections, in document order |
| `full_text` | All extracted article text, in document order, starting at the requested offset |

The agent may start with overview or request full text directly. The tool
description asks it to use the abstract/outline to investigate compound
properties, mechanisms, cellular effects, morphology, and relevant experimental
conditions, expanding reading as needed.

## Parameters

| Parameter | Type / default | Meaning |
|---|---|---|
| `article_id` | string, required | Europe PMC record ID or PMCID |
| `source` | string, `MED` | Record source; PMCID lookup uses the PMCID itself |
| `mode` | `overview`, `sections`, `full_text`; default `overview` | Reading scope |
| `section_ids` | list of strings or null; default null | IDs returned in the outline; required and unique for sections mode, otherwise null or empty |
| `offset` | nonnegative integer, `0` | Character offset in the chosen text |
| `max_chars` | integer 1–24000, `12000` | Initial page size; research sections/full_text automatically expand locally up to 96000 characters |

The Agents SDK's strict tool schema requires explicit parameters, including those
with defaults. Each parameter has an English description. Invalid modes,
parameter combinations, duplicate IDs, and invalid limits are rejected before
HTTP retrieval. Unknown section IDs or out-of-range offsets return explicit
errors. No new dependency is required.

## Returned content

The standard `status`, `data`, `sources`, and `message` envelope remains. The
model view of `data` includes:

- `article`: record identifiers and bibliography, without a duplicate raw abstract.
- `mode`, `abstract_available`, `fulltext_status`, and `fulltext_error`.
- `sections`: the complete outline with IDs, titles, parent IDs, levels, character
  lengths, and positions in the normalized article. Untitled sections and body
  text outside named sections remain accessible. Selecting a parent includes its
  children; overlapping parent/child selections appear once.
- `linked_materials`: supplementary-material links found in XML, marked
  `not_retrieved`; linked files are not automatically fetched.
- `reading`: actual page `text`, scope, section IDs, offsets, total character
  length, content digest, `has_more`, and `next_request`.
- `_model_view`: retrieval and text-exposure information.

`reading.next_request` contains all arguments for the next call. Keep its mode,
section IDs, and page size. Concatenating successive pages reconstructs the exact
selected normalized text. Pages prefer paragraph boundaries; a paragraph longer
than the limit is split at the character boundary with a continuation. Overview
also pages long abstracts, while keeping the full outline available.

In the research agent, sections/full_text automatically expand the requested
selection up to 96,000 characters in one tool response. Expansion uses the
already retrieved document and makes no extra model or HTTP calls. Ordinary
selections therefore finish without relying on the agent to ask for a small
remainder. Longer selections retain exact offsets and a valid `next_request`
with the original page size. `response_character_limit`,
`auto_read_additional_characters`, and an explicit continuation reason describe
the behavior. The low-level Python client defaults to manual paging; pass
`auto_read=True` to use the research behavior. Overview remains manually paged.

Article pages and outlines bypass generic list/string clipping. They never use
the old eight-window or 6000-character abstract selection. Findings and retrieval
records are saved as described in [research continuity](RESEARCH_CONTINUITY.md).
Successful official HTTP responses are
cached, so successive reads reuse the same response snapshots within a run.

## Full-text scope and exposure

Extracted full text includes available XML abstracts, body, back matter (including
references), and floating figure/table text. Inline text, table cells, captions,
and table footnotes are retained. Missing metadata abstracts can fall back to
the XML abstract. XML namespaces are supported. This is text extraction, not
figure-image interpretation or PDF/supplement downloading. The returned scope
describes this boundary explicitly.

`fulltext_status` separates accessible XML, missing PMC full text, failed
retrieval, invalid XML, and XML without readable body text. Overview can succeed
with only an abstract; sections/full_text return `unavailable` when body text
cannot be read. The raw XML, parsed document, complete tool result, and exact
model view remain in local receipts/cache; raw XML and unrequested text are not
sent to the model.

`reading.full_text_complete_in_this_response` is true only when that response
contains the entire nonempty full-text stream. The recorder separately tracks
unique full_text-mode ranges per document/source in the current run and sets
`_model_view.full_text_complete` only when coverage has no gaps. Repeated pages
do not increase coverage, and starting at a later offset cannot establish full
coverage. Section reading does not silently imply full-text coverage. These
records describe text provided through the tool, not model comprehension.

The four-stage research output schema is unchanged. Reports and saved-evidence
packets retain the exact already-shown reading pages and outline. Historical
`body_windows` records remain readable without regenerating their contents or
repeating retrieval. All development checks use local fixtures or mock HTTP
responses; no real model run is needed to test the tool.

## Findings and backend records

Article modes and continuation arguments remain available. The tool description
asks the agent to follow `reading.next_request` for overlong selections or explain
why further reading is unnecessary. There is no separate reading-task
queue or stop-reading tool, and reading coverage does not gate a final result.
The application retains source envelopes, complete receipts and exposure records
in the backend; the agent's source-envelope list is empty. See the
[simple research flow](RESEARCH_AGENT_SIMPLE.md) for finding persistence.

## English presentation of external text

The source XML and parsed original document stay unchanged in backend receipts.
The model view uses source-provided English titles and English abstracts when
available. Explicitly translated duplicate abstracts are omitted when the XML
also provides an English abstract. Mixed-language scientific passages without
an English counterpart are marked unavailable, rather than silently shortened
or translated. Section IDs remain stable; offsets and hashes are calculated
from the English presentation before paging, including on continuation calls.
`language_projection` records the omissions. Complete English-presentation
coverage does not set original full-text coverage to complete. The complete
model-request English guard remains enabled; no extra model call is introduced.
