# Current two-agent workflow

## Research Agent

Entry: `agent_runtime.run_research`; scientific prompt:
`src/realclip_d/prompts/research_v3.md`; separate output instructions:
`research_output_format_v3.md`; result contract: `ResearchResultV4`.

Inputs contain compound identifiers, supplied MoA and context. Current dataset
research used U2OS with unknown dose and treatment duration. Research receives no
images or CP measurements. A single `Runner.run` owns the ordinary model/tool
loop; it is not split into findings and synthesis conversations.

Tools are official PubChem PUG-REST/PUG-View identity and annotations, ChEMBL
mechanisms/activities/assays/targets/documents, and Europe PMC search and article
reading. Article modes are overview, selected sections and full text. Only the
OpenAI provider can additionally use hosted web search. The model chooses tools;
local Python performs HTTP requests, rate limiting, caching and source retention.

The four output stages are identification; source findings/research gaps;
shape hypotheses with basis and applicability; and where/what/how observation
guidance. Findings are copied to `findings.json` by the program. Raw API responses,
full tool receipts and exact model views remain in the run directory. There are
no independent notes/memory agents or citation-ID hard gates in this version.

## Handoff

`observation_protocol.prepare_observation` creates a separate appearance-only
view. It projects hypothesis change, applicability and inspection guidance;
research findings, compound identity, MoA and hypothesis basis are excluded.
A hash-bound revision can replace leaking text with reviewed appearance language.
This is the current deliberate boundary, not a complete biological anonymizer.

## Observation Agent

Entry: `observation_pipeline.run_pipeline` through `run_observation`.

| Internal module | Context and work |
| --- | --- |
| Baseline | Actual treatment/control images; independent visual observations and native-pixel inspection |
| Guided | Baseline, appearance hypotheses, images, crop tool and CP queries; can reject predictions or add unexpected findings |
| Writer | Visual evidence; no research-hypothesis checklist and no pixels; assembles sentences with evidence associations |
| Reviewer | Draft, visual evidence and images/crops; checks support, scope and conflicting observations |
| Summarize | Reviewed, eligible treatment appearances and target image scope; produces a short plain-English description |

Prompts are `observation_{baseline,guided,writer,reviewer}_v1.md` and
`summarize_v1.md`. Source images and returned crops are actual image input, not
text descriptions pretending to be images. `query_cp` reads the supplied CSV;
CP-only changes cannot establish visual confirmation. No network-research tools
are exposed to Observation.

The standard Cell Painting panel is DNA/RNA/ER/AGP/Mito. AGP combines actin,
Golgi and plasma membrane. Standard images do not directly identify tubulin,
lysosomes, target engagement, histone acetylation or a cell-death mechanism.

Outputs include the full reviewed report, observation and image/CP evidence,
module-level requests/responses/usage, and `observation/description.txt`. The
final short caption describes the selected treatment images. It deliberately
omits compound names, target labels and mechanisms. Research mechanism knowledge
is therefore not explicit mechanism-text supervision in the current caption.

## Portable model interface

The original profile is `provider=openai`, model `gpt-6-luna`, reasoning high.
The portable addition is `provider=openai_compatible`, a configurable served
model and base URL. `local_provider.LocalChatModel` uses the SDK Chat Completions
adapter. It re-exposes tool-returned images in associated user image messages,
normalizes provider usage, and calls the local server's `/tokenize` before any
generation reservation. Local reasoning controls are model-specific and can be
supplied through `local_extra_body`; OpenAI reasoning settings are not sent.

Research can use a text-only endpoint and Observation a VLM endpoint by running
`research` and `describe` with different configs. Both may also share one VLM.
Each stage preserves its own configuration and receipts. The agent client does
not allocate H200 memory; model serving belongs in a separate environment.

## Known research-quality work

The broader mechanism-first pilot is opt-in material under
`research_mechanism_first.experimental.md`. A diagnostic trial found both
related-compound identity drift and a long-context local-byte guard failure.
A corrected identity-anchored sample recovered a retraction notice. Broader
coverage is not established merely by producing more findings.

The current compact Europe PMC model view omits publication-type/correction
metadata present in raw records; systematic retraction-status retention is a
known follow-up. The current research-stage cloud profile retains a conservative
byte bound, while the new local profile uses server token counting. Keep these
limitations visible when evaluating a replacement model. No pilot result or
experimental prompt is silently adopted as the production default.
