# Research agent v2: from compound knowledge to testable morphological hypotheses

This page documents the historical V2 output contract and its validation rules.
New research runs use the simplified [four-stage V3 contract](RESEARCH_AGENT_V3.md).
Existing V2 results are retained unchanged; the V2 requirements below are not
imposed on new V3 outputs.

The research agent uses `gpt-6-luna` with reasoning effort `high` by default. It receives only compound identifiers, user-supplied known MoA, experimental conditions, and evidence returned by tools. It receives no images or CellProfiler data. Its output is a set of hypotheses to test, not an image description or a biological conclusion.

The current research prompt opens by describing its purpose: using compound identity, known mechanisms of action, and published evidence to develop testable hypotheses about cellular morphological changes. Project names and prompt-version labels are kept out of its introduction. It contains no language-policy block.

## Four research steps

1. **Identify the compound.** Establish the target compound's chemical identity using the supplied name or identifiers and available databases. Retrieve relevant synonyms and identifiers, report any remaining identity ambiguity, and use the supplied context to judge the relevance of subsequent research.
2. **Research and analyze compound information.** Use the supplied experimental context to assess relevance while building a broad understanding of the compound's properties, mechanisms, targets, and cellular effects relevant to morphology. Summarize and analyze findings from databases, literature, and other external sources, including reported links to morphological changes. Reserve original hypotheses and inferences for the next stage.
3. **Develop morphology hypotheses.** Combine the supplied MoA, the findings and analysis from the previous stage, and experimental context to predict potential visible changes in Cell Painting images. Explain the evidence behind each predicted appearance, distinguish source findings from predictions, and retain uncertainty where conditions or evidence are incomplete.
4. **Provide observation guidance.** For each hypothesis, describe where to look in Cell Painting images, which visual features and direction of change to examine, and how to compare treated and control images across comparable cells and fields. Explain which similarities or differences would support or challenge the hypothesis.

## Output format

The research prompt ends after observation guidance. A separate short
[`research_output_format_v1.md`](../src/realclip_d/prompts/research_output_format_v1.md)
describes how to organize the result; the SDK supplies the exact JSON schema.
The runtime sends this note as a separate developer message and saves it as
`research/output_format.md`, without adding it to the research instructions or
sample context. Saved-evidence continuations reuse the saved note when present;
older runs retain their original instructions.

## Evidence and applicability

| Hypothesis basis | Meaning | Required limitations |
|---|---|---|
| Direct literature observation | A read paper reports cellular morphology for the exact compound | Original cell system, dose, and time; whether current conditions match |
| Indirect literature evidence | A morphological observation concerns another molecule, analogue, or mechanism class | Extrapolation across molecules or conditions; cannot become a direct observation of the exact compound |
| MoA inference | Appearance inferred from known action, biochemical experiments, or mechanistic facts | Inference chain and uncertainty; does not represent an existing morphological report |

The contract values are `direct_literature`, `indirect_literature`, and `moa_inference`. A direct morphological report for the exact compound in another cell line remains a direct literature observation. That classification does not establish applicability to the current cell line, dose, or duration; extrapolation conditions must remain explicit.

Every research fact records its origin, compound scope, access level, experimental conditions, and limitations. Source conditions and current conditions are retained separately; unknown values stay unknown. Database parent relationships, biochemical experiments, database annotations, and input MoA do not become direct morphological evidence merely because they are cited.

Source records must match actual tool receipts. Search hits, abstracts, complete body text, and incomplete body text have distinct access statuses. Failed reads, unavailable full text, and truncated body text cannot support a claim of reading the complete paper. Cache references support manual inspection of raw responses; the program cannot prove that the model's scientific interpretation is faithful to the source.

The model receives explicitly labeled body-text excerpts. `read_article` retains the complete retrieval receipt and raw XML cache, but its model view omits XML and limits the abstract to 6,000 characters and body text to at most eight windows totaling approximately 12,000 characters. Windows prioritize compound names, morphology, and dose/duration passages. Their start/end offsets refer to whitespace-normalized body text. Omissions and full-text availability are recorded separately. A fact's `access="full_text"` indicates that its evidence comes from a retrieved full-text source, not that the model read the entire paper or missed no context.

A full-text fact must cite the `fullTextXML` source that actually returned body text. It may also cite metadata for the same paper. Metadata alone, or a full-text source for a different paper, cannot establish full-text access for that fact. This check concerns paper identity and access receipts; it does not rewrite the model's conclusions.

In `tools.jsonl`, `result` saves the complete tool result and `model_result` saves the compact result shown to the model, with identical source records. ChEMBL retrieves at most ten activity records by default; its model view shows at most ten activities/mechanisms and four linked records per category, retaining exact-molecule/parent scope and truncation counts. PubChem annotations and paper searches also have size guards with omission notices. Raw API responses remain available in `cache/`. Source receipts establish access, not whether a particular morphological conclusion is supported by the original text.

## Deliverables for every morphological change

Each hypothesis has a unique ID and contains:

- The observable appearance and classification as direct literature, indirect literature, or MoA inference.
- Referenced facts and a research rationale; facts cite real tool sources or explicit sample-input MoA.
- Original experimental conditions, current applicability, differences between conditions, and uncertainty.
- The structure/channel and overview/local extent for the second agent to inspect, plus the treatment/control comparison.
- Visual support criteria, observations that do not support the hypothesis, and criteria for inability to assess because of image quality, missing channels, or inadequate conditions.
- Optional CP compartment, channel, and measurement-family query hints; retain a missing status when no matching records exist.

| Output field | Saved content |
|---|---|
| `facts[].origin / compound_scope / access` | Literature, database, input, or inference; exact compound, related compound, or target class; abstract, full text, annotation, or input |
| `hypotheses[].basis / fact_ids / rationale` | Hypothesis category, supporting facts, and research rationale |
| `applicable_conditions / uncertainty` | Current applicability and extrapolation limits |
| `where_to_check / verification.image_checks` | Inspection locations and specific image checks |
| `verification.comparison / supporting_observation` | Treatment/control comparison and observations that support the hypothesis |
| `disconfirming_observation / verification.not_assessable_when` | Observations that do not support it and conditions preventing assessment |
| `verification.cp_checks` | Optional CP compartment, channel, measurement family, and scope of interpretation |

Top-level `identity_context`, `literature_review`, and `mechanism_review` store the results of the first three steps. Step four is represented by each hypothesis's `verification`. Every supplied MoA is retained as a fact with `origin="provided_moa"` and `access="input"`, even when no morphological hypotheses are generated. It must not be represented as retrieved evidence.

See the [synthetic structure example with two basis categories](../examples/research_v2_synthetic_example.json). It illustrates fields and handoff only, corresponds to no real molecule or paper, and is not a run directory with complete receipts. It cannot be used as input to `describe --research`.

Full provenance and rationales remain in the research record. The observation-agent view includes morphological checks and basis-category labels, but no drug name, specific MoA, mechanistic explanation, facts, or source references. Basis labels indicate uncertainty and cannot count as image evidence. The observation agent may still reject hypotheses, preserve conflicts between fields of view, and record unexpected morphology.

## Development and compatibility

New research runs use the v2 research contract and versioned prompts. Older v1 results remain readable and replayable; they are not implicitly researched again or upgraded to v2 evidence. Use a new output directory for a new prompt version and do not overwrite old runs. Inputs, outputs, and the two stages remain isolated. The default offline mock produces no hypotheses and is explicitly labeled synthetic. Nonempty hypotheses used in tests are also manually constructed examples, not evidence about real compounds or papers.

See the [README](../README.md) for running and budgeting. The framework defaults to offline mock mode. Real-compound research results still require human review of retrieval quality, inference, and whether verification plans are practical. The observation stage can read verification plans, but the pilot below does not start the observation agent.

## Three-compound pilot of the first agent

The independent `python -m realclip_d.research_pilot` utility selects Nocodazole, Staurosporine, and Rapamycin to examine direct literature evidence and MoA inference under different mechanism priors. `prepare --run-dir <run-directory>` calls only the free public PubChem identity interface, saves identity receipts, and freezes inputs and the manifest. New inputs set cell line to U2OS, leave dose and treatment duration null, and omit context notes. The research stage receives no images or CP. Editable inputs are saved under `inputs/research_pilot_u2os/`; existing run records retain their original inputs.

`run --run-dir <run-directory> --credential-file <PROJECT_ROOT>/.env` performs the live research explicitly authorized for this pilot. The utility reads `OPENAI_API_KEY` from the specified file only when `run` or `complete` is executed explicitly. It does not execute the file or import other environment keys, and it clears custom `OPENAI_BASE_URL`. Ordinary `realclip-d` commands still do not load `.env` automatically.

The framework's observation stage does not participate in this pilot. The three samples run sequentially and share a **$5** ledger. Each sample permits at most 16 model requests, including one explicitly added synthesis request if necessary, for at most 48 requests in total. Each request allows at most 12,000 output tokens, with a conservative input bound of 250,000. The standard service tier is used. Reservations use $0.25 per million input tokens, $0.75 per million output tokens, and $0.01 per web search. At most three web searches per request yield a maximum planned reservation of **$4.872** over 48 requests; continuation does not create another allowance. The distinction between estimates and billing, together with price sources, is saved in the manifest. Paid requests are not automatically retried. Unknown usage stops subsequent paid requests, and resumption does not automatically resubmit incomplete stages.

Explicitly run this command to complete existing research records:

```bash
python -m realclip_d.research_pilot complete --run-dir <run-directory> --credential-file <PROJECT_ROOT>/.env
```

It first revalidates original candidates locally with no model calls. Passing candidates become results without textual changes. If synthesis is necessary, it uses the same research-agent prompt and previously saved, model-visible evidence for at most one new request. Tools and web search are removed; retrieval is not repeated, and full body text that was not previously shown is not introduced. The evidence packet retains receipt indexes, exact query parameters, real sources, and previously shown article excerpts. Search hits remain discovery records. Continuation refuses unknown usage or an existing continuation record, retains frozen budget and request limits, and preserves original failures, candidates, and requests.

In the first pilot records, Nocodazole and Rapamycin stopped at the local input guard before synthesis; the next request had not been sent. Staurosporine's original candidate encountered an overly strict same-paper metadata/XML citation check. Correcting that rule allows local revalidation without changing research text or automatically creating another paid request. The run report records final completion status.

`python -m realclip_d.pilot_report --run-dir <run-directory>` only reads saved records and generates `index.html`, `REPORT.md`, and `summary.json`. The report presents queries by model turn, evidence-access levels, sources, and original English morphological hypotheses with their verification methods. Incomplete runs, failures, and candidates that did not pass reference checks are explicitly labeled. Completion and usage are established by that report, `status.json`, and `budget.json`, not inferred from passing framework tests.
