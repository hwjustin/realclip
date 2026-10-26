# Observation Agent implementation

Observation produces a grounded English description of the supplied treatment
and control wells. It first observes images without research guidance, then
uses guidance and available CellProfiler measurements for targeted checks,
writes a natural summary from evidence, and reviews that summary against images.
The detailed observation record and final prose are separate products.
Research hypotheses guide attention; unexpected findings are equally eligible.
Research Agent prompts, research results and their bindings are unchanged.

## Four separate model phases

Each phase is a separate model call sequence with a restricted input. The
separation is enforced by constructing different requests, not by asking one
conversation to ignore hypotheses it already saw.

| Phase | Inputs and tools | Output |
|---|---|---|
| Baseline | All original site/channel images, anonymous experimental context and image metadata; `inspect_image`; no hypotheses or CP values | `BaselineOutput`: observations, exact original-image coverage, limitations |
| Guided | All original images, unchanged baseline observations, appearance guidance and CP catalogue; `inspect_image`, `query_cp` | `GuidanceEvidenceOutput`: retained and added observations, hypothesis assessments, limitations |
| Writer | Observation evidence only; no images, research hypotheses or tools | `DescriptionDraft`: ordered natural sentences with observation references |
| Reviewer | All original images, generated crops, observation evidence and draft; `inspect_image`; no hypotheses, compound identity or CP query | `DescriptionReview`: keep, revise or delete each draft sentence |

The prompts are `observation_baseline_v1.md`, `observation_guided_v1.md`,
`observation_writer_v1.md` and `observation_reviewer_v1.md` under
`src/realclip_d/prompts`. Output contracts are supplied as strict JSON schemas.
The runtime shares the configured Observation turn and spending limits across
all four phases. At least four model turns are needed; the default is eight.
When the shared turn budget leaves room, a phase reserves at most one additional
model call to correct a typed-output validation failure. The SDK retains the same
strict schema. Correction receives the failed phase history and precise validation
errors, with tools disabled; it uses the same recorded model and paid-budget ledger.
The original output, errors, correction status and changed JSON paths are saved
under `phases/<phase>/corrections/`. A failed correction stops the run; transport
errors, refusals, truncated JSON and unknown usage are not automatically retried.
All ordinary and correction calls count toward the configured global turn limit,
with at least one call kept for each later phase. Full phase usage is assembled
from recorded calls, including nested correction calls. Changed paths are an audit
record, not a guarantee that only the incorrect field changed; final provenance,
baseline-prefix and sentence checks still run.
Separate phases do not imply four independent biological reviewers or an
empirically validated improvement in description quality.

Baseline reviews the full images before using crops: coverage and visible
abundance, cell outlines and spreading, DNA staining, channel-specific patterns,
site variation and image limitations. Near-empty fields remain in the review.
It can record common appearances without a clear treatment-specific change,
minority populations and unexpected findings. `reviewed_image_ids` contains
exactly the supplied original IDs; this is a declared coverage check, not proof
that every image was interpreted correctly.

Guided preserves every baseline observation exactly, including IDs, wording,
comparison kind, references and CP assessment. It adds new IDs for qualifications, corrected
interpretations, conflicts or integrated CP evidence. Retained history is not an
endorsement of every baseline claim. The writer receives this evidence and must
preserve the better-supported account and unresolved conflicts. All guidance
hypotheses receive one assessment; an empty hypothesis list is a normal input.

Writer can merge, paraphrase and order findings instead of concatenating selected
observation texts. It leads with informative broad appearances, preserves subset
and local scope, and avoids a hypothesis checklist or a long list of negatives.
Every sentence references existing visually confirmed observations. Useful
single-role appearances may enter the description when role and scope remain
explicit; they do not establish a treatment-control difference. New visual
findings, mechanisms, frequencies, causal conclusions and invented limitations
are forbidden. No trustworthy visual evidence means an empty description with
an explanation, not a default claim that treatment was unchanged.

Reviewer sees images again and checks every draft sentence once. `keep` retains
text and observation references exactly; `revise` uses existing visually
confirmed evidence; `delete` has empty text and references. It cannot add a new
finding discovered during review. Such a gap remains explicit for a future
observation pass. The application assembles kept/revised sentences in draft
order and saves their evidence links.

## Local image inspection

`inspect_image(image_id, x, y, width, height, question, selection_reason)` returns
an actual local image plus its ID and provenance. A text receipt alone is not a
substitute for seeing the pixels. Coordinates are integers in the supplied
source image's pixels; invalid or out-of-bounds rectangles are rejected. Crops
retain source pixels without resizing or artificial resolution enhancement.
Enlarging their display cannot recover unavailable optical detail.

The agent requests a crop to answer a concrete question, such as whether a
bright compact structure has one discernible outline or several adjacent ones.
Representative regions, controls and counterexamples matter alongside unusual
structures. A crop can clarify local appearance; full-field views establish
coverage and the prevalence of a pattern. The tool records the question,
selection reason, source image, bounding box, actual crop and hashes.

A parent and crop are related evidence, not independent sites or replicate
wells. Multiple sites of one well are not replicate wells. Treatment/control
sites with the same index do not pair individual cells. Use explicit alignment
metadata and visible correspondence before claiming the same cell across
channels; common field labels alone do not establish registration. A channel
may contain several stained structures, so stain identity does not automatically
establish a particular organelle or object identity.

## Input and research handoff

Supply the saved research context and all intended treatment/control well sites
and actual available channels. Site count is taken from input metadata rather
than assumed. Optional image fields are `well_id`, `parent_image_id`, `crop_box`,
`alignment_id`, `stain`, `pixel_size_um`, `acquisition_settings` and
`display_settings`. Dimensions and hashes come from the files. Private filenames
and paths do not enter model requests. Unknown metadata remains null.
A supplied `crop_box` is `[x, y, width, height]` in its supplied parent's pixels;
it is not automatically a box in an unprovided higher-resolution source.

The guidance handoff exposes appearance and comparison fields without compound
or mechanism identity. Literal private identifiers and selected research terms
are checked in views, narrative and CP metadata. These conservative guards do
not prove semantic anonymity. The baseline never receives the guidance; the
writer and reviewer receive observation evidence without hypothesis records.

When appearance handoff text needs an explicit revision, use `--handoff
revision.json`. Its `schema_version`, `sample_id`, `research_sha256` and `edits`
bind each original/replacement/reason to a specific hypothesis and field.
The hash is `storage.digest(research.model_dump(mode="json"))`, not the raw
file checksum. Stale hashes, wrong originals and duplicate edits fail. Original
research is retained; `handoff.json` records the observation-facing revision.
Fidelity of a handoff revision still requires scientific review.

## CellProfiler evidence

CP is optional. Baseline knows only whether CP exists: its observations use
`not_checked` when available, or `not_available` otherwise, with no CP references.
Only the Guided phase receives the CP catalogue and query tool. It can add a
new visual observation integrating returned measurements without rewriting the
baseline record. Writer/reviewer see the resulting evidence, not fresh CP tools.

Set `cp_preprocessing` to the actual upstream description when known. The tool
applies no additional transform. The catalogue lists groups, roles, scopes and
units without exposing feature IDs or values. `query_cp` pages whole feature
comparisons, keeping counterpart roles together or explicitly identifying a
missing role. `offset`, `limit` and `total` count comparisons; `total_rows` counts
measurements. `rows` preserves source values and IDs. Only IDs actually returned
to the agent may be cited. Use `query` to discover names and `feature_names` for
exact selection. Duplicate role/feature/scope identities are rejected.
Use compartment/channel/family combinations present together in the catalogue,
and correct rejected filters using the returned available-group suggestions.
An actual `not_found` response with `total=0` means the supplied query matched no
measurement. It does not mean the compound has no CP data, has zero cells or
shows no visual difference. An empty page when `total>0` is a pagination issue.
Object-number identifiers and neighbor features cannot substitute for a total
cell/nucleus count merely because their names contain Number or Count.

Optional `cp_metadata` supplies `value_kind`, `normalization_reference`,
`comparison_context`, `aggregation_method`, `within_feature_order_preserved`,
`quality_notes` and feature definitions. Each definition names an exact feature
and its meaning, with optional scale, raw unit and higher-value meaning. Feature
semantics are not guessed from names. Unknown interpretation facts remain null.

Each returned comparison reports feature IDs, supplied definition, literal
`numeric_order`, `comparison_ready` and `interpretation_gaps`. Readiness requires
finite counterparts, matching known units and scope, shared comparison context,
aggregation and definitions. Preprocessed values also require a normalization
reference and confirmed order preservation. This permits an interpreted
same-feature comparison, not statistical significance, raw effect size or a
single-cell distribution. Well-level CP does not confirm a particular cell or
minority population. Mixed granularity directions across scales do not by
themselves prove unchanged texture.

Each observation has `id`, `text`, `scope`, required `comparison_kind`,
`image_ids`, `cp_feature_ids`, `visually_confirmed` and `cp_assessment`.
`comparison_kind` is `treatment_control`, `treatment_only`, `control_only` or
`measurement_only`. Contrasts and shared appearances asserted in both roles use
`treatment_control` with both roles in a common channel. A neutral appearance
bounded to treatment or control uses the corresponding single-role kind and
cites images only from that role. It may be a valid local or subset finding
without a counterpart image, but cannot establish enrichment, a treatment effect
or absence in the other role. Nonvisual CP-only observations use
`comparison_kind=measurement_only`. This explicit distinction corrects the
structural rejection of valid single-role records by a blanket paired-image
requirement; it does not establish biological accuracy or description quality.
CP assessment has `status`, `reason` and
`effect_on_description`. `supports`, `contradicts` and `mixed` require complete
comparison-ready exposed pairs. Unresolved interpretation metadata requires
`not_informative`; this preserves values while limiting interpretation.
`not_informative` still requires actual returned CP IDs. A visual observation may
instead use `no_matching_measurement` with no CP IDs when a relevant query
returned `not_found`, `total=0`, `rows=[]` and `comparisons=[]`. Its reason states
the measurement gap; its appearance claim must remain grounded in images.
Validation requires an application-written `query_cp` receipt from this run and
reproduces its arguments against the bound CP table to confirm zero matches.
Errors, absent CP and empty pages with remaining matches cannot authorize this
state. Receipt checks establish that an empty query actually happened; semantic
relevance to each observation still requires review. This state cannot be used
by measurement-only observations. Baseline still uses exactly `not_checked` or
`not_available`, without implying that a CP query was performed.
`not_checked` and `not_available` also have no CP IDs. Nonvisual measurements use
`measurement_only` and cannot enter the visual description. Relevant CP belongs
on the same new visual observation, with its actual effect on wording or scope;
unrelated measurements must not be attached to fill a field.
Writer and Reviewer keep internal query states and tool errors out of appearance
sentences; a material lack of quantitative measurement belongs in limitations.

## Assessments and validation

Guided hypothesis checks separately record treatment, control and difference
(`observed`, `not_observed` or `unassessable`). Support for a predicted change
requires an observed contrast, not presence alone. A supported or not_supported
visual check must cite at least one relevant valid `treatment_control` visual
observation. Treatment-only/control-only appearances or CP alone cannot establish
the predicted contrast. An unavailable comparison is
uncertain. Check statuses are supported/not_supported/uncertain; the overall
assessment is supported if all checks support, not_supported if all reject,
partially_supported if supporting and other checks coexist, and uncertain
otherwise. Every hypothesis is assessed once; evidence can be reused without
being counted as independent. Guidance assessments do not determine which
findings are eligible for description.

Validation checks sample identity, exact baseline retention, coverage declarations,
unique IDs, exposed image/CP references, appropriate CP statuses, hypothesis
coverage, sentence-to-observation links and review completeness. A
`treatment_control` observation needs both roles in a common channel, while a
single-role observation needs images only from its declared role. Writer and
reviewer preserve this distinction instead of manufacturing a comparison from
one-sided evidence. A final visual
sentence cannot rely only on measurements. Structural checks do not establish
that a claimed image difference, a paraphrase or a scientific inference is true.

## Saved artifacts and completion

Each `observation/phases/{baseline,guided,writer,reviewer}/` directory saves
`instructions.md`, `output_schema.json`, `request_input.json`, model request and
response receipts, call/usage records and `output.json`. Root observation
artifacts include `candidate_output.json`, `result.json`,
`description_evidence.json`, `pipeline.json`, `image_exposure.json`, `tools.jsonl`
and `crops/manifest.json`. Generated crops and provenance are retained locally.
The evidence export links final sentences to observations and their original or
crop images, CP relationship and scope. Completion seals bind saved artifacts;
use a new output directory for each run. Validation failure preserves candidate
and transport evidence and does not seal a successful stage.

Live model requests use the provider's input-token count endpoint on the actual
converted request before reserving generation cost. Counts include images,
tool/output schemas and conversation history, with a 16,384-token margin.
Base64 size is not treated as vision-token count. A failed or oversized count
sends no generation. No conversation truncation is enabled. The four phases
share the Observation turn budget and paid-budget ledger; local crops are tools,
not additional paid model calls by themselves. A new live run still requires
explicit authorization within its configured budget.

## Offline checks

Run inside `D`. These commands use local scripted models, block network access
and do not load credentials or offer a live mode:

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m realclip_d.observation_pipeline_offline \
  --prepared runs/observation_pilot_inputs_20261021 \
  --output runs/observation_pipeline_offline_20261021
```

The dedicated verifier exercises all four phases, original-image coverage,
actual crop tool transport/provenance, phase input separation, CP exposure,
sentence/review contracts and saved stages against the prepared well pairs.
It does not invent normalization metadata or biological image interpretation.
Its scripted descriptions remain empty and hypotheses uncertain. Successful
coverage and crop checks validate software behavior, not biological accuracy.
The report identifies the use of a local scripted model and zero paid cost.

The existing prepared-pair CP verifier also invokes the current pipeline:

```bash
.venv/bin/python -m realclip_d.observation_cp_offline \
  --prepared runs/observation_pilot_inputs_20261021 \
  --output runs/observation_cp_pipeline_offline_20261021
```

The default CLI remains mock. Mock execution cannot stand in for a real-image
biological evaluation, and paid Observation cannot consume synthetic research.
A future authorized evaluation should separately assess completeness, factual
support, scope preservation and natural prose, with hidden run identity and
balanced presentation order when comparing versions.
