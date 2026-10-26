# Concurrent compound research

`realclip-d research-batch` runs complete research agents concurrently across
compounds. Each worker has its own thread, event loop, model client, conversation,
tool receipts, cache and output directory. Each compound still uses the approved
four-stage research prompt and exactly one `Runner.run`, including its normal
model/tool loop. Findings are saved by the existing research workflow. This command
never runs the observation agent.

The default is ten workers; `--workers` accepts 1 through 32. The default research
profile matches the last successful three-compound run: `gpt-6-luna`, high
reasoning, up to 50 model turns, 100 function calls, 12,000 output tokens, and an
explicit conservative input reservation of 1,000,000. `--config` overrides that
profile with the supplied settings.

## Commands

From `<PROJECT_ROOT>`, an offline example using the saved pilot inputs:

```bash
D/.venv/bin/realclip-d research-batch \
  --inputs D/runs/research_pilot_3_simple_20261020_233540/inputs/*.json \
  --output D/runs/research_batch_mock \
  --workers 3
```

For a real batch, provide input files, choose `live` explicitly, and set one total
budget for the entire batch. For example, after placing the desired research
inputs in `D/batch_inputs/`:

```bash
D/.venv/bin/realclip-d research-batch \
  --inputs D/batch_inputs/*.json \
  --output D/runs/research_batch_live \
  --workers 3 \
  --mode live \
  --budget-usd 100 \
  --credential-file <PROJECT_ROOT>/.env
```

These commands are examples; creating this feature did not execute a paid batch.
Every input is validated before workers start. Sample IDs must be unique and
contain only letters, digits, underscores, dots or hyphens, starting with a letter
or digit. Inputs contain compound information, known MoA and experimental context;
they must not supply images or CP files. The optional credential file is read once
before workers start, loading only the authorized `OPENAI_API_KEY` entry.

## Shared limits and outputs

All PubChem clients within one batch share a thread-safe limiter, defaulting to
two requests per second across all workers. Each worker retains its own HTTP
client and cache. Existing retrieval backoff behavior remains. Separate batch
processes do not share this in-memory limiter; use one batch command to manage
compound concurrency.

All workers share one budget ledger. Reservation, completion and unknown-usage
writes are serialized so simultaneous requests cannot reserve the same remaining
budget. Charges remain conservative estimates using the configured rates.
No automatic paid retry is enabled. Unknown usage prevents further paid requests;
requests already in flight may finish and their returned usage is still recorded.

`status.json` updates as compounds finish, with per-compound status and duration.
Each compound saves its usual artifacts under `samples/<sample_id>/research/`,
including `result.json`, `findings.json`, sources and model/tool records. A normal
compound failure does not discard other compounds' results. The command exits
with status 2 if any compound fails or is blocked.

Rerunning the same inputs/settings into the same output directory reuses verified
completed research without another model call; worker count may be changed.
Incomplete paid stages retain their attempts and are not automatically resubmitted.

## Expected speed

The recent serial research run took approximately 221 seconds for three compounds,
with individual durations of 69–78 seconds. Three workers can overlap those waits.
The ideal lower bound for that workload is approximately the slowest single
compound, rather than the sum of all three. This has not yet been measured in a
live parallel run. Shared database limits, model API quotas and differing amounts
of literature work can reduce the speedup. Parallelism does not divide total
model usage or cost.

## Ten-worker live test

The first live ten-worker test completed all ten new compounds in 139.9 seconds
of research runtime. Individual research tasks took 98.0–139.9 seconds. Recorded
task intervals and model-request intervals both reached a concurrency of ten.
No request had unknown usage, and the observation agent was not run.

The compounds were paclitaxel, cytochalasin D, doxorubicin, etoposide, bortezomib,
brefeldin A, thapsigargin, chloroquine, rotenone and trichostatin A. All used
U2OS with unspecified dose and duration and the unchanged four-stage prompt.
The run is saved under `runs/research_batch_10_20261021_000130/`; its `index.html`
contains complete English outputs and links to original records.

Returned token usage, including cache reads/writes and hosted web tool calls,
gives a standard-price estimate of approximately $0.196 for this run; this is
not reconciled billing. The budget ledger's intentionally conservative estimate
is approximately $1.802. These new compounds were not also run serially, so the
test establishes live concurrency and throughput rather than a matched speedup.

## Sixteen-worker resource benchmark

The next live run selected 16 new compounds directly from the current dataset's
1,847 treatment compounds, retaining dataset IDs, original MoA labels and exact
InChIKeys in `selection.json`. The earlier ten were mechanism examples and did not
match this main dataset. Inputs again used U2OS with unspecified dose and duration.
The research prompt was unchanged; the observation agent was not run.

All 16 completed in 138.4 seconds, with 16 simultaneous research tasks and model
requests. The dedicated research process averaged 0.63% of the machine's 12
logical CPUs and peaked at 47.64% in half-second samples. Maximum observed RSS
was approximately 225 MiB. Minimum available WSL memory was 10.98 GiB, and no
swap-out pages occurred during the run. WSL metrics do not measure all Windows
host applications.

Database telemetry observed 573 requests: no 429 or 503, three HTTP 500 responses,
and five PubChem 404 responses. All model attempts completed with known usage;
automatic paid retries remained disabled. The run's monitor snapshot covered
`httpx` database clients, while the installed OpenAI SDK used `httpx2`, so OpenAI
rate-limit response headers were not captured. Account RPM/TPM allowances remain
unknown. `RunMonitor` now supports both transports, verified offline with the
installed SDK and mocked HTTP; the paid run was not repeated.

The hardware measurements support trying 32 workers, but API capacity and speed
at 32 had not been tested in that benchmark. The CLI now permits 32 workers.
This differs in compound
selection from the ten-worker run and is not a matched speedup comparison.

Artifacts are under `runs/research_batch_16_20261021_003900/`: `index.html` has
complete English outputs; `execution_summary.json` combines results, usage and
measurement coverage; `telemetry/` contains resource samples, HTTP events and
the resource summary. Returned-usage standard-price cost is approximately
$0.269, distinct from the conservative budget reservation ledger's $2.402.

For future dedicated batch scripts, use `with RunMonitor(output / "telemetry"):`
around `run_batch(...)`. The monitor samples every half second and wraps HTTP
send methods without changing retry behavior. It records response limit headers
and request host/path, excluding query strings, request bodies and request
headers. Use a fresh telemetry directory for each run; do not run multiple
monitors in the same process.

## Detached full-dataset campaign

The one-shot campaign under `runs/research_full_32_20261021_011000/` uses 32
research workers with the approved prompt and updated retrieval tools. Its
`campaign_plan.json` lists scheduled compounds, reused sealed live results and
any incomplete paid attempts held for review. Deduplication uses the full
InChIKey and U2OS/dose/duration context; previous results retain their original
files. New inputs contain no image or CP fields.

`job.py launch` starts a new OS session with stdin disconnected and stdout/stderr
in `run.log`. It refuses to launch a duplicate. The source snapshot is frozen,
so later workspace edits do not change a running campaign. This is a local
process, not a recurring LLM automation. The machine and WSL must remain running;
there is no automatic restart or paid retry.

`progress.json`, `costs.json`, and `rate_limits.json` update every 30 seconds.
Resource samples and raw HTTP status/response-limit headers are in `telemetry/`.
The credential loader reads only `OPENAI_API_KEY` from the explicitly supplied
project `.env`; request headers and bodies are excluded from HTTP telemetry.
On completion, `result_index.json` includes both new and reused result paths,
and `failed_samples.json` lists unfinished work. Costs based on returned usage
are separate from the conservative budget ledger. The shared budget retains
the existing unknown-usage stop: a possibly charged failed request is preserved
and prevents new paid calls, rather than being automatically resubmitted.

The continuation campaign `research_full_32_resume_20261021_012100/` uses
`isolate_failures=True`: each reservation has a compound owner, and a provider
rejection or unknown-usage error stops that compound only. The rejected request
is never rephrased/retried, and its full conservative reservation stays counted
against the shared cap while unrelated compounds proceed. A reservation overrun
still blocks the whole pool. The original CLI/default global-stop behavior is
unchanged. Prior spent/uncertain charges are deducted from the continuation's
available cap; prior incomplete stages are held for review. The restart does
not change the scientific prompt or agent workflow.
