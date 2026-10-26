# Findings as the research record

The current implementation is described in [the simple research agent](RESEARCH_AGENT_SIMPLE.md).

One researcher runs the original four-stage prompt through one `Runner.run`,
including normal tool calls. Once it returns, Python saves the second stage's
findings, reported conditions and research gaps directly to `findings.json`.
The full four-stage result is saved as usual. No extra model is called to save,
rewrite or summarize findings.

Independent notes, local memory-query tools, evidence-index management, reading
stop tools and context checkpoints have been removed from the active flow.
Article pagination remains part of `read_article`; it does not require a separate
reading-task queue. No pending-reading gate forces another research conversation.

Source records and raw retrievals are saved in backend artifacts. New model
outputs do not contain citation IDs or a bibliography, and citation metadata does
not block completion. Basic output structure and hypothesis/guidance associations
are still checked. This is not biological accuracy validation.

The historical memory implementation and failed pilot outputs remain in their
existing run directories for inspection. Current research does not import or
reuse them, and this change does not rerun them.
