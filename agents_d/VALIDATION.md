# Portable release validation — 2026-10-26

- Clean wheel installation into a separate temporary Python 3.12 environment.
- Imports verified from the installed package, outside the source workspace.
- Full offline suite: **616 passed, 19 skipped**. Skips require historical run
  artifacts deliberately excluded from this portable release.
- Complete synthetic research + five-module Observation workflows passed for
  both the original profile and the local-model configuration in mock mode.
- Real Agents SDK plus mocked native HTTP transport checked structured output,
  function execution, initial images, crop-tool images, local counting before
  generation, matching count/generation messages and tools, chat-template
  options, usage normalization, credential isolation and no automatic retries.
- Server capability check tested with mocked local endpoints; its opt-in
  generation probe distinguishes the initial red image from a returned blue
  tool image using JSON schema output.
- Default scientific research prompt SHA-256:
  `2ee97f9ea07e849090b30fad70db9ac0a8b3ac2c4a59b821b843a5c463175fad`.
- No paid model generation, actual H200 inference, new biological validation,
  or CLIP training was performed for this export. Run the included server check
  against the selected model on the target H200 before real sample generation.

The original D source and all existing research/observation runs are preserved.
The new local-provider integration exists only in this portable copy.
