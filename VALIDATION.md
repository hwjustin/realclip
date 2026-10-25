# Validation

Validated on the source machine on 2026-10-25 using Python 3.12 and the pinned
dependencies in `pyproject.toml`. The H200 host has not been benchmarked yet.

The test suite covers complete encoder unfreezing and optimizer coverage,
gradients through every view of variable-site wells, equivalence to standard
bidirectional InfoNCE when captions differ, deterministic interrupted training
and resume, checkpoint compatibility, held-out split isolation, portable
manifest identities, image and caption checksums, path validation, archive
extraction, hardlink staging, and code-only publishing.

The deterministic image preprocessing was also compared with the original
implementation on eight synthetic uint16 cases, including constant channels
and rectangular full fields; output tensors matched exactly.

A real GPU smoke test used the existing local pretrained weights and two real
training wells containing six and nine sites respectively: all 75 uint16 TIFF
images, five channels per site, and their original descriptions. It ran full
518-pixel preprocessing, BF16 forward, contrastive loss, backward, and one
AdamW update with an image microbatch of one on an RTX 4070 Ti SUPER.

- Complete DINOv2-small trainable parameters: 22,056,576.
- Complete BiomedCLIP text tower trainable parameters: 109,710,848.
- Both encoders received nonzero gradients and both changed weights.
- Observed single-step loss: 0.73073995.
- Observed peak allocated GPU memory: 2.577 GiB.
- Observed elapsed time: 9.31 seconds.

These are source-machine smoke-test measurements, not H200 throughput or
training-quality results. Full-epoch validation and held-out test performance
remain to be measured on the training machine.
