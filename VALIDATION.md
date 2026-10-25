# Validation

Validated on the source machine on 2026-10-25 using Python 3.12 and the pinned
dependencies in `pyproject.toml`. The H200 host has not been benchmarked yet.
The complete package test suite passed: **59 tests**.

The direct-S3 path uses the same frozen dataset identity and descriptions as
the portable transfer package. All 25,900 raw-image entries were matched to
their original public URL, SHA256, and byte count through 735 source pairing
records; there were no missing entries or conflicting mappings. Original TIFF
bytes are downloaded without re-encoding. The raw payload totals
60,014,495,288 bytes. Download scripts do not require PyTorch or AWS credentials.

The repository includes a benchmark command for a small concurrent download.
Its network-byte counter excludes cached files, and its elapsed time includes
verification and disk writes. The rate measured on the target server remains
unknown until that command is run there. Source-machine upload measurements
and the GPU smoke test below are separate from that measurement.

Downloader regression tests use a local HTTP server, covering valid Range
resume, a server that ignores Range, incorrect Content-Range, interrupted
responses and retries, checksum failure without replacing existing data,
benchmark-to-full cache reuse, verification with no network traffic,
cross-process locking, stale-report replacement, failure exit status, and
path/URL validation. Metadata-export tests cover missing or conflicting
source mappings, unchanged captions and split identities, and deterministic
compressed artifacts. Loading the real 735-well metadata with system Python
was also checked without importing PyTorch. The README's Bash blocks passed
`bash -n` syntax checking.
Two actual public S3 TIFF downloads also passed their frozen byte counts and
SHA256 checksums on the source machine. They are a connectivity/integrity
smoke test, not a measurement of the H200 server's download performance.

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
