#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
PYTHON_BIN="${PYTHON_BIN:-python}"
output_dir="${1:-runs/mock-smoke}"
"$PYTHON_BIN" -m realclip_d validate --input examples/sample.json
"$PYTHON_BIN" -m realclip_d run --input examples/sample.json --output "$output_dir" --mode mock
