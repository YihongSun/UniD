#!/usr/bin/env bash
set -euo pipefail

python -m pip install -r requirements.txt

echo "Inference dependencies installed in: $(python -c 'import sys; print(sys.executable)')"
echo "Activate the CUDA-enabled environment used for this installation before running UniD."
