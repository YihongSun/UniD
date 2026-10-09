#!/usr/bin/env bash
set -euo pipefail

cat >&2 <<'MSG'
This release does not download UniD training or evaluation checkpoints.
Place the base model, specialist checkpoints, and any unified checkpoint at the local paths referenced by your config, or update those configs.
No private storage access is required or attempted.
MSG
exit 2
