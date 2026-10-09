#!/usr/bin/env bash
set -euo pipefail

cat >&2 <<'MSG'
This release does not download or host research datasets.
Obtain each dataset from its upstream source and follow its license and access terms.
Arrange datasets in the layouts expected by the configs, then set:
  export UNID_DATA_ROOT=/path/to/local/data
See README.md for the evaluation commands and expected dataset families.
MSG
exit 2
