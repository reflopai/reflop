#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# One-time setup of benchmark/.venv for the live suite (and again after changing what it tests).
#   ./benchmark/setup.sh                    the client from this checkout (editable): test before a release
#   ./benchmark/setup.sh --release 0.4.2    reflop==0.4.2 from PyPI: test what users install
# The C API suite (a10) builds libreflop from this checkout, or, for a release, from that tag's source, which
# run_full.sh extracts to benchmark/cache/src-v<version>.
set -euo pipefail
cd "$(dirname "$0")"
command -v uv >/dev/null || { echo "needs uv (https://docs.astral.sh/uv/)"; exit 1; }
release=""
case "${1:-}" in
  "") ;;
  --release) release="${2:?--release needs a version, e.g. 0.4.2}" ;;
  *) echo "usage: $0 [--release X.Y.Z]"; exit 2 ;;
esac
[ -x .venv/bin/python ] || uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
if [ -n "$release" ]; then
  uv pip install --python .venv/bin/python "reflop==$release"
else
  uv pip install --python .venv/bin/python -e ../client/python
fi
.venv/bin/python -c 'import reflop; print("reflop", reflop.__version__, "from", reflop.__file__)'
