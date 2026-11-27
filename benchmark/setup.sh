#!/usr/bin/env bash
# One-time setup (and after updating reflop): .venv from uv.lock, vendor/reflop at the client's release.
#   ./setup.sh
#   uv lock --upgrade-package reflop && ./setup.sh     # move to the newest reflop release first
set -euo pipefail
cd "$(dirname "$0")"
command -v uv >/dev/null || { echo "needs uv (https://docs.astral.sh/uv/)"; exit 1; }
uv sync --python 3.12
ver=$(.venv/bin/python -c 'import reflop; print(reflop.__version__)')

# vendor/reflop: source of libreflop for the C API suite (a10_capi), kept at the same tag as the client.
[ -d vendor/reflop/.git ] || { mkdir -p vendor; git clone https://github.com/reflopai/reflop.git vendor/reflop; }
git -C vendor/reflop fetch --tags --quiet origin
git -C vendor/reflop checkout --quiet "v$ver"
echo "reflop $ver, vendor/reflop at $(git -C vendor/reflop describe --tags --always)"
.venv/bin/reflop status
