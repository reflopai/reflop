#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# The live suite: reFLOP end-to-end tests and benchmarks against a running reflopd. Writes machine-readable
# results to benchmark/results/<datetime>/ (never committed: they hold host names and addresses).
#   ./benchmark/run_full.sh                         every suite in live/suites (~70 min)
#   ./benchmark/run_full.sh b01 b02                 only suites whose names start with these
#   E2E_QUICK=1 ./benchmark/run_full.sh             tiny matrices (harness smoke test)
#   E2E_OWNER_SSH=<host> ./benchmark/run_full.sh    also drive the Owner over ssh (Owner suites skip otherwise)
#   E2E_LOCAL=1 E2E_OWNER_HOME=<dir> ...            the Owner is a throwaway reflopd on this machine
#   E2E_SOAK_MINUTES=60 ./benchmark/run_full.sh b04
# Set up first with ./benchmark/setup.sh. See benchmark/README.md.
#
# results/<datetime>/
#   run.json          versions (client, reflopd, torch), settings, per-suite exit code / seconds / case counts
#   <suite>.json      cases (id, status, detail, seconds, data) and metrics (benchmark rows) of each suite
#   logs/             console output of each suite, every Owner command, Owner state saved for cleanup
set -uo pipefail
cd "$(dirname "$0")"
ROOT=$(pwd)
PY="${PYTHON:-$ROOT/.venv/bin/python}"
[ -x "$PY" ] || { echo "no $PY: run ./benchmark/setup.sh first"; exit 1; }

export E2E_RUN_ID="${E2E_RUN_ID:-$(date +%Y-%m-%d_%H-%M-%S)}"
OUT="$ROOT/results/$E2E_RUN_ID"
[ -e "$OUT" ] && { echo "$OUT already exists"; exit 1; }
mkdir -p "$OUT/logs"
E2E_SCRATCH=$(mktemp -d "${TMPDIR:-/tmp}/reflop-e2e.XXXXXX")
export E2E_SCRATCH PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" PYTHONUNBUFFERED=1
# Suites print through tee, so they cannot see the terminal themselves: color if ours is one (logs stay plain).
if [ -z "${E2E_COLOR:-}" ]; then if [ -t 1 ] && [ -z "${NO_COLOR:-}" ]; then E2E_COLOR=1; else E2E_COLOR=0; fi; fi
export E2E_COLOR

# The C API suite (a10) builds libreflop from the same source as the client under test: this checkout when the
# client is installed from it (setup.sh), else the release's tag, extracted once to cache/src-v<version>.
ver=$("$PY" -c 'import reflop; print(reflop.__version__)')
client=$("$PY" -c 'import reflop, os; print(os.path.realpath(reflop.__file__))')
REPO=$(cd .. && pwd)
case "$client" in
  "$REPO"/client/python/*) echo "client: this checkout (reflop $ver)" ;;
  *)
    src="$ROOT/cache/src-v$ver"
    if [ ! -d "$src" ]; then
      git -C "$REPO" rev-parse -q --verify "refs/tags/v$ver" >/dev/null || git -C "$REPO" fetch --quiet origin tag "v$ver" ||
        { echo "no tag v$ver for the C API suite's source"; rm -rf "$OUT" "$E2E_SCRATCH"; exit 1; }
      mkdir -p "$src.part" && git -C "$REPO" archive "v$ver" | tar -x -C "$src.part" && mv "$src.part" "$src"
    fi
    export E2E_SRC="$src"
    echo "client: release $ver from $client; C API source v$ver"
    ;;
esac

ALL=()
for f in e2e/suites/[a-z][0-9]*_*.py; do ALL+=("$(basename "$f" .py)"); done
sel=()
if [ $# -eq 0 ]; then sel=("${ALL[@]}"); else
  for want in "$@"; do for s in "${ALL[@]}"; do [[ $s == $want* ]] && sel+=("$s"); done; done
fi
[ ${#sel[@]} -gt 0 ] || { echo "no suite matches: $*"; rm -rf "$OUT" "$E2E_SCRATCH"; exit 2; }

state=interrupted
finish() {
  echo "######## owner cleanup ($(date +%H:%M:%S))"
  "$PY" -m live.owner cleanup >> "$OUT/logs/owner_cleanup.log" 2>&1
  "$PY" -m live.manifest finish "$state" > /dev/null 2>&1
  rm -rf "$E2E_SCRATCH"
}
trap finish EXIT
trap 'exit 130' INT TERM

echo "run $E2E_RUN_ID: reflop $ver, ${#sel[@]} suites -> $OUT"
echo "Owner: $("$PY" -c 'from live import owner; print(owner.where() if owner.available() else "none (Owner suites skip)")')"
"$PY" -m live.manifest init "${sel[@]}" || { state=failed; exit 1; }
"$PY" -m live.owner save-state > /dev/null 2>&1
# Warn (do not block) when something other than reflop holds GPU memory on the Owner: benchmarks will be skewed and
# large allocations can fail (or push that job out of memory).
"$PY" - "$OUT/run.json" <<'EOF'
import json, sys
g = json.load(open(sys.argv[1])).get("owner_gpu_start") or {}
busy = [p for p in g.get("processes", []) if not p["reflop_worker"]]
if busy:
    print("\033[33mWARNING: the Owner's GPU is in use by non-reflop processes (results will be skewed):\033[0m")
    for p in busy:
        print(f"  pid {p['pid']} {p['user']} {p['used_memory_mib']} MiB: {p['cmd'][:120]}")
EOF

t_all=$(date +%s)
for s in "${sel[@]}"; do
  echo
  echo "######## $s  ($(date +%H:%M:%S))"
  t=$(date +%s)
  "$PY" -u -m "live.suites.$s" 2>&1 | tee >(sed -l $'s/\x1b\\[[0-9;]*m//g' > "$OUT/logs/$s.log")
  rc=${PIPESTATUS[0]}
  echo "######## $s finished: exit $rc in $(( $(date +%s) - t )) s"
  "$PY" -m live.manifest suite "$s" "$rc" "$(( $(date +%s) - t ))"
done
state=completed
echo
echo "######## done in $(( ($(date +%s) - t_all) / 60 )) min: $OUT"
