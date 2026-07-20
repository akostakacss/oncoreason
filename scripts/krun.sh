#!/usr/bin/env bash
# Canonical Kaggle run: push -> wait -> pull+archive. Archiving is built in, so every
# run is durably recorded under runs/ (audit trail). Never push a kernel by hand for a
# real run — use this, so nothing goes unrecorded.
#
# Usage:  scripts/krun.sh <stage> <label>
#   <stage>  a directory under notebooks/, e.g. policy_sampling. Its kernel-metadata.json
#            supplies the kernel slug, so stages never overwrite each other.
#   <label>  short tag for this run, e.g. n8-t08.
set -euo pipefail

STAGE="${1:?usage: scripts/krun.sh <stage> <label>}"
LABEL="${2:?usage: scripts/krun.sh <stage> <label>}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

DIR="notebooks/$STAGE"
META="$DIR/kernel-metadata.json"
[ -f "$META" ] || { echo "[krun] no $META (stages: $(ls notebooks/ | tr '\n' ' '))"; exit 1; }
KERNEL=$(python3 -c "import json,sys; print(json.load(open('$META'))['id'])")

# Training stages run for hours; poll for up to 12h (Kaggle's own batch ceiling) rather
# than the 40min the smoke test needed. Interval backs off so long runs are cheap to watch.
MAX_WAIT_S=${MAX_WAIT_S:-43200}

echo "[krun] stage=$STAGE kernel=$KERNEL label=$LABEL"
kaggle kernels push -p "$DIR" | sed 's/^/[krun] /'

echo "[krun] waiting (up to $((MAX_WAIT_S / 3600))h)..."
start=$(date +%s); interval=20
while :; do
  s=$(kaggle kernels status "$KERNEL" 2>&1 | head -1)
  # Treat QUEUED as still-pending: the old check broke out of the loop on anything that
  # was not RUNNING, so a queued kernel was pulled before it had produced any output.
  if ! echo "$s" | grep -qE 'RUNNING|QUEUED'; then
    echo "[krun] $s"; break
  fi
  elapsed=$(( $(date +%s) - start ))
  if [ "$elapsed" -ge "$MAX_WAIT_S" ]; then
    echo "[krun] TIMEOUT after ${elapsed}s — kernel still $s; not pulling (output would be partial)"
    exit 1
  fi
  printf '[krun] %s  (%dm elapsed)\n' "$s" "$((elapsed / 60))"
  sleep "$interval"
  [ "$interval" -lt 120 ] && interval=$((interval + 10))
done

echo "[krun] pulling + archiving output..."
python3 scripts/pull_run.py "$STAGE" "$LABEL"
echo "[krun] done. See runs/INDEX.tsv"
