#!/usr/bin/env bash
# Canonical Kaggle run: push -> wait -> pull+save. Saving is built in, so every
# run is durably archived in runs/ (audit trail). Never push the kernel by hand
# for a real run — use this, so nothing goes unrecorded.
#
# Usage:  scripts/krun.sh <label>
#   <label> is a short tag for the run, e.g. 3b-baseline, prm-v1.
set -euo pipefail

LABEL="${1:?usage: scripts/krun.sh <label>}"
KERNEL="ta1010/gate1-smoke-test"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

echo "[krun] pushing $KERNEL ($LABEL)..."
kaggle kernels push -p notebooks | sed 's/^/[krun] /'

echo "[krun] waiting for completion..."
for i in $(seq 1 120); do
  s=$(kaggle kernels status "$KERNEL" 2>&1 | head -1)
  echo "$s" | grep -q RUNNING || { echo "[krun] $s"; break; }
  sleep 20
done

echo "[krun] pulling + archiving output..."
python3 scripts/pull_run.py "$LABEL"
echo "[krun] done. See runs/INDEX.tsv"
