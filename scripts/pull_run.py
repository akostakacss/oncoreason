#!/usr/bin/env python3
"""Pull the latest Kaggle kernel output and save it durably under runs/.

Usage:  python scripts/pull_run.py <label>
Writes runs/<timestamp>-<label>.md (readable) and .log (raw), and prints a summary.
"""
import datetime
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile

KERNEL = "ta1010/gate1-smoke-test"
label = sys.argv[1] if len(sys.argv) > 1 else "run"

repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
runs = os.path.join(repo, "runs")
os.makedirs(runs, exist_ok=True)

tmp = tempfile.mkdtemp()
subprocess.run(["kaggle", "kernels", "output", KERNEL, "-p", tmp], check=True)
logs = glob.glob(os.path.join(tmp, "*.log"))
if not logs:
    print("no log file pulled"); sys.exit(1)
raw = logs[0]

stdout, gpu, errs = [], set(), set()
for ln in open(raw):
    ln = ln.strip().lstrip(",")
    if not ln.startswith("{"):
        continue
    try:
        o = json.loads(ln)
    except Exception:
        continue
    d = o.get("data", "")
    if o.get("stream_name") == "stdout":
        stdout.append(d)
    for g in ("Tesla T4", "Tesla P100"):
        if g in d:
            gpu.add(g)
    for e in ("AcceleratorError", "no kernel image", "Traceback (most recent call last)"):
        if e in d:
            errs.add(e)
text = "".join(stdout)

ts = datetime.datetime.now().strftime("%Y%m%d-%H%M")
base = os.path.join(runs, f"{ts}-{label}")
shutil.copy(raw, base + ".log")
with open(base + ".md", "w") as f:
    f.write(f"# Kaggle run — {label}\n\n")
    f.write(f"- kernel: `{KERNEL}`  ·  https://www.kaggle.com/code/{KERNEL}\n")
    f.write(f"- GPU: {', '.join(sorted(gpu)) or 'unknown'}\n")
    f.write(f"- errors: {', '.join(sorted(errs)) or 'none'}\n\n")
    f.write("## stdout\n\n```\n" + text.strip() + "\n```\n")

# plain-text of just the model's case outputs
i = text.find("CASE 1")
cases = text[i:].strip() if i >= 0 else text.strip()
with open(base + "-cases.txt", "w") as f:
    f.write(cases + "\n")

# append-only audit index (searchable manifest of every run)
index = os.path.join(runs, "INDEX.tsv")
new = not os.path.exists(index)
with open(index, "a") as f:
    if new:
        f.write("timestamp\tlabel\tgpu\terrors\tfiles\n")
    f.write("\t".join([
        ts, label,
        ",".join(sorted(gpu)) or "unknown",
        ",".join(sorted(errs)) or "none",
        os.path.basename(base),
    ]) + "\n")

print("saved:", base + ".md")
print("saved:", base + "-cases.txt")
print("indexed in:", index)
print("GPU:", ", ".join(sorted(gpu)) or "unknown", "| errors:", ", ".join(sorted(errs)) or "none")
print("---- head of stdout ----")
print(text[:700])
