#!/usr/bin/env python3
"""Pull a Kaggle kernel's output and archive it durably under runs/.

Usage:  python scripts/pull_run.py <stage> <label>

Writes runs/<timestamp>-<label>.md (readable), .log (raw), and — for training stages —
runs/<timestamp>-<label>-artifacts/ holding every non-log file the kernel produced
(adapters, checkpoints, sampled traces). Earlier versions downloaded those into a
tempdir and discarded them, so a multi-hour training run survived only as a transcript.
"""
import datetime
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile

if len(sys.argv) < 3:
    sys.exit("usage: python scripts/pull_run.py <stage> <label>")
stage, label = sys.argv[1], sys.argv[2]

repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
meta_path = os.path.join(repo, "notebooks", stage, "kernel-metadata.json")
if not os.path.exists(meta_path):
    sys.exit(f"no kernel-metadata.json for stage {stage!r} ({meta_path})")
KERNEL = json.load(open(meta_path))["id"]

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
    for e in ("AcceleratorError", "no kernel image", "Traceback (most recent call last)",
              "CUDA out of memory"):
        if e in d:
            errs.add(e)
text = "".join(stdout)

ts = datetime.datetime.now().strftime("%Y%m%d-%H%M")
base = os.path.join(runs, f"{ts}-{label}")
shutil.copy(raw, base + ".log")

# Everything the kernel produced other than the run log: model adapters, sampled traces,
# reports. These are the actual deliverable of a training stage.
artifacts = []
art_dir = base + "-artifacts"
for path in sorted(glob.glob(os.path.join(tmp, "**", "*"), recursive=True)):
    if os.path.isdir(path) or path == raw:
        continue
    rel = os.path.relpath(path, tmp)
    dest = os.path.join(art_dir, rel)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    shutil.copy2(path, dest)
    artifacts.append((rel, os.path.getsize(path)))

with open(base + ".md", "w") as f:
    f.write(f"# Kaggle run — {label}\n\n")
    f.write(f"- stage: `{stage}`\n")
    f.write(f"- kernel: `{KERNEL}`  ·  https://www.kaggle.com/code/{KERNEL}\n")
    f.write(f"- GPU: {', '.join(sorted(gpu)) or 'unknown'}\n")
    f.write(f"- errors: {', '.join(sorted(errs)) or 'none'}\n")
    if artifacts:
        f.write(f"\n## artifacts ({len(artifacts)})\n\n")
        for rel, size in artifacts:
            f.write(f"- `{rel}` — {size / 1e6:.2f} MB\n")
    f.write("\n## stdout\n\n```\n" + text.strip() + "\n```\n")

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
if artifacts:
    print(f"saved: {art_dir}/ ({len(artifacts)} files, "
          f"{sum(s for _, s in artifacts) / 1e6:.1f} MB)")
print("indexed in:", index)
print("GPU:", ", ".join(sorted(gpu)) or "unknown", "| errors:", ", ".join(sorted(errs)) or "none")
print("---- head of stdout ----")
print(text[:700])
