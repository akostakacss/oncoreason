#!/usr/bin/env python3
"""Generate notebooks/policy_sampling/policy_sampling.ipynb.

The notebook is generated rather than hand-edited so the sampling configuration lives in
version control as readable Python instead of buried in notebook JSON.
"""
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# "plain": each case sees only its own evidence (stage B).
# "hard":  distractor evidence from other cases is mixed in and a citation is mandatory,
#          so the negative class becomes citation *correctness* rather than *presence*.
MODE = sys.argv[1] if len(sys.argv) > 1 else "plain"
STAGE = {"plain": "policy_sampling", "hard": "policy_sampling_hard"}[MODE]
OUT = os.path.join(REPO, "notebooks", STAGE, f"{STAGE}.ipynb")


# nbformat concatenates `source` entries directly, so each line must keep its own newline.
# Splitting on "\n" without them collapses the whole cell onto one line.
def _lines(text):
    return text.strip().splitlines(keepends=True)


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": _lines(text)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": _lines(text)}


DISTRACTORS = """

# Hard mode: mix in evidence drawn from OTHER cases, shown to the policy indistinguishably
# from the real pool. Citing one is a genuine error that is still mechanically detectable,
# which turns the negative class from "cited nothing" into "cited the wrong thing". It also
# resurrects the `swap` failure mode stage B found never occurs naturally, because a policy
# shown only one case's evidence has no other case's ids available to misuse.
import random
rng = random.Random(17)
all_items = [(cid, txt) for c in cases for cid, txt in c['evidence'].items()]
K_DISTRACTORS = 4
for c in cases:
    own = set(c['evidence'])
    pool = [it for it in all_items if it[0] not in own]
    c['distractors'] = dict(rng.sample(pool, min(K_DISTRACTORS, len(pool))))
    merged = list(c['evidence'].items()) + list(c['distractors'].items())
    rng.shuffle(merged)                      # so position does not leak which are real
    c['prompt_pool'] = dict(merged)
nd = [len(c['distractors']) for c in cases]
print(f'distractors per case: mean {sum(nd)/len(nd):.1f} '
      f'| prompt pool mean {sum(len(c["prompt_pool"]) for c in cases)/len(cases):.1f}')
"""

# In plain mode a step may legitimately carry no citation; in hard mode that escape is closed
# and the pool contains irrelevant records, so the failure the policy can still make is citing
# the WRONG record — which is the distinction the verifier actually needs to learn.
RULE1_PLAIN = """1. Every factual claim about a variant, therapy or guideline must cite evidence ids from the
   EVIDENCE list. Do not cite an id that is not listed."""

RULE1_HARD = """1. EVERY step must end with a [CITE: ...] list naming ids from the EVIDENCE list.
   A step with no citation is not acceptable.
1b. The EVIDENCE list contains records that do NOT bear on this patient's profile. Cite only
   records that genuinely support the claim you are making. Citing an irrelevant record is
   worse than citing fewer."""

RULE1 = RULE1_HARD if MODE == "hard" else RULE1_PLAIN

cells = [
md("""
# Phase 5 · Stage B — Policy sampling

**What this produces:** N candidate reasoning traces per case, sampled from
Qwen2.5-3B-Instruct, with each cited evidence id checked against the evidence actually
retrieved for that case.

**Why it is the keystone of the GPU stage.** The deterministic scaffold emits citations only
from real retrievals, so it *cannot* produce an ungrounded step — which is why the PRM is
currently trained on counterfactual negatives constructed by `supervision.mine_negatives`,
and why its 0.914 held-out accuracy is on a semi-synthetic distribution. A sampling policy
hallucinates for real. This stage produces those real negatives, and it produces the N>1
candidates per case that Best-of-N needs in order to mean anything.

**Outputs** (to `/kaggle/working/`): `candidates.jsonl`, `sampling_report.json`.
"""),

md("## 1 · GPU"),
code("!nvidia-smi"),

md("""
## 2 · Install

Only `transformers` + `accelerate`. **Do not reinstall torch** — Kaggle ships it matched to
the GPU driver.
"""),
code("!pip -q install -U 'transformers>=4.44' 'accelerate>=0.34' 2>&1 | tail -2"),

md("""
## 3 · Code and data

Code comes from the public repo; cases and the cached CIViC/ClinVar/cBioPortal responses come
from the private `oncoreason-data` dataset (they are git-ignored as regenerable, so they are
not in the clone). Unpacking them into the repo's own `data/` layout means the datasource
cache paths resolve with no code change, and evidence retrieval is reproducible offline.
"""),
code("""
import os, subprocess, sys, zipfile, glob, shutil

REPO = '/tmp/oncoreason'
if not os.path.exists(REPO):
    subprocess.run(['git', 'clone', '--depth', '1',
                    'https://github.com/akostakacss/oncoreason', REPO], check=True)

# Locate the attached data by searching for its contents rather than assuming a mount path:
# Kaggle derives the mount directory from the dataset slug and expands uploaded archives, so
# hardcoding /kaggle/input/<name>/<file>.zip is two assumptions that can each break silently.
print('/kaggle/input contains:', sorted(os.listdir('/kaggle/input')))

for z in glob.glob('/kaggle/input/**/*.zip', recursive=True):
    with zipfile.ZipFile(z) as zf:
        zf.extractall('/tmp/unpacked')

hits = (glob.glob('/kaggle/input/**/cases.jsonl', recursive=True)
        + glob.glob('/tmp/unpacked/**/cases.jsonl', recursive=True))
if not hits:
    tree = subprocess.run(['find', '/kaggle/input', '-maxdepth', '3'],
                          capture_output=True, text=True).stdout
    raise FileNotFoundError('cases.jsonl not found under /kaggle/input:\\n' + tree[:3000])

cases_src = os.path.dirname(hits[0])
cache_src = os.path.join(os.path.dirname(cases_src), 'cache')
print('cases from:', cases_src, '| cache from:', cache_src)

os.makedirs(f'{REPO}/data/public', exist_ok=True)
for a, b in [(cases_src, f'{REPO}/data/cases'), (cache_src, f'{REPO}/data/public/cache')]:
    if os.path.isdir(a) and not os.path.exists(b):
        shutil.copytree(a, b)   # copy, not move: /kaggle/input is read-only

subprocess.run([sys.executable, '-m', 'pip', '-q', 'install', '-e', REPO], check=True)
sys.path.insert(0, f'{REPO}/src')
print('cases:', sum(1 for _ in open(f'{REPO}/data/cases/cases.jsonl')))
print('cached evidence files:', len(glob.glob(f'{REPO}/data/public/cache/**/*.json', recursive=True)))
"""),

md("""
## 4 · Build the evidence pool per case

For each case, retrieve exactly what the CPU pipeline retrieves: CIViC/ClinVar evidence per
alteration plus the guideline chunks. This pool is what goes into the prompt **and** what a
citation is later checked against — so "cited an id that was never retrieved for this case"
is a mechanically detectable hallucination rather than a judgement call.
"""),
code("""
import json
from oncoreason.agents.guideline_index import index_docs
from oncoreason.agents.tools import (variant_lookup, guideline_lookup,
                                     evidence_citations, guideline_citations)
from oncoreason.datasources import get_source
from oncoreason.retrieval.base import BM25Retriever

guideline = BM25Retriever(source='esmo_index'); guideline.index(index_docs())
civic, clinvar = get_source('civic'), get_source('clinvar')

cases = []
with open(f'{REPO}/data/cases/cases.jsonl') as f:
    for line in f:
        d = json.loads(line)
        tumor = d['context'].get('tumor_type', 'lung')
        pool = {}
        for a in d['alterations']:
            ev, _ = variant_lookup(a['gene'], a['variant'], tumor, civic, clinvar)
            for c in evidence_citations(ev):
                pool[c.citation_id] = c.claim
            chunks, _ = guideline_lookup(a['gene'], tumor, guideline)
            for c in guideline_citations(chunks):
                pool[c.citation_id] = c.claim
        cases.append({'case_id': d['case_id'], 'alterations': d['alterations'],
                      'context': d['context'], 'evidence': pool})

GLOBAL_IDS = {cid for c in cases for cid in c['evidence']}
sizes = [len(c['evidence']) for c in cases]
print(f'{len(cases)} cases | evidence per case: min {min(sizes)} '
      f'mean {sum(sizes)/len(sizes):.1f} max {max(sizes)} | {len(GLOBAL_IDS)} distinct ids')
""" + (DISTRACTORS if MODE == "hard" else "")),

md("""
## 5 · Prompt

The output format is strict so steps and citations parse mechanically — an unparseable
sample is a data point about the policy, not something to repair by hand. Framing matches the
Gate 1 probe: assistive, deferral-positive, claims tied to evidence.
"""),
code(('''
SYSTEM = """You are a decision-support assistant for a molecular tumor board. You assist
clinicians and do not replace them. You defer when the evidence is insufficient.

You are given a molecular profile and a numbered EVIDENCE list. Reason in explicit steps.

RULES:
{RULE1}
2. If the evidence does not support a recommendation, defer. A deferral beats a guess.
3. Do not overstate certainty.
4. Use EXACTLY this format and write nothing after the DEFER line:

STEP 1: <one reasoning step> [CITE: id, id]
STEP 2: <one reasoning step> [CITE: id]
...
RECOMMENDATION: <ranked therapy options, or "defer">
CONFIDENCE: <0.00-1.00>
DEFER: <yes or no>"""

def build_prompt(case):
    alts = "; ".join(f"{a['gene']} {a['variant']} ({a.get('kind','mutation')})"
                     for a in case['alterations'])
    ctx = case['context']
    # what the policy is shown may include distractors; what a citation is judged against
    # is always the case's real evidence, so the two must not be conflated here
    shown = case.get('prompt_pool', case['evidence'])
    ev = "\\n".join(f"  {cid}: {claim}" for cid, claim in shown.items()) or "  (none retrieved)"
    return (f"TUMOR TYPE: {ctx.get('tumor_type','lung')}\\n"
            f"STAGE: {ctx.get('stage') or 'unknown'}\\n"
            f"ALTERATIONS: {alts}\\n\\n"
            f"EVIDENCE:\\n{ev}\\n\\n"
            f"Assess the actionability of this profile and give a recommendation.")

print(build_prompt(cases[2])[:900])
''').replace("{RULE1}", RULE1)),   # plain .format would collide with the cell's f-strings

md("""
## 6 · Load the policy

Attached via `model_sources` so it loads from local disk rather than downloading. fp16 on a
single T4; the model is only doing inference here, so no quantisation is needed.
"""),
code("""
import torch, time, glob
from transformers import AutoModelForCausalLM, AutoTokenizer

local = glob.glob('/kaggle/input/qwen2.5/transformers/3b-instruct/*/')
MODEL = local[0] if local else 'Qwen/Qwen2.5-3B-Instruct'
print('loading', MODEL)

tok = AutoTokenizer.from_pretrained(MODEL)
tok.padding_side = 'left'
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16,
                                             device_map={'': 0})
model.eval()
print('loaded |', torch.cuda.get_device_name(0),
      f'| {torch.cuda.memory_allocated()/1e9:.1f} GB allocated')
"""),

md("""
## 7 · Sample

`N_SAMPLES` per case in one batched `generate` call (`num_return_sequences`), which is far
cheaper than N sequential passes.

**Temperature is the knob that decides whether this stage is useful.** Too low and the N
samples are identical, so there is nothing for Best-of-N to rank and no DPO pair clears the
margin; too high and the output is incoherent, so the verifier learns a distinction that is
trivially easy. 0.8 is the starting point and the spread reported at the end is what tells
you whether to move it.

Results are appended per case, so a timeout costs one case rather than the run.
"""),
code("""
import re, json, os

N_SAMPLES, TEMPERATURE, TOP_P, MAX_NEW = 8, 0.8, 0.95, 700
SEED = 17
torch.manual_seed(SEED)

OUTF = '/kaggle/working/candidates.jsonl'
open(OUTF, 'w').close()

STEP_RE = re.compile(r'^STEP\\s*(\\d+)\\s*:\\s*(.*?)\\s*(?:\\[CITE:\\s*([^\\]]*)\\])?\\s*$',
                     re.IGNORECASE)

def parse(text, case):
    \"\"\"Parse one generation into steps + citations, and label each step structurally.\"\"\"
    steps, rec, conf, defer = [], [], None, False
    for line in text.split('\\n'):
        line = line.strip()
        m = STEP_RE.match(line)
        if m:
            cited = [c.strip() for c in (m.group(3) or '').split(',') if c.strip()]
            unresolvable = [c for c in cited if c not in GLOBAL_IDS]
            off_case = [c for c in cited if c in GLOBAL_IDS and c not in case['evidence']]
            steps.append({
                'index': int(m.group(1)), 'text': m.group(2),
                'cited': cited, 'unresolvable': unresolvable, 'off_case': off_case,
                # a real negative: an invented id, an id belonging to another case, or a
                # claim asserted with no supporting record at all
                'label_sound': not unresolvable and not off_case and bool(cited),
            })
        elif line.upper().startswith('RECOMMENDATION:'):
            rec = [t.strip() for t in line.split(':', 1)[1].split(';') if t.strip()]
        elif line.upper().startswith('CONFIDENCE:'):
            try:
                conf = float(re.findall(r'[\\d.]+', line)[0])
            except Exception:
                conf = None
        elif line.upper().startswith('DEFER:'):
            defer = 'yes' in line.lower()
    return {'steps': steps, 'recommendation': rec, 'confidence': conf,
            'abstained': defer, 'parsed_ok': bool(steps) and rec is not None}

t_start = time.time()
for i, case in enumerate(cases, 1):
    msgs = [{'role': 'system', 'content': SYSTEM},
            {'role': 'user', 'content': build_prompt(case)}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = tok(prompt, return_tensors='pt').to(model.device)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=True,
                             temperature=TEMPERATURE, top_p=TOP_P,
                             num_return_sequences=N_SAMPLES,
                             pad_token_id=tok.pad_token_id)
    dt = time.time() - t0
    gen = [tok.decode(o[enc['input_ids'].shape[1]:], skip_special_tokens=True) for o in out]

    with open(OUTF, 'a') as f:
        for k, g in enumerate(gen):
            p = parse(g, case)
            f.write(json.dumps({'case_id': case['case_id'], 'sample': k,
                                'model': MODEL, 'temperature': TEMPERATURE,
                                'seed': SEED, 'raw': g, **p}) + '\\n')
    ok = sum(parse(g, case)['parsed_ok'] for g in gen)
    print(f'CASE {i}/{len(cases)} {case["case_id"][:44]:<44} '
          f'{dt:5.1f}s  parsed {ok}/{N_SAMPLES}', flush=True)

print(f'\\ntotal {(time.time()-t_start)/60:.1f} min')
"""),

md("""
## 8 · Report

The numbers to read before spending another GPU session:

- **parse rate** — if low, the prompt format is losing, not the model.
- **unsound step rate** — this is the whole point. If it is ~0 the policy is not hallucinating
  and the temperature is too low; if it is ~1 the samples are noise.
- **score spread per case** — near-zero spread means Best-of-N and DPO have nothing to work
  with, whatever the verifier says.
"""),
code("""
import statistics, json

rows = [json.loads(l) for l in open(OUTF)]
steps = [s for r in rows for s in r['steps']]
unsound = [s for s in steps if not s['label_sound']]

by_case = {}
for r in rows:
    by_case.setdefault(r['case_id'], []).append(r)
spreads = []
for cid, rs in by_case.items():
    fr = [sum(s['label_sound'] for s in r['steps']) / max(1, len(r['steps'])) for r in rs]
    if len(fr) > 1:
        spreads.append(max(fr) - min(fr))

report = {
    'n_samples': len(rows),
    'n_cases': len(by_case),
    'samples_per_case': N_SAMPLES,
    'temperature': TEMPERATURE,
    'model': MODEL,
    'parse_rate': round(sum(r['parsed_ok'] for r in rows) / len(rows), 3),
    'n_steps': len(steps),
    'unsound_step_rate': round(len(unsound) / max(1, len(steps)), 3),
    'invented_id_steps': sum(1 for s in steps if s['unresolvable']),
    'off_case_id_steps': sum(1 for s in steps if s['off_case']),
    'uncited_steps': sum(1 for s in steps if not s['cited']),
    'abstain_rate': round(sum(r['abstained'] for r in rows) / len(rows), 3),
    'mean_within_case_spread': round(statistics.fmean(spreads), 3) if spreads else None,
    'mean_steps_per_sample': round(statistics.fmean([len(r['steps']) for r in rows]), 2),
}
json.dump(report, open('/kaggle/working/sampling_report.json', 'w'), indent=1)
for k, v in report.items():
    print(f'{k:28} {v}')

print('\\n--- examples of real ungrounded steps ---')
for s in unsound[:5]:
    why = ('invented ' + ','.join(s['unresolvable'])) if s['unresolvable'] else \\
          ('off-case ' + ','.join(s['off_case'])) if s['off_case'] else 'no citation'
    print(f'  [{why}] {s["text"][:100]}')
"""),
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

# Compile every code cell the way papermill reconstructs it — by concatenating `source`
# with no separator. Joining on "\n" instead would hide a cell whose lines lost their
# newlines, which is exactly how the first run of this stage died on the GPU.
failed = 0
for i, c in enumerate(cells):
    if c["cell_type"] != "code":
        continue
    src = "".join(c["source"])
    if src.lstrip().startswith("!"):
        continue                      # shell magic is not valid Python
    try:
        compile(src, f"cell{i}", "exec")
    except SyntaxError as e:
        failed += 1
        print(f"  cell {i}: SyntaxError line {e.lineno}: {e.msg}")
if failed:
    raise SystemExit(f"{failed} cell(s) failed to compile; not writing {OUT}")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    json.dump(nb, f, indent=1)
print(f"wrote {OUT} ({len(cells)} cells, all code cells compile)")
