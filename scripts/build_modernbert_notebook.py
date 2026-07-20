#!/usr/bin/env python3
"""Generate notebooks/prm_modernbert/prm_modernbert.ipynb — stage F.

Stage D established that inlining evidence text lowers TF-IDF's accuracy, because judging
whether a record supports a claim is a *relational* question and bag-of-ngrams sees only the
union of two word bags. That makes this a clean single-variable experiment: hold the feature
string fixed at "step + ids + evidence text" and vary only the model class. If ModernBERT
beats TF-IDF on identical inputs, cross-attention is doing the work.

Input comes from the stage E hard-sampling kernel via `kernel_sources`, so no artifact is
round-tripped through this machine.
"""
import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "notebooks", "prm_modernbert", "prm_modernbert.ipynb")


def _lines(text):
    return text.strip().splitlines(keepends=True)


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": _lines(text)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": _lines(text)}


cells = [
md("""
# Phase 5 · Stage F — ModernBERT vs TF-IDF, feature string held fixed

**The question.** Stage C measured that a PRM trained on constructed negatives scores 0.530
balanced accuracy on real policy output — barely above chance. Stage D then tested the
obvious fix, inlining what each cited record says, and found it made TF-IDF *worse* in every
arm. The diagnosis was that judging support is relational: bag-of-ngrams sees the union of
the claim's words and the evidence's words and has no way to represent whether they
correspond.

**So this is a single-variable test.** Same data, same split, same feature string
(`step + ids + evidence text`). The only thing that changes is the model class.

- If ModernBERT beats TF-IDF here, **cross-attention is doing the work** and the stage D
  result was about model capacity, not about the information being useless.
- If it does not, the problem is the **label distribution**, not the architecture — and no
  amount of model capacity will fix a verifier trained on the wrong negatives.

Either answer is publishable within this project. The second one is the more important
finding if it happens, and it is the one that decides whether RFT/DPO are worth running at
all: a verifier near chance gives a near-random reward signal.

**Outputs** (`/kaggle/working/`): `stageF_modernbert.json`, `prm_modernbert/` (weights).
"""),

md("## 1 · GPU"),
code("!nvidia-smi"),

md("""
## 2 · Install
`transformers>=4.48` is required — ModernBERT landed in 4.48 and earlier versions raise
`KeyError: 'modernbert'` on the config. Do not reinstall torch.
"""),
code("!pip -q install -U 'transformers>=4.48' 'accelerate>=0.34' 2>&1 | tail -2"),

md("""
## 3 · Code, data, and the stage E candidates

The sampled traces arrive from the stage E kernel through `kernel_sources`, mounted read-only
under `/kaggle/input/`. As in stage B the mount is located by searching for its contents
rather than by assuming a path.
"""),
code("""
import os, subprocess, sys, glob, shutil, zipfile, json

REPO = '/tmp/oncoreason'
if not os.path.exists(REPO):
    subprocess.run(['git', 'clone', '--depth', '1',
                    'https://github.com/akostakacss/oncoreason', REPO], check=True)

print('/kaggle/input contains:', sorted(os.listdir('/kaggle/input')))
for z in glob.glob('/kaggle/input/**/*.zip', recursive=True):
    with zipfile.ZipFile(z) as zf:
        zf.extractall('/tmp/unpacked')

hits = (glob.glob('/kaggle/input/**/cases.jsonl', recursive=True)
        + glob.glob('/tmp/unpacked/**/cases.jsonl', recursive=True))
if not hits:
    tree = subprocess.run(['find', '/kaggle/input', '-maxdepth', '3'],
                          capture_output=True, text=True).stdout
    raise FileNotFoundError('cases.jsonl not found:\\n' + tree[:3000])
cases_src = os.path.dirname(hits[0])
cache_src = os.path.join(os.path.dirname(cases_src), 'cache')

os.makedirs(f'{REPO}/data/public', exist_ok=True)
for a, b in [(cases_src, f'{REPO}/data/cases'), (cache_src, f'{REPO}/data/public/cache')]:
    if os.path.isdir(a) and not os.path.exists(b):
        shutil.copytree(a, b)

cand = glob.glob('/kaggle/input/**/candidates.jsonl', recursive=True)
if not cand:
    tree = subprocess.run(['find', '/kaggle/input', '-maxdepth', '3'],
                          capture_output=True, text=True).stdout
    raise FileNotFoundError('candidates.jsonl not found — is the stage E kernel attached '
                            'via kernel_sources?\\n' + tree[:3000])
CANDIDATES = cand[0]
print('candidates:', CANDIDATES, '|', sum(1 for _ in open(CANDIDATES)), 'samples')

# The labelling modules ship in the private dataset rather than the public clone, because
# they are not committed yet. Copy them INTO the cloned repo's scripts/ dir rather than just
# path-inserting the mount: both derive their repo root from __file__, so running them from
# /kaggle/input would resolve data/cases/cases.jsonl against the wrong parent.
for py in glob.glob('/kaggle/input/**/scripts/*.py', recursive=True) + \
          glob.glob('/tmp/unpacked/**/scripts/*.py', recursive=True):
    shutil.copy(py, f'{REPO}/scripts/' + os.path.basename(py))
print('labelling modules:', sorted(os.path.basename(x) for x in
                                   glob.glob(f'{REPO}/scripts/*label*.py')))

subprocess.run([sys.executable, '-m', 'pip', '-q', 'install', '-e', REPO], check=True)
sys.path.insert(0, f'{REPO}/src'); sys.path.insert(0, f'{REPO}/scripts')
"""),

md("""
## 4 · Relabel and build both label sets

Two labels per step, so the experiment is a 2x2 over {backend} x {label}:

- **structural** — the citation resolves and belongs to this case. Bookkeeping. This is what
  stages B-D used, and what stage C measured at 0.530 on real output.
- **semantic** — layered on top: a step is unsound if it *contradicts* the record it cites
  (claims benefit while citing RESISTANCE / DOES_NOT_SUPPORT), names a therapy the record is
  not about, or cites a record for a different tumour type. Structural failure remains the
  floor, so citing nothing is still unsound.

Stage D found that inlining evidence text made TF-IDF worse, and diagnosed it as the label
having no relational question for the extra information to answer. If that diagnosis is right,
the semantic label should reverse the effect. If it is wrong, it will not.
"""),
code("""
from relabel_candidates import build_evidence_pools, normalise
from semantic_labels import build_record_index, label_step

pools, global_ids = build_evidence_pools()
claims = {cid: txt for p in pools.values() for cid, txt in p.items()}
records, therapy_vocab, tumors = build_record_index()
print(f'{len(records)} structured records | {len(therapy_vocab)} therapies')

examples = []
for line in open(CANDIDATES):
    r = json.loads(line)
    pool = pools.get(r['case_id'], {})
    tumor = tumors.get(r['case_id'], 'lung')
    for s in r['steps']:
        resolved, invented, prose = normalise(', '.join(s.get('cited') or []), global_ids)
        off_case = [c for c in resolved if c not in pool]
        structural = bool(resolved) and not invented and not off_case
        sem = label_step(s['text'], resolved, records, tumor, therapy_vocab)
        # structural failure is the floor; semantics can only take a sound step away
        semantic = structural and sem['label_semantic_sound'] and not sem['disease_mismatch']
        examples.append({
            'case_id': r['case_id'], 'step_text': s['text'],
            'evidence_ids': resolved,
            'evidence_text': [claims[c] for c in resolved if c in claims],
            'label_structural': structural,
            'label_semantic': semantic,
            'contradicted': sem['contradicted'],
            'therapy_mismatch': sem['therapy_mismatch'],
            'disease_mismatch': sem['disease_mismatch'],
        })

n = len(examples)
comp = {
    'n_steps': n,
    'structural_sound': sum(e['label_structural'] for e in examples),
    'semantic_sound': sum(e['label_semantic'] for e in examples),
    'uncited': sum(1 for e in examples if not e['evidence_ids']),
    'cited_other_case': sum(1 for e in examples
                            if any(c not in pools[e['case_id']] for c in e['evidence_ids'])),
    'contradicted': sum(e['contradicted'] for e in examples),
    'therapy_mismatch': sum(e['therapy_mismatch'] for e in examples),
    'disease_mismatch': sum(e['disease_mismatch'] for e in examples),
}
for k, v in comp.items():
    print(f'  {k:22} {v:6}' + (f'  ({v/n:.1%})' if k != 'n_steps' else ''))
print()
print('If cited_other_case is now a meaningful share, stage E worked: the policy')
print('now had other case ids available to misuse, which it did not in stage B.')
"""),

md("""
## 5 · Split, and the evaluation helpers

Case-level split via the project's own `split_by_case`, so no case appears on both sides.
Confidence intervals resample **cases**, not steps — resampling steps would break the same
leakage discipline the split enforces, and MTBBench's own evaluation uses case-level bootstrap
resampling with 95% CIs rather than bare point estimates.
"""),
code("""
import random
from oncoreason.training import PRM, PRMConfig, split_by_case

SEED = 17
train, test, test_cases = split_by_case(examples, test_frac=0.3, seed=SEED)
print(f'{len(train)} train / {len(test)} test steps, {len(test_cases)} held-out cases')

def _bal_acc(probs, y):
    tp = sum(1 for p,t in zip(probs,y) if p>=0.5 and t==1)
    fp = sum(1 for p,t in zip(probs,y) if p>=0.5 and t==0)
    tn = sum(1 for p,t in zip(probs,y) if p<0.5 and t==0)
    fn = sum(1 for p,t in zip(probs,y) if p<0.5 and t==1)
    rp = tp/(tp+fn) if tp+fn else 0.0
    rn = tn/(tn+fp) if tn+fp else 0.0
    return (rp+rn)/2, dict(tp=tp, fp=fp, tn=tn, fn=fn, recall_sound=round(rp,3),
                           recall_unsound=round(rn,3))

def evaluate(model, exs, label_key, n_boot=1000):
    probs = model.predict_proba(exs)
    y = [1 if e[label_key] else 0 for e in exs]
    bal, cm = _bal_acc(probs, y)
    by_case = {}
    for p, t, e in zip(probs, y, exs):
        by_case.setdefault(e['case_id'], []).append((p, t))
    cids = list(by_case)
    rng = random.Random(SEED)
    boots = []
    for _ in range(n_boot):
        pick = [by_case[rng.choice(cids)] for _ in cids]
        flat = [pt for grp in pick for pt in grp]
        boots.append(_bal_acc([p for p,_ in flat], [t for _,t in flat])[0])
    boots.sort()
    return {'balanced_accuracy': round(bal,3),
            'ci95': [round(boots[int(.025*n_boot)],3), round(boots[int(.975*n_boot)],3)],
            'accuracy': round((cm['tp']+cm['tn'])/len(exs),3),
            'positive_rate': round(sum(y)/len(exs),3), **cm}
"""),

md("""
## 6 · The 2x2

Four fits. TF-IDF is seconds; ModernBERT is a few minutes each. Every cell sees the identical
feature string (`step + ids + evidence text`) and the identical split — only the backend and
the label change.
"""),
code("""
import time, json

results = {}
for label_key in ('label_structural', 'label_semantic'):
    tr = [dict(e, label_sound=e[label_key]) for e in train]
    te = [dict(e, label_sound=e[label_key]) for e in test]

    m = PRM(PRMConfig(seed=SEED)).fit(tr)
    results[f'tfidf__{label_key}'] = evaluate(m, te, 'label_sound')
    print(f'tfidf      {label_key:18} {results[f"tfidf__{label_key}"]["balanced_accuracy"]:.3f} '
          f'{results[f"tfidf__{label_key}"]["ci95"]}')

    t0 = time.time()
    # max_length 512, not the 4096 default: these steps plus evidence are short, and padding
    # to 4096 would spend the T4 on padding tokens for no information gain.
    cfg = PRMConfig(backend='modernbert', seed=SEED, epochs=3, batch_size=8,
                    lr=2e-5, max_length=512)
    mb = PRM(cfg).fit(tr)
    results[f'modernbert__{label_key}'] = evaluate(mb, te, 'label_sound')
    print(f'modernbert {label_key:18} '
          f'{results[f"modernbert__{label_key}"]["balanced_accuracy"]:.3f} '
          f'{results[f"modernbert__{label_key}"]["ci95"]}  ({time.time()-t0:.0f}s)')
    if label_key == 'label_semantic':
        mb.save('/kaggle/working/prm_modernbert_semantic')
"""),

md("## 7 · Report and verdict"),
code("""
report = {
    'seed': SEED,
    'features': 'step + ids + evidence text (identical in every cell)',
    'source': os.path.basename(CANDIDATES),
    'n_train': len(train), 'n_test': len(test), 'n_test_cases': len(test_cases),
    'label_composition': comp,
    'results': results,
    'base_model': 'answerdotai/ModernBERT-base',
    'ci_method': 'case-level bootstrap, 1000 iterations, 95 percent',
}
json.dump(report, open('/kaggle/working/stageF_modernbert.json','w'), indent=1)
print(json.dumps(report['results'], indent=1))

best = max(results.values(), key=lambda r: r['balanced_accuracy'])
best_key = [k for k,v in results.items() if v is best][0]
d_model = (results['modernbert__label_semantic']['balanced_accuracy']
           - results['tfidf__label_semantic']['balanced_accuracy'])
d_label = (results['modernbert__label_semantic']['balanced_accuracy']
           - results['modernbert__label_structural']['balanced_accuracy'])

print()
print(f'best cell: {best_key} at {best["balanced_accuracy"]} {best["ci95"]}')
print(f'model-class effect (semantic labels): {d_model:+.3f}')
print(f'label effect (modernbert):            {d_label:+.3f}')
print()
if best['balanced_accuracy'] < 0.65 or best['ci95'][0] < 0.55:
    print('GATE: FAIL. No cell is reliably usable as a reward signal. RFT/DPO would inherit '
          'a near-random selection rule, so do not run them on this verifier. The honest '
          'deliverable is this measurement.')
else:
    print(f'GATE: PASS on {best_key}. A verifier at {best["balanced_accuracy"]} '
          f'{best["ci95"]} is usable for trace selection; RFT/DPO can proceed with the '
          'reward signal named explicitly.')
"""),
]

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}

failed = 0
for i, c in enumerate(cells):
    if c["cell_type"] != "code":
        continue
    src = "".join(c["source"])
    if src.lstrip().startswith("!"):
        continue
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
