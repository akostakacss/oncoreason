# Phase 7 — Packaging (the reviewer's front door)

**Status:** ✅ done · **Compute:** CPU-local

## Aim

Turn six phases of code, tests and internal write-ups into something a reviewer can pick up
cold: a front page that states the real headline numbers, does not bury the failures Phase 6
was built to find, and can be verified in one command rather than taken on faith. Packaging is
the only phase whose job is *legibility*, not new capability — no new metrics, no new code.

## Method

| Component | What it does |
|---|---|
| Fresh reproducibility check | Re-ran `pytest -q` and `python scripts/run_pipeline.py` from a clean shell, immediately before writing this document, and diffed the output against every number quoted in `PHASE5_posttraining.md` / `PHASE6_evaluation.md`. |
| `README.md` rewrite | Front page reordered around what a reviewer reads first: what this is, the headline numbers (good *and* bad), how to run it, where the honest limitations and the backlog live. |
| `phases/README.md` status table | Updated to reflect Phase 7 done, so the index and the front page agree. |
| Cross-links preserved | `roadmap_figure.pdf`, `publications/REFERENCES.md`, `docs/MTBBENCH_INTEGRATION.md` — packaging assembles these, it does not replace them. |

## Reasoning: the three decisions that mattered

**1. Package around the findings Phase 6 caught, not despite them.** The two numbers a careful
reviewer will remember are the ones that look bad: 0/50 abstentions and ECE 0.304. A report that
hid these behind the flattering 0.96 concordance would read, to a lab that works on exactly this
kind of failure mode, as either naive or dishonest. Foregrounding them is the more persuasive
move for this specific audience, not just the more honest one.

**2. Reproducibility is a claim, and claims get checked before they get repeated.** Rather than
copy the numbers already sitting in the Phase 5/6 documents, I re-ran both the test suite and
the pipeline fresh, from a clean environment, right before writing this document, and confirmed
the output matches to the last digit (PRM accuracy 0.9135802469135802 → the documented 0.914;
TP 86 / FP 12 / TN 62 / FN 2 exact; concordance 0.96, molecular agreement 0.24, ECE 0.3044,
information-gathering r = -0.256 all exact). This is the same discipline that caught
overclaimed "calibrated confidence" and "runs on GPU" language earlier in the project — a
number that has not been re-derived is a liability, not an asset.

**3. Packaging has a scope boundary, and the MTBBench backlog sits outside it.** It would be easy
to let "package the project" drift into "also wire up MTBBench" or "also fix the abstention
defect" — both are real, both are documented, and neither is done. Phase 7's job is to make the
*current, honest* state legible, not to quietly close out the backlog under the packaging label.
The backlog stays a backlog, visible and dated, in `docs/MTBBENCH_INTEGRATION.md` and the
Honest limitations sections of Phases 5 and 6.

## Results

- **71/71 tests pass** on a clean re-run.
- **`python scripts/run_pipeline.py` reproduces every headline number** quoted anywhere in
  `phases/` and `docs/`, from a single CPU command with no manual steps. No number in this
  repository is asserted without a corresponding artifact under `results/`.
- `README.md` now leads with: what the system does → the headline numbers **including the two
  failures** → one-command quickstart → where the honest limitations and backlog live. Previously
  it led with a phase-status table; the numbers a reviewer actually wants were three clicks away.
- `phases/README.md` status table updated: Phase 7 now reads ✅ done, closing the phase log.

## Interpretation

Packaging did not change what the system does — it changed what a reader encounters first. The
project's actual selling point was never "0.96 concordance" (shown in Phase 6 to be the least
meaningful number in the harness); it is that the evaluation harness found two real defects in
its own system and reported both plainly. That is easy to lose if the front page reads like a
results table instead of a claim that has been checked. This phase makes that the first thing a
reviewer sees rather than something they find on page four.

## Trade-offs, and what I did not do

| Decision | Alternative | Why I chose this |
|---|---|---|
| **Edit pass over existing artifacts** | Generate a new summary report / dashboard | A new document is one more place for the numbers to drift from the code. Editing `README.md` and re-verifying against `results/*.json` keeps a single source of truth. |
| **Re-run everything fresh before writing this doc** | Trust the numbers already written in Phase 5/6 | Cheap to do (CPU, seconds) and directly on-brand for a project whose thesis is "verify before you claim." |
| **Leave the MTBBench backlog undone** | Implement the adapter now, under the "packaging" label | Scope discipline. Phase 7 is about legibility of what exists; starting new code here would blur the phase boundary the whole `phases/` log is built to keep honest. |
| **No new figure** | Regenerate `roadmap_figure.pdf` with a Phase-7 box | The existing figure already shows the five-stage pipeline; Phase 7 packages that pipeline, it is not a sixth stage in the architecture. Adding a box would overstate what changed. |

## Honest limitations

1. This phase does not fix anything Phase 5 or 6 found wrong — the abstention defect, the
   circular guideline gold, and the never-measured audit kappa are all still open, and still
   listed as open in the documents a reviewer will read.
2. "Reviewer-ready" was judged by re-reading the materials as a reviewer would, not by an
   external read; no second person has checked this packaging.

## Tests

No new tests — Phase 7 packages, it does not add capability. The existing **71 passing tests**
are the reproducibility evidence for this phase, re-confirmed on a clean run as documented above.

## Next

All seven phases are now built and documented. What remains is backlog, not phases: the
MTBBench-derived non-circular gold standard, the abstention fix, the Phase 4.3 human audit
kappa, and a baseline arm — all tracked in `docs/MTBBENCH_INTEGRATION.md` and the Honest
limitations sections of `PHASE5_posttraining.md` and `PHASE6_evaluation.md`.
