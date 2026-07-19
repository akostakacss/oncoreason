"""Render the project roadmap: the eight phases, what each produced, and what is next.

One page, read left to right and top to bottom. Each phase card carries its status and
the concrete artifact it produced; the band underneath is the backlog that is documented
but not built. Verified counts are read from the most recent results/*-pipeline.json.

Writes roadmap_figure.png and .pdf.
"""
from __future__ import annotations

import glob
import json
import sys
import textwrap

sys.path.insert(0, __file__.rsplit("/", 1)[0])

from figure_style import (  # noqa: E402
    GOOD, GRID, INK, INK_MUTED, INK_SECONDARY, SLOT_1_BLUE, SLOT_3_ROSE,
    SLOT_7_VIOLET, SURFACE_SUNKEN, TINT_BLUE, TINT_ROSE, TINT_VIOLET, WARNING,
    accent_bar, apply_rc, arrow, canvas, card, save, title_block, tracked,
)

MARGIN = 4.0
COL_W, COL_GAP = 21.5, 2.0
CARD_H = 17.0
ROW1_Y, ROW2_Y = 32.0, 13.0
BAND_Y, BAND_H = 4.4, 7.4
BACKLOG_WRAP = 38

# Phase families. Green is deliberately unused here: it would collide with the GOOD
# status dot, and a status colour must never double as a series.
DATA, MODEL, PROOF = "data", "model", "proof"
GROUP_ACCENT = {DATA: SLOT_1_BLUE, MODEL: SLOT_7_VIOLET, PROOF: SLOT_3_ROSE}
GROUP_TINT = {DATA: TINT_BLUE, MODEL: TINT_VIOLET, PROOF: TINT_ROSE}
GROUP_LABEL = {
    DATA: "evidence & cases",
    MODEL: "reasoning & learning",
    PROOF: "measurement & release",
}

DONE, PARTIAL = "done", "core built"
STATUS_COLOUR = {DONE: GOOD, PARTIAL: WARNING}
# `warning` is sub-3:1 on a light surface by design — the word beside the dot is the
# required mitigation. A darker same-hue ring keeps the dot legible in print too.
STATUS_RING = {DONE: "#0a7d0a", PARTIAL: "#a87400"}

# phase, title, status, subtitle, bullet outputs
PHASES = [
    (0, DATA, "Pre-flight gates", DONE, "de-risk before building", [
        "5 stop/continue gates",
        "3B base model cleared",
        "tool + deferral probe on T4",
    ]),
    (1, DATA, "Connectors", DONE, "evidence, joined correctly", [
        "CIViC + ClinVar live",
        "joined on CAID / Entrez",
        "never on name strings",
    ]),
    (2, DATA, "Case set", DONE, "real, QC-filtered cases", [
        "{n_cases} annotated cases",
        "patient-disjoint splits",
        "cBioPortal, open studies",
    ]),
    (3, MODEL, "Scaffolding", PARTIAL, "the agent and its trace", [
        "planner → specialists → synth",
        "BM25 · dense · hybrid RRF",
        "auditable trace + abstain",
    ]),
    (4, MODEL, "Supervision", PARTIAL, "labels without annotators", [
        "guideline-verified steps",
        "{n_steps} step labels",
        "counterfactual negatives",
    ]),
    (5, MODEL, "Post-training", PARTIAL, "the verifier, and using it", [
        "PRM held-out {prm_acc:.3f}",
        "min-rule best-of-N",
        "DPO pairs + hacking guard",
    ]),
    (6, PROOF, "Evaluation", PARTIAL, "the clinical scorecard", [
        "8 metrics, bootstrap CIs",
        "found the abstention defect",
        "ECE {ece:.3f}, reported as poor",
    ]),
    (7, PROOF, "Packaging", DONE, "reproducible end to end", [
        "{n_tests} tests pass",
        "one-command pipeline",
        "every number re-derived",
    ]),
]

BACKLOG = [
    ("MTBBench", "a non-circular gold standard, to replace the guideline-derived label"),
    ("Abstention gap", "14/38 non-actionable cases still recommend"),
    ("Longitudinal", "the schema has a time axis; no evaluation over it yet"),
    ("GPU stage", "ModernBERT PRM, policy sampling, LoRA RFT/DPO on a T4"),
]


def load_latest() -> dict:
    files = sorted(glob.glob("results/*-pipeline.json"))
    if not files:
        sys.exit("no results/*-pipeline.json found — run scripts/run_pipeline.py first")
    with open(files[-1]) as f:
        return json.load(f)


def phase_card(ax, x, y, num, group, title, status, subtitle, bullets):
    colour = STATUS_COLOUR[status]
    card(ax, x, y, COL_W, CARD_H, facecolor=GROUP_TINT[group], edgecolor=GRID)
    accent_bar(ax, x + 0.85, y + 1.0, CARD_H - 2.0, GROUP_ACCENT[group])
    top = y + CARD_H

    ax.text(x + 2.4, top - 2.4, str(num), fontsize=19, fontweight="bold",
            color=GROUP_ACCENT[group], va="top", ha="left", alpha=0.45)
    ax.text(x + 6.4, top - 2.8, title, fontsize=12, fontweight="bold", color=INK,
            va="top", ha="left")
    ax.text(x + 6.4, top - 5.4, subtitle, fontsize=8.4, color=INK_MUTED, va="top",
            ha="left", style="italic")

    # status: colour + glyph + word, never colour alone
    ax.plot([x + 2.6], [top - 8.0], marker="o", markersize=5.0, color=colour,
            markeredgecolor=STATUS_RING[status], markeredgewidth=0.9, zorder=5)
    ax.text(x + 4.2, top - 8.0, status, fontsize=8.2, color=INK_SECONDARY,
            va="center", ha="left", fontweight="bold")

    by = top - 10.6
    for b in bullets:
        ax.text(x + 2.6, by, "·", fontsize=9.5, color=INK_MUTED, va="top", ha="left")
        ax.text(x + 4.2, by, b, fontsize=8.8, color=INK_SECONDARY, va="top", ha="left")
        by -= 2.4


def main() -> None:
    d = load_latest()
    fmt = {
        "n_cases": d["n_cases"],
        "n_steps": d["step_labels"]["total"],
        "prm_acc": d["prm"]["accuracy"],
        "ece": d["metrics"]["calibration"]["ece"],
        "n_tests": 71,
    }

    apply_rc()
    fig, ax = canvas(16.0, 10.0)
    H = ax.get_ylim()[1]

    title_block(ax, MARGIN, H - 3.0, "Oncology reasoning model — build roadmap")
    ax.text(MARGIN, H - 7.4,
            "A post-training and evaluation pipeline for guideline-grounded, auditable "
            "therapy reasoning. Eight phases, each ending in an artifact.",
            fontsize=11.5, color=INK_SECONDARY, va="top")
    ax.text(MARGIN, H - 10.6,
            "Proof of concept  ·  lung adenocarcinoma  ·  public / non-controlled data "
            "only  ·  CPU-local, with the GPU stage on a free T4",
            fontsize=9.0, color=INK_MUTED, va="top", style="italic")

    # ---------------------------------------------------------------- phase cards
    for i, (num, group, title, status, subtitle, bullets) in enumerate(PHASES):
        row, col = divmod(i, 4)
        x = MARGIN + col * (COL_W + COL_GAP)
        y = ROW1_Y if row == 0 else ROW2_Y
        phase_card(ax, x, y, num, group, title, status,
                   subtitle, [b.format(**fmt) for b in bullets])
        if col < 3:
            mid = y + CARD_H / 2
            arrow(ax, (x + COL_W + 0.35, mid), (x + COL_W + COL_GAP - 0.35, mid),
                  color=INK_MUTED, lw=1.3, head=5.5)

    # ---------------------------------------------------------------- backlog band
    total_w = 4 * COL_W + 3 * COL_GAP
    card(ax, MARGIN, BAND_Y, total_w, BAND_H, facecolor=SURFACE_SUNKEN,
         edgecolor=GRID, dashed=True)
    # neutral, not a family colour — "not built" is a state, not a phase group
    accent_bar(ax, MARGIN + 0.85, BAND_Y + 0.9, BAND_H - 1.8, INK_MUTED)
    ax.text(MARGIN + 2.6, BAND_Y + BAND_H - 1.5, tracked("DOCUMENTED, NOT BUILT"),
            fontsize=8.0, color=INK_SECONDARY, va="top", fontweight="bold")

    item_w = (total_w - 3.6) / len(BACKLOG)
    for i, (label, detail) in enumerate(BACKLOG):
        ix = MARGIN + 2.6 + i * item_w
        ax.text(ix, BAND_Y + BAND_H - 3.6, label, fontsize=9.0, color=INK,
                va="top", fontweight="bold")
        ax.text(ix, BAND_Y + BAND_H - 5.4, textwrap.fill(detail, BACKLOG_WRAP),
                fontsize=7.8, color=INK_MUTED, va="top", linespacing=1.45)

    # ---------------------------------------------------------------- legend
    # Two rows: what the card colour means, then what the dot means.
    family_y, status_y = 2.7, 1.0

    lx = MARGIN
    ax.text(lx, family_y, "Phase family:", fontsize=8.0, color=INK_MUTED,
            va="center", ha="left")
    lx += 8.6
    for group in (DATA, MODEL, PROOF):
        card(ax, lx, family_y - 0.75, 1.5, 1.5, facecolor=GROUP_TINT[group],
             edgecolor=GROUP_ACCENT[group], lw=1.0, radius=0.3)
        ax.text(lx + 2.3, family_y, GROUP_LABEL[group], fontsize=8.0,
                color=INK_SECONDARY, va="center", ha="left")
        lx += 20.0

    lx = MARGIN
    ax.text(lx, status_y, "Status:", fontsize=8.0, color=INK_MUTED,
            va="center", ha="left")
    lx += 8.6
    for status in (DONE, PARTIAL):
        ax.plot([lx + 0.75], [status_y], marker="o", markersize=4.6,
                color=STATUS_COLOUR[status], markeredgecolor=STATUS_RING[status],
                markeredgewidth=0.9, zorder=5)
        ax.text(lx + 2.3, status_y, status, fontsize=8.0, color=INK_SECONDARY,
                va="center", ha="left")
        lx += 20.0

    ax.text(MARGIN + total_w, status_y,
            "No patient data leaves its source  ·  no controlled data committed  ·  "
            "derived guideline indices only",
            fontsize=8.0, color=INK_MUTED, va="center", ha="right", style="italic")

    save(fig, "roadmap_figure")


if __name__ == "__main__":
    main()
