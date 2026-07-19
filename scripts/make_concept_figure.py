"""Render the project's concept figure: what the system does and what it produces.

The thesis in one page — a reasoning model adjudicates the population-level guideline
prior against an individual molecular signal, and emits a recommendation that cites its
evidence, carries calibrated uncertainty and can abstain.

Headline numbers are read from the most recent results/*-pipeline.json rather than typed,
so the figure cannot drift from the pipeline. Writes concept_figure.png and .pdf.
"""
from __future__ import annotations

import glob
import json
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])

from figure_style import (  # noqa: E402
    GRID, INK, INK_MUTED, INK_SECONDARY, SLOT_1_BLUE, SLOT_2_GREEN, SLOT_3_ROSE,
    SLOT_7_VIOLET, SURFACE_SUNKEN, TINT_BLUE, TINT_GREEN, TINT_ROSE, TINT_VIOLET,
    accent_bar, apply_rc, arrow, canvas, card, eyebrow, save, title_block, tracked,
)

# ---- layout constants (canvas units; canvas is 100 wide) ---------------------------
MARGIN = 5.0
FOOT_Y = 1.2
BAND_Y, BAND_H = 3.2, 6.8            # evaluation band
MAIN_Y = 13.4                        # bottom of the three main columns
MAIN_H = 22.1
CARD_H = 10.0                        # each input card
CARD_GAP = 2.1

COL_IN_X, COL_IN_W = MARGIN, 30.0
COL_AD_X, COL_AD_W = 41.5, 25.0
COL_OUT_X, COL_OUT_W = 72.0, 23.0


def load_latest() -> dict:
    files = sorted(glob.glob("results/*-pipeline.json"))
    if not files:
        sys.exit("no results/*-pipeline.json found — run scripts/run_pipeline.py first")
    with open(files[-1]) as f:
        return json.load(f)


def input_card(ax, y, colour, tint, heading, tag, detail, footnote):
    card(ax, COL_IN_X, y, COL_IN_W, CARD_H, facecolor=tint, edgecolor=GRID)
    accent_bar(ax, COL_IN_X + 0.9, y + 1.0, CARD_H - 2.0, colour)
    tx = COL_IN_X + 2.6
    ax.text(tx, y + CARD_H - 2.6, heading, fontsize=12.5, fontweight="bold",
            color=INK, va="top")
    ax.text(COL_IN_X + COL_IN_W - 1.8, y + CARD_H - 2.7, tag, fontsize=10,
            color=colour, va="top", ha="right", style="italic", fontweight="bold")
    ax.text(tx, y + CARD_H - 6.1, detail, fontsize=9.4, color=INK_SECONDARY, va="top")
    ax.text(tx, y + 1.4, footnote, fontsize=8.6, color=INK_MUTED, va="bottom",
            style="italic")


def main() -> None:
    d = load_latest()
    m = d["metrics"]
    n_cases = d["n_cases"]
    n_abstained = d["n_abstained"]
    n_cites = m["citation_grounding"]["n_citations"]
    cite_rate = m["citation_grounding"]["resolved"]["rate"]
    ece = m["calibration"]["ece"]
    prm_acc = d["prm"]["accuracy"]

    apply_rc()
    fig, ax = canvas(15.0, 8.0)
    H = ax.get_ylim()[1]

    # ---------------------------------------------------------------- header
    title_block(ax, MARGIN, H - 3.2,
                "Grounded, individualized oncology treatment reasoning")
    ax.text(MARGIN, H - 8.0,
            "A reasoning model adjudicates population-level evidence against an "
            "individual molecular signal, and returns a",
            fontsize=11.5, color=INK_SECONDARY, va="top")
    ax.text(MARGIN, H - 11.2,
            "recommendation that cites its evidence, states its uncertainty, and "
            "abstains when the evidence is thin.",
            fontsize=11.5, color=INK_SECONDARY, va="top")

    eyebrow(ax, COL_IN_X, H - 15.6, "Inputs")
    eyebrow(ax, COL_AD_X, H - 15.6, "Adjudication")
    eyebrow(ax, COL_OUT_X, H - 15.6, "Output")

    # ---------------------------------------------------------------- inputs
    prior_y = MAIN_Y + CARD_H + CARD_GAP
    signal_y = MAIN_Y
    input_card(ax, prior_y, SLOT_1_BLUE, TINT_BLUE, "Population evidence", "prior",
               "ESMO / NCCN-derived guideline index  ·  CIViC  ·  ClinVar",
               "joined on CAID and Entrez ID, never name strings")
    input_card(ax, signal_y, SLOT_2_GREEN, TINT_GREEN, "Individual tumor", "signal",
               "molecular profile  ·  histology  ·  clinical timeline",
               f"{n_cases} real, patient-disjoint cases from cBioPortal")

    # ---------------------------------------------------------------- adjudicator
    card(ax, COL_AD_X, MAIN_Y, COL_AD_W, MAIN_H, facecolor=TINT_VIOLET,
         edgecolor=SLOT_7_VIOLET, lw=1.5)
    cx = COL_AD_X + COL_AD_W / 2
    top = MAIN_Y + MAIN_H
    ax.text(cx, top - 3.0, "Reasoning model", fontsize=14.5, fontweight="bold",
            color=INK, va="top", ha="center")
    ax.text(cx, top - 6.8, "the adjudicator", fontsize=10.5, color=INK_SECONDARY,
            va="top", ha="center", style="italic")

    rows = [
        ("retrieval-grounded", "BM25 · dense · hybrid RRF"),
        ("process-supervised", f"PRM held-out {prm_acc:.3f}"),
        ("post-trained", "verifier-guided RFT / DPO"),
    ]
    ry = MAIN_Y + 12.4
    for label, detail in rows:
        ax.text(cx, ry, label, fontsize=10.6, color=INK, va="top", ha="center",
                fontweight="bold")
        ax.text(cx, ry - 2.3, detail, fontsize=8.8, color=INK_MUTED, va="top",
                ha="center")
        ry -= 4.4

    arrow(ax, (COL_IN_X + COL_IN_W + 0.9, prior_y + CARD_H / 2),
          (COL_AD_X - 0.9, prior_y + CARD_H / 2), color=SLOT_1_BLUE, lw=1.8)
    arrow(ax, (COL_IN_X + COL_IN_W + 0.9, signal_y + CARD_H / 2),
          (COL_AD_X - 0.9, signal_y + CARD_H / 2), color=SLOT_2_GREEN, lw=1.8)
    ax.text((COL_IN_X + COL_IN_W + COL_AD_X) / 2, MAIN_Y + MAIN_H / 2, "adjudicate",
            fontsize=9.0, color=INK_MUTED, ha="center", va="center", style="italic")

    # ---------------------------------------------------------------- output
    card(ax, COL_OUT_X, MAIN_Y, COL_OUT_W, MAIN_H, facecolor=TINT_ROSE,
         edgecolor=SLOT_3_ROSE, lw=1.5)
    ax.text(COL_OUT_X + 2.4, top - 3.0, "Auditable recommendation", fontsize=12.5,
            fontweight="bold", color=INK, va="top")

    claims = [
        ("cites its evidence", f"{n_cites} citations, {cite_rate:.0%} resolve"),
        ("states its uncertainty", f"ECE {ece:.3f} — reported, not smoothed"),
        ("abstains when unsure", f"{n_abstained}/{n_cases} cases deferred"),
        ("ranks beyond-guideline options", "when evidence outranks the prior"),
    ]
    cy = MAIN_Y + 15.6
    for label, detail in claims:
        ax.plot([COL_OUT_X + 3.0], [cy - 0.5], marker="o", markersize=4.0,
                color=SLOT_3_ROSE, zorder=5)
        ax.text(COL_OUT_X + 4.6, cy, label, fontsize=10.0, color=INK, va="top")
        ax.text(COL_OUT_X + 4.6, cy - 2.3, detail, fontsize=8.6, color=INK_MUTED,
                va="top")
        cy -= 4.0

    arrow(ax, (COL_AD_X + COL_AD_W + 0.9, MAIN_Y + MAIN_H / 2),
          (COL_OUT_X - 0.9, MAIN_Y + MAIN_H / 2), color=INK_SECONDARY, lw=1.8)

    # ---------------------------------------------------------------- evaluation band
    card(ax, MARGIN, BAND_Y, 100 - 2 * MARGIN, BAND_H, facecolor=SURFACE_SUNKEN,
         edgecolor=GRID)
    # neutral accent — the band is structure, not one of the four coloured regions
    accent_bar(ax, MARGIN + 0.9, BAND_Y + 0.9, BAND_H - 1.8, INK_MUTED)
    bx = MARGIN + 2.6
    ax.text(bx, BAND_Y + BAND_H - 1.7, tracked("EVALUATION"), fontsize=8.2,
            color=INK_SECONDARY, va="top", fontweight="bold")
    ax.text(bx, BAND_Y + BAND_H - 3.9,
            "guideline concordance  ·  molecular interpretation  ·  step soundness  ·  "
            "tool reliability  ·  citation grounding  ·  calibration  ·  deferral  ·  "
            "information gathering",
            fontsize=9.2, color=INK_SECONDARY, va="top")
    ax.text(bx, BAND_Y + 0.9,
            f"Eight metrics, each with a percentile bootstrap 95% CI at n = {n_cases}. "
            "Degenerate-by-construction results are labelled as such.",
            fontsize=8.4, color=INK_MUTED, va="bottom", style="italic")

    arrow(ax, (cx, MAIN_Y - 0.6), (cx, BAND_Y + BAND_H + 0.6), color=INK_MUTED,
          lw=1.3, dashed=True)

    ax.text(MARGIN, FOOT_Y,
            "Proof of concept on public data (cBioPortal · CIViC · ClinVar · ESCAT). "
            "The molecular modality is text-encoded; native multimodal fusion is the "
            "documented roadmap.",
            fontsize=8.4, color=INK_MUTED, va="bottom", style="italic")

    save(fig, "concept_figure")


if __name__ == "__main__":
    main()
