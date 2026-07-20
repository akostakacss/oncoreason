#!/usr/bin/env python3
"""Publication-grade figures for the GPU stages: what was produced, and what went wrong.

Reads the saved stage artifacts so the numbers on the figures cannot drift from the results:
  results/gpu_stage/stageB_sampling.json   + the relabelled candidates
  results/gpu_stage/stageC_prm_transfer.json

Palette: the project's validated categorical slots (blue / rose / violet). Verified with the
dataviz validator at surface #fcfcfb — worst adjacent CVD separation ΔE 13.4, all checks pass.
Failure magnitudes are carried by direct labels and a chance reference line, not by colour
alone, so the figures survive greyscale and CVD.
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402
from matplotlib.patches import FancyBboxPatch                     # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))

from figure_style import (GRID, INK, INK_MUTED, INK_SECONDARY,     # noqa: E402
                          SLOT_1_BLUE, SLOT_3_ROSE, SLOT_7_VIOLET,
                          SURFACE, SURFACE_SUNKEN, TINT_BLUE, apply_rc, eyebrow, save,
                          tracked)

GPU = os.path.join(REPO, "results", "gpu_stage")


# ---- marks -------------------------------------------------------------------------
def hbar(ax, y, width, color, *, height=0.52, radius=0.06):
    """A horizontal bar with rounded data-end, anchored at x=0."""
    if width <= 0:
        return
    ax.add_patch(FancyBboxPatch(
        (0, y - height / 2), width, height,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=color, edgecolor="none", zorder=3, mutation_aspect=0.55))


def vbar(ax, x, height_, color, *, width=0.46, radius=0.012):
    if height_ <= 0:
        return
    ax.add_patch(FancyBboxPatch(
        (x - width / 2, 0), width, height_,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=color, edgecolor="none", zorder=3, mutation_aspect=8))


def recessive(ax, *, xgrid=False, ygrid=False):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
        ax.spines[s].set_linewidth(1.0)
    ax.tick_params(colors=INK_MUTED, labelsize=9.5, length=0)
    if xgrid:
        ax.xaxis.grid(True, color=GRID, lw=0.9, zorder=0)
    if ygrid:
        ax.yaxis.grid(True, color=GRID, lw=0.9, zorder=0)
    ax.set_axisbelow(True)


def stat_tile(fig, x, y, w, h, value, label, note=None, accent=SLOT_1_BLUE):
    """A hero number — the right form when a single figure is the whole message."""
    ax = fig.add_axes((x, y, w, h)); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=0.06",
                                facecolor=SURFACE_SUNKEN, edgecolor=GRID, lw=1.0))
    ax.add_patch(FancyBboxPatch((0.035, 0.14), 0.018, 0.72,
                                boxstyle="round,pad=0,rounding_size=0.008",
                                facecolor=accent, edgecolor="none"))
    ax.text(0.11, 0.60, value, fontsize=25, fontweight="bold", color=INK, va="center")
    ax.text(0.11, 0.29, label, fontsize=10, color=INK_SECONDARY, va="center")
    if note:
        ax.text(0.11, 0.13, note, fontsize=8.4, color=INK_MUTED, va="center", style="italic")


# ---- stage B -----------------------------------------------------------------------
def stage_b() -> None:
    rep = json.load(open(os.path.join(GPU, "stageB_sampling.json")))
    cand = os.path.join(REPO, "runs", "20260720-0912-b-n8-t08-artifacts",
                        "candidates-relabelled.jsonl")
    rows = [json.loads(l) for l in open(cand)]
    steps = [s for r in rows for s in r["steps"]]
    uncited = sum(1 for s in steps if not s["cited"] and not s["invented"])
    invented = sum(1 for s in steps if s["invented"])
    off_case = sum(1 for s in steps if s["off_case"])

    fig = plt.figure(figsize=(16, 10))
    fig.text(0.045, 0.968, "Stage B — policy sampling on a Kaggle T4",
             fontsize=22, fontweight="bold", color=INK, va="top")
    fig.text(0.045, 0.922,
             "400 traces sampled from Qwen2.5-3B-Instruct across 50 cases, to replace "
             "constructed negatives with observed ones",
             fontsize=11.5, color=INK_SECONDARY, va="top")
    fig.text(0.045, 0.893, "N = 8 per case · temperature 0.8 · seed 17 · 25.2 min · "
             "run 20260720-0912-b-n8-t08",
             fontsize=9, color=INK_MUTED, va="top", style="italic")

    # what was produced
    fig.text(0.045, 0.852, tracked("WHAT WAS PRODUCED"), fontsize=8.5,
             color=INK_MUTED, fontweight="bold", va="top")
    tiles = [("400", "candidate traces", "50 cases × 8 samples", SLOT_1_BLUE),
             ("100%", "parse rate", "400/400, no samples lost", SLOT_1_BLUE),
             ("0.967", "within-case spread", "Best-of-N now has signal", SLOT_1_BLUE),
             ("1076", "labelled steps", "mean 2.69 per sample", SLOT_1_BLUE)]
    for i, (v, lab, note, acc) in enumerate(tiles):
        stat_tile(fig, 0.045 + i * 0.2325, 0.700, 0.2075, 0.122, v, lab, note, acc)

    fig.text(0.045, 0.660, tracked("WHAT WENT WRONG"), fontsize=8.5,
             color=INK_MUTED, fontweight="bold", va="top")

    # -- left: composition of the negative class ------------------------------------
    # generous left inset: the category names are the y tick labels and must not clip
    ax = fig.add_axes((0.150, 0.330, 0.290, 0.265))
    recessive(ax, xgrid=True)
    labels = ["cited nothing", "invented an id", "cited another\ncase's evidence"]
    vals = [uncited, invented, off_case]
    cols = [SLOT_1_BLUE, SLOT_7_VIOLET, SLOT_3_ROSE]
    for i, (v, c) in enumerate(zip(vals, cols)):
        hbar(ax, len(vals) - 1 - i, v, c, height=0.42)
        ax.text(v + 14 if v else 14, len(vals) - 1 - i, str(v), va="center",
                fontsize=11.5, fontweight="bold", color=INK)
    ax.set_yticks(range(len(vals))); ax.set_yticklabels(labels[::-1], fontsize=10.5)
    ax.set_xlim(0, 585); ax.set_ylim(-0.65, 2.65)
    ax.set_xlabel("unsound steps", fontsize=9.5, color=INK_MUTED, labelpad=6)
    ax.set_title("The negative class is 95% one easy failure",
                 fontsize=12.5, fontweight="bold", color=INK, loc="left", pad=12)

    # -- right: the labeller over-counted -------------------------------------------
    ax2 = fig.add_axes((0.590, 0.330, 0.360, 0.265))
    recessive(ax2, ygrid=True)
    groups = ["steps with\ninvented ids", "distinct\ninvented strings"]
    before, after = [61, 42], [26, 16]
    for i in range(2):
        vbar(ax2, i - 0.135, before[i], "#c9c8c1", width=0.22)
        vbar(ax2, i + 0.135, after[i], SLOT_7_VIOLET, width=0.22)
        ax2.text(i - 0.135, before[i] + 2.2, str(before[i]), ha="center", fontsize=10.5,
                 color=INK_SECONDARY)
        ax2.text(i + 0.135, after[i] + 2.2, str(after[i]), ha="center", fontsize=10.5,
                 fontweight="bold", color=INK)
    ax2.set_xticks(range(2)); ax2.set_xticklabels(groups, fontsize=10.5)
    ax2.set_xlim(-0.5, 1.5); ax2.set_ylim(0, 78)
    ax2.set_ylabel("count", fontsize=9.5, color=INK_MUTED)
    ax2.set_title("My labeller over-counted hallucinations 2.3×",
                  fontsize=12.5, fontweight="bold", color=INK, loc="left", pad=12)
    # legend: two series, so identity is never carried by colour alone
    for yy, col, txt, ink in ((70.5, "#c9c8c1", "as labelled in-notebook", INK_SECONDARY),
                              (63.0, SLOT_7_VIOLET, "after relabelling", INK)):
        ax2.add_patch(FancyBboxPatch((0.62, yy), 0.075, 4.0,
                                     boxstyle="round,pad=0,rounding_size=0.02",
                                     facecolor=col, edgecolor="none"))
        ax2.text(0.72, yy + 2.0, txt, fontsize=9.2, color=ink, va="center")

    # -- captions, placed in figure space so they cannot collide with the axes -------
    fig.text(0.150, 0.268,
             "481 of 507 unsound steps simply carry no citation — very nearly the signal\n"
             "the `strip` counterfactual already gave. Sampling made the negatives real,\n"
             "but it did not make them hard.",
             fontsize=9.4, color=INK_MUTED, va="top", linespacing=1.5)
    fig.text(0.590, 0.268,
             "Exact string matching scored `gl-ret` as invented (real id, prefix dropped) and\n"
             "split only on commas, so `civic:A and civic:B` counted as one fabrication.\n"
             "18 prose fragments were reclassified as uncited rather than hallucinated.",
             fontsize=9.4, color=INK_MUTED, va="top", linespacing=1.5)

    # the zero finding, called out rather than left as an empty bar
    ax4 = fig.add_axes((0.045, 0.045, 0.905, 0.098)); ax4.axis("off")
    ax4.set_xlim(0, 1); ax4.set_ylim(0, 1)
    ax4.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=0.04",
                                 facecolor=TINT_BLUE, edgecolor=GRID, lw=1.0))
    ax4.text(0.014, 0.74, "Zero off-case citations in 1076 steps.", fontsize=10.4,
             fontweight="bold", color=INK, va="center")
    ax4.text(0.014, 0.44,
             "The `swap` strategy in mine_negatives — modelled on the TP53 tumor-type "
             "mismatch — tests a failure mode this policy never exhibits, because it only "
             "ever sees",
             fontsize=10.0, color=INK, va="center")
    ax4.text(0.014, 0.17,
             "one case's evidence. Half the synthetic training signal describes nothing real.",
             fontsize=10.0, color=INK, va="center")
    save(fig, os.path.join(GPU, "stageB_figure"))


# ---- stage C -----------------------------------------------------------------------
def stage_c() -> None:
    d = json.load(open(os.path.join(GPU, "stageC_prm_transfer.json")))
    arms = [("A · trained on synthetic\ntested on synthetic", "control", "arm_a_on_synthetic_holdout", SLOT_1_BLUE),
            ("A · trained on synthetic\ntested on REAL", "the question", "arm_a_synthetic_to_real", SLOT_3_ROSE),
            ("B · trained on real\ntested on REAL", "ceiling", "arm_b_real_to_real", SLOT_7_VIOLET)]

    fig = plt.figure(figsize=(15, 8.5))
    fig.text(0.05, 0.955, "Stage C — the verifier does not transfer to real hallucinations",
             fontsize=22, fontweight="bold", color=INK, va="top")
    fig.text(0.05, 0.902,
             "A process reward model trained on constructed negatives, tested on the policy "
             "output it would actually have to judge",
             fontsize=11.5, color=INK_SECONDARY, va="top")
    fig.text(0.05, 0.866, "Same case-level split for every arm (15 of 50 cases held out, 323 test "
             "steps) · TF-IDF backend held constant · seed 17",
             fontsize=9, color=INK_MUTED, va="top", style="italic")

    for panel, (key, title, floor) in enumerate([
            ("balanced_accuracy", "Balanced accuracy", 0.5),
            ("recall_unsound", "Recall on unsound steps — the decisive number", None)]):
        ax = fig.add_axes((0.05 + panel * 0.495, 0.245, 0.40, 0.515))
        recessive(ax, ygrid=True)
        for i, (lab, _, akey, col) in enumerate(arms):
            v = d[akey][key]
            vbar(ax, i, v, col, width=0.34, radius=0.004)
            ax.text(i, v + 0.028, f"{v:.3f}", ha="center", fontsize=13,
                    fontweight="bold", color=INK)
        if floor is not None:
            ax.axhline(floor, color=INK_MUTED, lw=1.4, ls=(0, (5, 4)), zorder=4)
            ax.text(-0.54, floor + 0.020, "chance", fontsize=9.5, color=INK_MUTED,
                    ha="left", style="italic")
        ax.set_xticks(range(3))
        ax.set_xticklabels([a[0] for a in arms], fontsize=9.8)
        ax.set_ylim(0, 1.09); ax.set_xlim(-0.6, 2.6)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_title(title, fontsize=12.5, fontweight="bold", color=INK, loc="left", pad=14)

    # the reading, stated rather than left to the reader
    ax3 = fig.add_axes((0.05, 0.030, 0.895, 0.118)); ax3.axis("off")
    ax3.set_xlim(0, 1); ax3.set_ylim(0, 1)
    ax3.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=0.03",
                                 facecolor=TINT_BLUE, edgecolor=GRID, lw=1.0))
    ax3.text(0.016, 0.76,
             "The control arm reproduces Phase 5's published 0.914 exactly, so the setup is "
             "faithful before anything is concluded from the other two arms.",
             fontsize=10.2, color=INK, va="center")
    ax3.text(0.016, 0.42,
             "The verifier catches 83.8% of constructed negatives and only 11.4% of real ones. "
             "Its honest figure on policy output is 0.53 balanced accuracy, not 0.914 —",
             fontsize=10.2, color=INK, va="center")
    ax3.text(0.016, 0.15,
             "it learned the shape of the counterfactuals rather than the concept of "
             "groundedness. Arm B is not a good verifier either: 95% of real negatives are "
             "simply uncited.",
             fontsize=10.2, color=INK, va="center")
    save(fig, os.path.join(GPU, "stageC_figure"))


# ---- stage D -----------------------------------------------------------------------
def stage_d() -> None:
    ids = json.load(open(os.path.join(GPU, "stageC_ids_only.json")))
    txt = json.load(open(os.path.join(GPU, "stageC_with_text.json")))
    arms = [("A · synthetic\ntested on synthetic", "arm_a_on_synthetic_holdout"),
            ("A · synthetic\ntested on REAL", "arm_a_synthetic_to_real"),
            ("B · real\ntested on REAL", "arm_b_real_to_real")]

    fig = plt.figure(figsize=(14, 8.5))
    fig.text(0.055, 0.965, "Stage D — inlining evidence text made the verifier worse",
             fontsize=22, fontweight="bold", color=INK, va="top")
    fig.text(0.055, 0.915,
             "Stage C predicted that showing the model what each cited record says would raise "
             "the ceiling. It did not.",
             fontsize=11.5, color=INK_SECONDARY, va="top")
    fig.text(0.055, 0.879, "Same split, same arms, same TF-IDF backend — the feature string is "
             "the only variable · evidence text resolved for 570 of 1076 steps · seed 17",
             fontsize=9, color=INK_MUTED, va="top", style="italic")

    ax = fig.add_axes((0.075, 0.30, 0.545, 0.50))
    recessive(ax, ygrid=True)
    for i, (lab, key) in enumerate(arms):
        a, b = ids[key]["balanced_accuracy"], txt[key]["balanced_accuracy"]
        vbar(ax, i - 0.155, a, SLOT_1_BLUE, width=0.26, radius=0.004)
        vbar(ax, i + 0.155, b, SLOT_3_ROSE, width=0.26, radius=0.004)
        ax.text(i - 0.155, a + 0.025, f"{a:.3f}", ha="center", fontsize=10.5, color=INK)
        ax.text(i + 0.155, b + 0.025, f"{b:.3f}", ha="center", fontsize=10.5,
                fontweight="bold", color=INK)
        ax.text(i, -0.115, f"{b - a:+.3f}", ha="center", fontsize=10.5,
                fontweight="bold", color=SLOT_3_ROSE)
    ax.axhline(0.5, color=INK_MUTED, lw=1.4, ls=(0, (5, 4)), zorder=4)
    ax.text(-0.56, 0.518, "chance", fontsize=9.5, color=INK_MUTED, ha="left", style="italic")
    ax.set_xticks(range(3)); ax.set_xticklabels([a[0] for a in arms], fontsize=9.8)
    ax.set_ylim(0, 1.06); ax.set_xlim(-0.6, 2.6)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel("balanced accuracy", fontsize=9.5, color=INK_MUTED)
    ax.set_title("Every arm falls; the transfer arm falls below chance",
                 fontsize=12.5, fontweight="bold", color=INK, loc="left", pad=14)
    # legend sits over the middle arm, the only column with clear headroom
    for yy, col, t in ((0.965, SLOT_1_BLUE, "step + ids"),
                       (0.885, SLOT_3_ROSE, "step + ids + evidence text")):
        ax.add_patch(FancyBboxPatch((0.58, yy), 0.10, 0.042,
                                    boxstyle="round,pad=0,rounding_size=0.01",
                                    facecolor=col, edgecolor="none"))
        ax.text(0.72, yy + 0.021, t, fontsize=9.4, color=INK, va="center")

    ax2 = fig.add_axes((0.665, 0.30, 0.275, 0.50)); ax2.axis("off")
    ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
    ax2.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=0.035",
                                 facecolor=SURFACE_SUNKEN, edgecolor=GRID, lw=1.0))
    ax2.text(0.07, 0.93, tracked("WHY"), fontsize=8.5, color=INK_MUTED,
             fontweight="bold", va="top")
    body = [("Judging support is relational.",
             "TF-IDF sees the union of the claim's\nwords and the evidence's words, never\n"
             "whether they correspond."),
            ("The text barely varies.",
             "CIViC summaries are formulaic, so a\ncitation swapped from another case reads\n"
             "almost identically to the right one."),
            ("So it dilutes what worked.",
             "Adding shared vocabulary to every cited\nstep pushes sound and unsound closer\n"
             "together in feature space.")]
    y = 0.80
    for head, sub in body:
        ax2.text(0.07, y, head, fontsize=10, fontweight="bold", color=INK, va="top")
        ax2.text(0.07, y - 0.055, sub, fontsize=9.1, color=INK_SECONDARY, va="top",
                 linespacing=1.5)
        y -= 0.265

    ax3 = fig.add_axes((0.055, 0.045, 0.885, 0.175)); ax3.axis("off")
    ax3.set_xlim(0, 1); ax3.set_ylim(0, 1)
    ax3.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=0.025",
                                 facecolor=TINT_BLUE, edgecolor=GRID, lw=1.0))
    ax3.text(0.014, 0.80, "The intervention is necessary but not sufficient.",
             fontsize=10.6, fontweight="bold", color=INK, va="center")
    ax3.text(0.014, 0.53,
             "The information really is required to judge support — it simply cannot be "
             "exploited by a model that cannot attend across the claim/evidence boundary.",
             fontsize=10.1, color=INK, va="center")
    ax3.text(0.014, 0.24,
             "This sharpens the ModernBERT test rather than removing it: hold the feature "
             "string fixed and vary only the model class. That is now a single-variable "
             "experiment.",
             fontsize=10.1, color=INK, va="center")
    save(fig, os.path.join(GPU, "stageD_figure"))


if __name__ == "__main__":
    apply_rc()
    stage_b()
    stage_c()
    stage_d()
