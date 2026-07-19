"""Render the Phase 5/6 headline results as one page: confusion matrix, calibration
reliability diagram, deferral (risk-coverage) curve, bootstrap-CI forest plot,
information-gathering strip plot, and the abstention-gap bar. Reads the most recent
results/*-pipeline.json; writes summary/results_figure.png and .pdf.
"""
import glob
import json
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"
SEQ_BLUE = "#2a78d6"
SEQ_BLUE_LIGHT = "#9ec5f4"
GOOD = "#0ca30c"
WARNING = "#fab219"
SERIOUS = "#ec835a"
CRITICAL = "#d03b3b"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial"],
    "text.color": INK,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK_SECONDARY,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})


def load_latest():
    files = sorted(glob.glob("results/*-pipeline.json"))
    if not files:
        sys.exit("no results/*-pipeline.json found — run scripts/run_pipeline.py first")
    with open(files[-1]) as f:
        return json.load(f), files[-1]


def panel_confusion(ax, prm):
    mat = np.array([[prm["tn"], prm["fp"]], [prm["fn"], prm["tp"]]])
    vmax = mat.max()
    ax.imshow(mat, cmap=_blue_cmap(), vmin=0, vmax=vmax)
    labels = [["TN", "FP"], ["FN", "TP"]]
    for i in range(2):
        for j in range(2):
            frac = mat[i, j] / vmax
            txt_color = "white" if frac > 0.55 else INK
            ax.text(j, i, f"{labels[i][j]}\n{mat[i, j]}", ha="center", va="center",
                     color=txt_color, fontsize=11, fontweight="bold")
    ax.set_xticks([0, 1], ["pred unsound", "pred sound"])
    ax.set_yticks([0, 1], ["true unsound", "true sound"])
    ax.set_title(f"A. PRM held-out confusion matrix (acc {prm['accuracy']:.3f})",
                 fontsize=10, color=INK, loc="left")
    for spine in ax.spines.values():
        spine.set_visible(False)


def _blue_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("seq_blue", [SURFACE, SEQ_BLUE_LIGHT, SEQ_BLUE])


def panel_calibration(ax, calibration):
    ax.plot([0, 1], [0, 1], linestyle="--", color=INK_MUTED, linewidth=1.5,
             label="perfect calibration", zorder=1)
    pts = [b for b in calibration["reliability_curve"] if b.get("n", 0) > 0]
    xs = [b["mean_confidence"] for b in pts]
    ys = [b["accuracy"] for b in pts]
    ns = [b["n"] for b in pts]
    sizes = [40 + 12 * n for n in ns]
    ax.scatter(xs, ys, s=sizes, color=SEQ_BLUE, zorder=3, label="observed (dot size = n)")
    ax.plot(xs, ys, color=SEQ_BLUE, linewidth=1.5, zorder=2)
    for x, y, n in zip(xs, ys, ns):
        offset = (0, 14) if y < 0.5 else (0, -18)
        ax.annotate(f"n={n}", (x, y), textcoords="offset points", xytext=offset,
                     fontsize=8, color=INK_SECONDARY, ha="center")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.08, 1.08)
    ax.set_xlabel("mean confidence")
    ax.set_ylabel("accuracy")
    ax.set_title(f"B. Calibration — ECE {calibration['ece']:.3f}, "
                 f"Brier {calibration['brier']['value']:.3f}",
                 fontsize=10, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def panel_deferral(ax, deferral):
    pts = deferral["points"]
    # several thresholds can land on the same (coverage, accuracy) point — merge their labels
    grouped = {}
    for p in pts:
        key = (round(p["coverage"], 4), round(p["accuracy"], 4))
        grouped.setdefault(key, {"lo": p["accuracy_ci95"][0], "hi": p["accuracy_ci95"][1],
                                  "thr": []})
        grouped[key]["thr"].append(p["threshold"])
    keys = sorted(grouped, key=lambda k: -k[0])
    cov = [k[0] for k in keys]
    acc = [k[1] for k in keys]
    lo = [grouped[k]["lo"] for k in keys]
    hi = [grouped[k]["hi"] for k in keys]
    ax.fill_between(cov, lo, hi, color=SEQ_BLUE_LIGHT, alpha=0.5, zorder=1)
    ax.plot(cov, acc, color=SEQ_BLUE, linewidth=2, marker="o", zorder=3)
    for i, k in enumerate(keys):
        label = "τ=" + ",".join(str(t) for t in grouped[k]["thr"])
        offset = (10, -4) if i % 2 == 0 else (10, 10)
        ax.annotate(label, k, textcoords="offset points", xytext=offset,
                     fontsize=8, color=INK_SECONDARY)
    ax.set_xlim(0, 1.15)
    ax.set_ylim(0.85, 1.03)
    ax.set_xlabel("coverage (fraction answered)")
    ax.set_ylabel("accuracy on answered cases")
    ax.set_title("C. Deferral: accuracy vs coverage as abstain\nthreshold τ moves (monotone, 95% CI band)",
                 fontsize=10, color=INK, loc="left")
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def panel_forest(ax, metrics):
    rows = [
        ("Guideline concordance (top-1)\ncircular gold — least meaningful",
         metrics["guideline_concordance"]["top1"]["rate"],
         metrics["guideline_concordance"]["top1"]["ci95"], WARNING),
        ("Molecular interpretation agreement\n(post abstain-threshold fix)",
         metrics["molecular_interpretation_accuracy"]["agreement"]["rate"],
         metrics["molecular_interpretation_accuracy"]["agreement"]["ci95"], WARNING),
        ("Step soundness\ndegenerate-by-construction",
         metrics["reasoning_step_accuracy"]["step_soundness"]["rate"],
         metrics["reasoning_step_accuracy"]["step_soundness"]["ci95"], WARNING),
        ("Citation grounding\nstructural check only",
         metrics["citation_grounding"]["resolved"]["rate"],
         metrics["citation_grounding"]["resolved"]["ci95"], WARNING),
        ("Tool call success rate", metrics["tool_use_reliability"]["success"]["rate"],
         metrics["tool_use_reliability"]["success"]["ci95"], GOOD),
    ]
    ys = np.arange(len(rows))[::-1]
    for y, (label, rate, ci, color) in zip(ys, rows):
        ax.plot(ci, [y, y], color=INK_MUTED, linewidth=1.5, zorder=1)
        ax.scatter([rate], [y], s=90, color=color, zorder=3, edgecolor="white", linewidth=0.8)
        ax.text(1.03, y, f"{rate:.2f}", va="center", fontsize=9, color=INK)
    ax.set_yticks(ys, [r[0] for r in rows], fontsize=8.5)
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("rate (95% bootstrap CI)")
    ax.set_title("D. Headline metrics with bootstrap 95% CIs\n"
                 "(color = read: green good · amber caveated · red concerning)",
                 fontsize=10, color=INK, loc="left")
    ax.axvline(1.0, color=GRID, linewidth=1, zorder=0)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def panel_information_gathering(ax, info):
    rng = np.random.default_rng(17)
    correct_vals = np.array(info["evidence_when_correct"])
    incorrect_vals = np.array(info["evidence_when_incorrect"])
    for x0, vals, color, label in [(0, correct_vals, GOOD, "correct"),
                                     (1, incorrect_vals, CRITICAL, "incorrect")]:
        jitter = rng.uniform(-0.12, 0.12, len(vals))
        ax.scatter(x0 + jitter, vals, s=22, color=color, alpha=0.65, zorder=2, label=label)
        ax.hlines(np.mean(vals), x0 - 0.2, x0 + 0.2, color=INK, linewidth=2, zorder=3)
    ax.set_xticks([0, 1], ["correct", "incorrect"])
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylabel("evidence items retrieved")
    ax.set_title(f"E. Information gathering — r = {info['pointbiserial_r']:.3f} "
                 f"(p = {info['p_value']:.3f})\nmore evidence tracks with harder, "
                 f"not better-informed, cases",
                 fontsize=10, color=INK, loc="left")
    ax.grid(True, axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def panel_abstention(ax, mia):
    # rows: ground-truth actionable / not; cols: system abstained / recommended
    mat = np.array([[mia["abstained_when_actionable"], mia["recommended_when_actionable"]],
                     [mia["abstained_when_not_actionable"], mia["recommended_when_not_actionable"]]])
    vmax = mat.max()
    ax.imshow(mat, cmap=_blue_cmap(), vmin=0, vmax=vmax)
    labels = [["miss", "correct"], ["correct", "residual gap"]]
    for i in range(2):
        for j in range(2):
            frac = mat[i, j] / vmax
            txt_color = "white" if frac > 0.55 else INK
            ax.text(j, i, f"{labels[i][j]}\n{mat[i, j]}", ha="center", va="center",
                     color=txt_color, fontsize=10, fontweight="bold")
    ax.set_xticks([0, 1], ["abstained", "recommended"])
    ax.set_yticks([0, 1], ["ground truth:\nactionable", "ground truth:\nnot actionable"])
    ax.set_title("F. Abstention decision, post-fix\n"
                  "(14/38 residual: guideline tier alone still clears\nthe threshold — a gold-standard definition gap)",
                  fontsize=9.5, color=INK, loc="left")
    for spine in ax.spines.values():
        spine.set_visible(False)


def main():
    data, path = load_latest()
    metrics = data["metrics"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    panel_confusion(axes[0, 0], data["prm"])
    panel_calibration(axes[0, 1], metrics["calibration"])
    panel_deferral(axes[0, 2], metrics["deferral_curve"])
    panel_forest(axes[1, 0], metrics)
    panel_information_gathering(axes[1, 1], metrics["information_gathering"])
    panel_abstention(axes[1, 2], metrics["molecular_interpretation_accuracy"])
    fig.suptitle("oncoreason — Phase 5/6 headline results  ·  "
                  f"source: {path}", fontsize=11, color=INK_SECONDARY, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.subplots_adjust(hspace=0.55, wspace=0.35)
    fig.savefig("summary/results_figure.png", dpi=200)
    fig.savefig("summary/results_figure.pdf")
    print("wrote summary/results_figure.png and summary/results_figure.pdf")


if __name__ == "__main__":
    main()
