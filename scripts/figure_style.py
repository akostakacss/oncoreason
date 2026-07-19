"""Shared design tokens and primitives for the project's figures.

One place for ink, surface, palette and box/arrow drawing so the concept, roadmap and
results figures read as one system. Colours are the validated categorical slots in fixed
order (blue, green, ...); status colours are reserved for state and never reused as a series.

Used by make_concept_figure.py and make_roadmap_figure.py.
"""
from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

matplotlib.use("Agg")

# ---- ink & surface ---------------------------------------------------------------
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#6b6a65"        # >= 4.5:1 on every tint below (AA small text)
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"
SURFACE_SUNKEN = "#f4f3ef"

# ---- categorical slots (fixed order — never cycled) -------------------------------
SLOT_1_BLUE = "#2a78d6"
SLOT_2_GREEN = "#008300"
SLOT_3_ROSE = "#d55181"      # stepped darker than #e87ba4 to clear 3:1 on the surface
SLOT_7_VIOLET = "#4a3aa7"

# ---- pastel tints -----------------------------------------------------------------
# Card fills. Each is a desaturated tint of the accent it pairs with, so fill and
# accent carry the same identity. Verified >= 4.5:1 against INK_MUTED, so every text
# token stays AA-legible on top of them.
TINT_BLUE = "#eaf2fd"
TINT_GREEN = "#e8f4e8"
TINT_VIOLET = "#eeecf9"
TINT_ROSE = "#fceef3"

# ---- status (reserved: state only) ------------------------------------------------
GOOD = "#0ca30c"
WARNING = "#eda100"
CRITICAL = "#d03b3b"

FONT = ["DejaVu Sans", "Arial", "Helvetica"]


def apply_rc() -> None:
    """Set the shared rcParams. Call once per figure script."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": FONT,
        "text.color": INK,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "pdf.fonttype": 42,      # embed as TrueType so text stays selectable
        "ps.fonttype": 42,
    })


def canvas(width: float, height: float, units: float = 100.0):
    """A single full-bleed axes in 0..units coordinates, y up. Returns (fig, ax)."""
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, units)
    ax.set_ylim(0, units * height / width)
    ax.axis("off")
    return fig, ax


def card(ax, x, y, w, h, *, facecolor=SURFACE, edgecolor=GRID, lw=1.2,
         radius=1.1, dashed=False, zorder=2, alpha=1.0):
    """A rounded panel. x, y is the bottom-left corner."""
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=facecolor, edgecolor=edgecolor, linewidth=lw,
        linestyle=(0, (4, 3)) if dashed else "solid",
        zorder=zorder, alpha=alpha, mutation_aspect=1,
    )
    ax.add_patch(box)
    return box


def accent_bar(ax, x, y, h, color, *, w=0.55, zorder=3):
    """A thin vertical rule marking a card's left edge — carries the card's identity."""
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0,rounding_size=0.26",
        facecolor=color, edgecolor="none", zorder=zorder,
    ))


def arrow(ax, start, end, *, color=INK_MUTED, lw=1.6, dashed=False,
          head=7.0, zorder=4, shrink=0.0):
    """A flow arrow from start to end (both (x, y) in canvas units)."""
    ax.add_patch(FancyArrowPatch(
        start, end,
        arrowstyle=f"-|>,head_width={head / 22:.3f},head_length={head / 15:.3f}",
        mutation_scale=10, color=color, linewidth=lw,
        linestyle=(0, (4, 3)) if dashed else "solid",
        shrinkA=shrink, shrinkB=shrink, zorder=zorder,
        joinstyle="miter", capstyle="butt",
    ))


def title_block(ax, x, y, title, subtitle=None, note=None, *,
                title_size=20, subtitle_size=11.5, note_size=9):
    """Figure title, subtitle and a muted scope note, top-aligned at (x, y)."""
    ax.text(x, y, title, fontsize=title_size, fontweight="bold",
            color=INK, va="top", ha="left")
    dy = 0
    if subtitle:
        dy += title_size * 0.19
        ax.text(x, y - dy, subtitle, fontsize=subtitle_size,
                color=INK_SECONDARY, va="top", ha="left")
    if note:
        dy += subtitle_size * 0.30
        ax.text(x, y - dy, note, fontsize=note_size, color=INK_MUTED,
                va="top", ha="left", style="italic")


def tracked(text: str) -> str:
    """Letter-spaced text. Matplotlib has no tracking property, so space it manually."""
    return " ".join(text)


def eyebrow(ax, x, y, text, *, color=INK_MUTED, size=8.5):
    """A small uppercase section label."""
    ax.text(x, y, tracked(text.upper()), fontsize=size, color=color, va="center",
            ha="left", fontweight="bold")


def save(fig, stem: str) -> None:
    """Write <stem>.png (screen) and <stem>.pdf (vector) with identical geometry."""
    fig.savefig(f"{stem}.png", dpi=200)
    fig.savefig(f"{stem}.pdf")
    plt.close(fig)
    print(f"wrote {stem}.png and {stem}.pdf")
