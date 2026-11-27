"""README banner and social preview for LLaMEA, in the BLADE visual family.

Same canvas, typography and layout as the BLADE banner
(XAI-liacs/BLADE, docs/media/banner/make_banner_v2.py, style "g"), with
LLaMEA's teal/mint accent instead of BLADE's violet/pink. The mark is a
double helix whose rungs are a population of candidate algorithms coloured
by fitness, and the graphic on the right is an evolution run: offspring per
generation, with the best-so-far lineage highlighted.

Run from the repository root (needs matplotlib and Pillow):

    uv run --with matplotlib python docs/media/banner/make_banner.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

OUT = Path(__file__).resolve().parents[1]
FONT = (
    "Inter"
    if any("Inter" in f.name for f in font_manager.fontManager.ttflist)
    else "DejaVu Sans"
)

TITLE = "LLaMEA"
TAGLINE = "Large Language Model Evolutionary Algorithm"
SUBLINE = "LLM-driven automated algorithm design"

# BLADE palette (shared) + LLaMEA accents.
THEMES = {
    "dark": dict(bg="#0b0d12", text="#ffffff", sub="#8ee6c0", faint="#2a3140"),
    "light": dict(bg="#f7f8fa", text="#0b2a3a", sub="#13806f", faint="#d5dbe3"),
}
FITNESS = LinearSegmentedColormap.from_list(
    "llamea", ["#1b4f72", "#1f8a9a", "#3cc7a6", "#8ee6c0", "#ffb547"]
)


def helix(ax, x0, y0, height, width, theme):
    """Double helix mark; the rungs are candidates coloured by fitness."""
    t = np.linspace(0, 2.6 * np.pi, 400)
    y = y0 + (t / t.max() - 0.5) * height
    for phase, alpha in ((0.0, 1.0), (np.pi, 0.55)):
        x = x0 + np.sin(t + phase) * width / 2
        ax.plot(x, y, color=theme["text"], lw=7, alpha=alpha, solid_capstyle="round")
    rng = np.random.default_rng(3)
    for k, tk in enumerate(np.linspace(0.35, t.max() - 0.35, 11)):
        yk = y0 + (tk / t.max() - 0.5) * height
        xa = x0 + np.sin(tk) * width / 2
        xb = x0 + np.sin(tk + np.pi) * width / 2
        if abs(xa - xb) < width * 0.18:
            continue
        ax.plot([xa, xb], [yk, yk], color=theme["faint"], lw=2.2, zorder=1)
        c = FITNESS(min(1.0, k / 10 + rng.uniform(-0.05, 0.05)))
        ax.scatter([(xa + xb) / 2], [yk], s=120, color=c, zorder=3, lw=0)


def evolution(ax, x0, x1, y0, y1, theme):
    """Generations left to right, fitness upwards: offspring vary around
    their parent and the best-so-far lineage climbs (highlighted)."""
    rng = np.random.default_rng(11)
    gens, per_gen = 8, 5
    xs = np.linspace(x0, x1, gens)
    span = y1 - y0
    parents = [y0 + 0.15 * span]
    lineage = [(xs[0], parents[0])]
    ax.scatter([xs[0]], parents, s=110, color=FITNESS(0.05), lw=0, zorder=2)
    for g in range(1, gens):
        children, origins = [], []
        for _ in range(per_gen):
            parent = parents[rng.integers(len(parents))]
            child = np.clip(parent + rng.normal(0.06, 0.12) * span, y0, y1)
            children.append(child)
            origins.append(parent)
        children, origins = np.array(children), np.array(origins)
        for c, o in zip(children, origins):
            ax.plot([xs[g - 1], xs[g]], [o, c], color=theme["faint"], lw=1.2, zorder=1)
        fit = (children - y0) / span
        ax.scatter(np.full(per_gen, xs[g]), children, s=60 + 110 * fit,
                   c=FITNESS(fit), lw=0, zorder=2)
        best = max(lineage[-1][1], children.max())
        lineage.append((xs[g], best))
        # elitist survivors: the best few
        parents = list(np.sort(np.append(children, lineage[-1][1]))[-3:])
    for (xa, ya), (xb, yb) in zip(lineage, lineage[1:]):
        f = ((ya + yb) / 2 - y0) / span
        ax.plot([xa, xb], [ya, yb], color=FITNESS(f), lw=3.4, zorder=3,
                solid_capstyle="round")
    xb, yb = lineage[-1]
    ax.scatter([xb], [yb], s=380, color="#ffb547", zorder=4, lw=0)


def render(theme_name, size, path, social=False):
    theme = THEMES[theme_name]
    w, h = size
    fig = plt.figure(figsize=(w / 200, h / 200), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    fig.patch.set_facecolor(theme["bg"])

    if not social:
        helix(ax, 0.105 * w, 0.5 * h, 0.62 * h, 0.085 * w, theme)
        tx = 0.2 * w
        ax.text(tx, 0.60 * h, TITLE, fontsize=h / 11, fontweight="bold",
                family=FONT, color=theme["text"], va="center")
        ax.text(tx, 0.39 * h, TAGLINE, fontsize=h / 37, family=FONT,
                color=theme["text"], va="center")
        ax.text(tx, 0.28 * h, SUBLINE, fontsize=h / 37, family=FONT,
                color=theme["sub"], va="center")
        evolution(ax, 0.72 * w, 0.94 * w, 0.12 * h, 0.88 * h, theme)
    else:
        helix(ax, 0.12 * w, 0.5 * h, 0.45 * h, 0.09 * w, theme)
        tx = 0.22 * w
        ax.text(tx, 0.58 * h, TITLE, fontsize=w / 34, fontweight="bold",
                family=FONT, color=theme["text"], va="center")
        ax.text(tx, 0.44 * h, TAGLINE, fontsize=w / 125, family=FONT,
                color=theme["text"], va="center")
        ax.text(tx, 0.36 * h, SUBLINE, fontsize=w / 125, family=FONT,
                color=theme["sub"], va="center")
        evolution(ax, 0.74 * w, 0.92 * w, 0.30 * h, 0.72 * h, theme)
    fig.savefig(path, facecolor=theme["bg"])
    plt.close(fig)
    print("wrote", path)


def render_mark(theme_name, path):
    """Square helix mark for the docs sidebar and favicon (transparent)."""
    theme = THEMES[theme_name]
    fig = plt.figure(figsize=(2.56, 2.56), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 512)
    ax.set_ylim(0, 512)
    ax.axis("off")
    helix(ax, 256, 256, 440, 300, theme)
    fig.savefig(path, transparent=True)
    plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    STATIC = OUT.parent / "_static"
    render_mark("light", STATIC / "llamea-mark.png")
    render_mark("dark", STATIC / "llamea-mark-dark.png")
    render("dark", (3200, 800), OUT / "llamea-banner-dark.png")
    render("light", (3200, 800), OUT / "llamea-banner.png")
    render("dark", (2560, 1280), STATIC / "llamea-social-preview.png", social=True)
