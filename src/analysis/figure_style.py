"""Small publication-style layer adapted from figures4papers conventions."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PALETTE = {
    "blue_main": "#0F4D92",
    "blue_secondary": "#3775BA",
    "green": "#8BCF8B",
    "red": "#B64342",
    "red_light": "#E9A6A1",
    "neutral": "#9B9B9B",
    "dark": "#272727",
    "teal": "#42949E",
    "violet": "#9A4D8E",
}


def apply_publication_style(font_size: int = 12, axes_linewidth: float = 1.5) -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial Unicode MS", "PingFang SC", "Noto Sans CJK SC", "Arial", "Helvetica", "DejaVu Sans"],
        "font.size": font_size,
        "axes.linewidth": axes_linewidth,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    })


def finalize_figure(fig, output_stem: Path, formats: list[str] | tuple[str, ...] = ("png", "pdf"), dpi: int = 300) -> list[Path]:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    # A figure using constrained_layout (notably multi-axis colorbars) cannot
    # be switched to tight_layout after the colorbar has been created.
    if not fig.get_constrained_layout():
        fig.tight_layout(pad=1.4)
    saved = []
    for fmt in formats:
        path = output_stem.with_suffix(f".{fmt}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.06)
        saved.append(path)
    plt.close(fig)
    return saved
