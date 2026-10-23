from __future__ import annotations

import textwrap
import zlib
from collections import Counter
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from PIL import Image

import config
from models import AnalysisResult

MAX_DENSE_CAPTIONS = 3
PALETTE = plt.get_cmap("tab20").colors


def get_color(label: str):
    return PALETTE[zlib.crc32(label.encode("utf-8")) % len(PALETTE)]


def draw_result(
    ax,
    result: AnalysisResult,
    min_confidence: float = config.MIN_OBJECT_CONFIDENCE,
    font_size: int = 9,
    title_width: int = 40,
) -> None:
    ax.imshow(Image.open(result.image_path).convert("RGB"))

    for obj in result.objects:
        if obj.confidence < min_confidence:
            continue

        color = get_color(obj.name)
        ax.add_patch(
            patches.Rectangle(
                (obj.x, obj.y), obj.width, obj.height,
                linewidth=2, edgecolor=color, facecolor="none",
            )
        )
        ax.text(
            obj.x + 3, obj.y + 3,
            f"{obj.name} {obj.confidence:.0%}",
            color="black", fontsize=font_size, va="top", ha="left",
            bbox=dict(facecolor=color, alpha=0.85, pad=1.5, edgecolor="none"),
        )

    caption = textwrap.fill(result.caption, width=title_width)
    tags = textwrap.fill("Tags: " + ", ".join(result.tags[:8]), width=title_width)
    details = "".join(
        "\n" + textwrap.fill(f"- {text}", width=title_width, subsequent_indent="  ")
        for text in result.dense_captions[:MAX_DENSE_CAPTIONS]
    )
    ax.set_xlabel(
        f"{caption} ({result.caption_confidence:.0%}){details}\n{tags}",
        fontsize=font_size,
        loc="left",
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _finish(fig, save_path: Path | None, show: bool) -> None:
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Tersimpan: {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def show_single(
    result: AnalysisResult,
    save_path: Path | None = None,
    min_confidence: float = config.MIN_OBJECT_CONFIDENCE,
    show: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    draw_result(ax, result, min_confidence, font_size=11, title_width=70)
    _finish(fig, save_path, show)


def show_object_summary(
    results: list[AnalysisResult],
    save_path: Path | None = None,
    min_confidence: float = config.MIN_OBJECT_CONFIDENCE,
    top_n: int = 10,
    show: bool = False,
) -> None:
    counter = Counter(
        obj.name for result in results for obj in result.objects
        if obj.confidence >= min_confidence
    )
    if not counter:
        print("Tidak ada objek terdeteksi untuk diringkas.")
        return

    labels, values = zip(*counter.most_common(top_n))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(labels, values, color=[get_color(label) for label in labels])
    ax.invert_yaxis()
    ax.set_xlabel("Jumlah deteksi")
    ax.set_title(f"Top {len(labels)} objek terdeteksi (confidence ≥ {min_confidence:.0%})")
    _finish(fig, save_path, show)
