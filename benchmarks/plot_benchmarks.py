"""Generate the OmniParseBench and olmOCR-bench benchmark charts in data/images/."""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "images"

# Exact palette matching the repo's benchmark charts
BG_COLOR = "#f9fafd"
MARKER_BLUE = "#001dbe"
MARKER_LIGHT_BLUE = "#396bd3"
MARKER_MID_BLUE = "#1d4ed8"
OTHER_SLATE = "#3a3d44"
DARK_TEXT = "#0f1217"
SUBTITLE_GRAY = "#595e66"
CAPTION_GRAY = "#9a9fa7"
SPINE_GRAY = "#d1d2d6"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "axes.facecolor": BG_COLOR,
        "figure.facecolor": BG_COLOR,
        "savefig.facecolor": BG_COLOR,
        "text.color": DARK_TEXT,
        "axes.labelcolor": "#2c3038",
        "xtick.color": "#2c3038",
        "ytick.color": "#2c3038",
    }
)


def _style_axes(ax):
    ax.set_facecolor(BG_COLOR)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(SPINE_GRAY)
    ax.spines["bottom"].set_color(SPINE_GRAY)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)


def _add_header_footer(fig, title: str, subtitle: str, footer: str):
    fig.text(
        0.055,
        0.925,
        title,
        fontsize=20,
        fontweight="medium",
        color=DARK_TEXT,
        ha="left",
        va="top",
    )
    fig.text(
        0.055,
        0.868,
        subtitle,
        fontsize=11.5,
        color=SUBTITLE_GRAY,
        ha="left",
        va="top",
    )
    fig.text(
        0.055,
        0.018,
        footer,
        fontsize=8.8,
        color=CAPTION_GRAY,
        ha="left",
        va="bottom",
    )


def plot_omniparsebench_scatter():
    """Quality vs. throughput on OmniParseBench (2,937 pages, 16,288 tests)."""
    fig = plt.figure(figsize=(9.6, 6.6), dpi=200)
    ax = fig.add_axes([0.085, 0.13, 0.885, 0.68])
    _style_axes(ax)

    # (label, throughput_pg_s, headline_score, is_marker, dx_pts, dy_pts, ha)
    points = [
        ("MinerU 2.5-Pro", 0.54, 87.1, False, 0, 18, "center"),
        ("DeepSeek-OCR-2", 0.80, 75.2, False, 0, -24, "center"),
        ("PaddleOCR-VL-1.6", 1.25, 78.7, False, 8, 18, "center"),
        ("PP-StructureV3", 0.18, 58.4, False, 0, -24, "center"),
        ("docling", 2.09, 54.0, False, 0, -24, "center"),
        ("Tesseract 5 (text)", 2.92, 17.6, False, 0, -24, "center"),
        ("Marker  vllm", 4.52, 85.9, True, 0, 18, "center"),
        ("Marker  balanced", 16.9, 80.5, True, 0, 18, "center"),
        ("Marker  fast", 23.6, 71.0, True, 0, -24, "center"),
        ("Marker  no_ocr", 57.0, 40.8, True, 0, 18, "center"),
    ]

    for label, x, y, is_marker, dx, dy, ha in points:
        color = MARKER_BLUE if is_marker else OTHER_SLATE
        text_color = MARKER_BLUE if is_marker else DARK_TEXT
        ax.scatter(
            [x],
            [y],
            s=360,
            c=color,
            edgecolors=DARK_TEXT,
            linewidths=1.8,
            zorder=4,
        )
        ax.annotate(
            label,
            (x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            va="center",
            fontsize=11.5,
            fontweight="medium" if is_marker else "normal",
            color=text_color,
            zorder=5,
        )

    ax.set_xscale("log")
    ax.set_xlim(0.11, 115)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.tick_params(axis="both", which="major", labelsize=10.5, length=0, pad=6)
    ax.tick_params(axis="x", which="minor", length=3, color="#666666")
    ax.set_xlabel(
        "throughput — pages / sec  ·  log scale, right is faster",
        fontsize=11.5,
        labelpad=10,
    )
    ax.set_ylabel("OmniParseBench headline score", fontsize=11.5, labelpad=10)

    _add_header_footer(
        fig,
        "OmniParseBench  ·  quality vs. throughput",
        "Headline score (Text, Tables, Layout) across 2,937 pages vs. sustained pages/sec.",
        "16,288 pass/fail tests across 22 suites and 92 languages. Marker fast / no_ocr / PP-StructureV3 / Tesseract run on CPU; VLM & hybrid modes on GPU.",
    )
    fig.savefig(OUT_DIR / "omniparsebench.png", dpi=200)
    plt.close(fig)


def plot_omniparsebench_scores_bar():
    """Ranked bar chart of OmniParseBench Headline scores across representative models."""
    fig = plt.figure(figsize=(9.6, 6.2), dpi=200)
    ax = fig.add_axes([0.085, 0.16, 0.885, 0.65])
    _style_axes(ax)

    items = [
        ("Datalab\nAPI", 94.1, False),
        ("Chandra 2", 90.1, False),
        ("Mistral\nOCR", 89.5, False),
        ("MinerU\n2.5-Pro", 87.1, False),
        ("Marker\nvllm", 85.9, True),
        ("GLM-OCR", 81.5, False),
        ("Marker\nbalanced", 80.5, True),
        ("Nemotron\n3.5", 80.0, False),
        ("Paddle\nVL 1.6", 78.7, False),
        ("DeepSeek\nOCR 2", 75.2, False),
        ("Marker\nfast", 71.0, True),
        ("Hunyuan\nOCR 1.5", 64.6, False),
        ("PP-Struct\nV3", 58.4, False),
        ("Marker\nno_ocr", 40.8, True),
    ]

    xs = np.arange(len(items))
    scores = [it[1] for it in items]
    colors = [MARKER_BLUE if it[2] else OTHER_SLATE for it in items]

    bars = ax.bar(
        xs,
        scores,
        width=0.64,
        color=colors,
        edgecolor=DARK_TEXT,
        linewidth=1.3,
        zorder=3,
    )

    for bar, (label, score, is_marker) in zip(bars, items):
        txt_color = MARKER_BLUE if is_marker else DARK_TEXT
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            score + 1.4,
            f"{score:.1f}",
            ha="center",
            va="bottom",
            fontsize=9.8,
            fontweight="medium" if is_marker else "normal",
            color=txt_color,
        )

    ax.set_xticks(xs)
    ax.set_xticklabels([it[0] for it in items], fontsize=9.2)
    for tick_label, (_, _, is_marker) in zip(ax.get_xticklabels(), items):
        if is_marker:
            tick_label.set_color(MARKER_BLUE)
            tick_label.set_fontweight("medium")

    ax.set_ylim(0, 102)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.tick_params(axis="both", which="major", length=0, pad=6)
    ax.set_ylabel("OmniParseBench headline score", fontsize=11.5, labelpad=10)
    ax.set_xlim(-0.65, len(items) - 0.35)

    _add_header_footer(
        fig,
        "OmniParseBench  ·  headline score",
        "Mean of Text (8,499), Tables (5,386), and Layout (2,403) pass rates across 2,937 pages.",
        "Datalab API & Mistral OCR are hosted APIs; open VLMs on GPU; Marker fast, Marker no_ocr, and PP-StructureV3 run on CPU. Higher is better.",
    )
    fig.savefig(OUT_DIR / "omniparsebench_scores_bar.png", dpi=200)
    plt.close(fig)


def plot_omniparsebench_families():
    """Grouped bar chart of OmniParseBench Text, Tables, and Layout scores."""
    fig = plt.figure(figsize=(9.6, 6.2), dpi=200)
    ax = fig.add_axes([0.085, 0.16, 0.885, 0.65])
    _style_axes(ax)

    # (label, text, tables, layout, is_marker)
    systems = [
        ("Chandra 2", 87.4, 87.4, 95.4, False),
        ("MinerU\n2.5-Pro", 77.4, 92.1, 91.7, False),
        ("Marker\nvllm", 83.8, 78.3, 95.7, True),
        ("GLM-OCR", 71.0, 84.2, 89.3, False),
        ("Marker\nbalanced", 79.2, 70.1, 92.2, True),
        ("Paddle\nVL 1.6", 61.7, 83.7, 90.7, False),
        ("DeepSeek\nOCR 2", 64.2, 70.0, 91.4, False),
        ("Marker\nfast", 62.3, 59.6, 91.2, True),
        ("PP-Struct\nV3", 55.6, 47.8, 71.7, False),
    ]

    xs = np.arange(len(systems))
    w = 0.25

    text_scores = [s[1] for s in systems]
    table_scores = [s[2] for s in systems]
    layout_scores = [s[3] for s in systems]

    ax.bar(
        xs - w,
        text_scores,
        width=w,
        color="#001dbe",
        edgecolor=DARK_TEXT,
        linewidth=1.0,
        label="Text (8,499 tests)",
        zorder=3,
    )
    ax.bar(
        xs,
        table_scores,
        width=w,
        color="#396bd3",
        edgecolor=DARK_TEXT,
        linewidth=1.0,
        label="Tables (5,386 tests)",
        zorder=3,
    )
    ax.bar(
        xs + w,
        layout_scores,
        width=w,
        color="#85a8f2",
        edgecolor=DARK_TEXT,
        linewidth=1.0,
        label="Layout (2,403 tests)",
        zorder=3,
    )

    ax.set_xticks(xs)
    ax.set_xticklabels([s[0] for s in systems], fontsize=9.8)
    for tick_label, s in zip(ax.get_xticklabels(), systems):
        if s[4]:
            tick_label.set_color(MARKER_BLUE)
            tick_label.set_fontweight("medium")

    ax.set_ylim(0, 108)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.tick_params(axis="both", which="major", length=0, pad=6)
    ax.set_ylabel("Pass rate (%)", fontsize=11.5, labelpad=10)
    ax.set_xlim(-0.6, len(systems) - 0.4)
    ax.legend(
        loc="upper right",
        frameon=False,
         ncol=3,
        fontsize=9.8,
        bbox_to_anchor=(1.0, 1.03),
    )

    _add_header_footer(
        fig,
        "OmniParseBench  ·  score by test family",
        "Per-family pass rates on Text (present, order, repeat), Tables (cell content & spans), and Layout (block labels).",
        "OmniParseBench: 16,288 unit tests on 2,937 pages. Marker fast and PP-StructureV3 run on CPU; Marker balanced & vllm and open VLMs run on GPU.",
    )
    fig.savefig(OUT_DIR / "omniparsebench_families.png", dpi=200)
    plt.close(fig)


def plot_throughput_bar():
    """Throughput ranked bar chart (log scale) across Marker modes and competitors."""
    fig = plt.figure(figsize=(9.6, 6.2), dpi=200)
    ax = fig.add_axes([0.085, 0.16, 0.885, 0.65])
    _style_axes(ax)

    items = [
        ("Marker\nno_ocr", 57.0, True),
        ("Marker\nfast", 23.6, True),
        ("Marker\nbalanced", 16.9, True),
        ("liteparse", 8.9, False),
        ("Marker\nvllm", 4.52, True),
        ("Tesseract 5", 2.92, False),
        ("docling", 2.09, False),
        ("MinerU", 0.54, False),
        ("PP-Struct\nV3", 0.18, False),
    ]

    xs = np.arange(len(items))
    vals = [it[1] for it in items]
    colors = [MARKER_BLUE if it[2] else OTHER_SLATE for it in items]

    bars = ax.bar(
        xs,
        vals,
        width=0.58,
        color=colors,
        edgecolor=DARK_TEXT,
        linewidth=1.3,
        zorder=3,
    )

    ax.set_yscale("log")
    ax.set_ylim(0.08, 140)

    for bar, (label, val, is_marker) in zip(bars, items):
        txt_color = MARKER_BLUE if is_marker else DARK_TEXT
        val_str = f"{val:.1f}" if val >= 1.0 else f"{val:.2f}"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val * 1.14,
            val_str,
            ha="center",
            va="bottom",
            fontsize=10.5,
            fontweight="medium" if is_marker else "normal",
            color=txt_color,
        )

    ax.set_xticks(xs)
    ax.set_xticklabels([it[0] for it in items], fontsize=10.0)
    for tick_label, (_, _, is_marker) in zip(ax.get_xticklabels(), items):
        if is_marker:
            tick_label.set_color(MARKER_BLUE)
            tick_label.set_fontweight("medium")

    ax.tick_params(axis="x", which="major", length=0, pad=6)
    ax.tick_params(axis="y", which="major", labelsize=10.5, length=0, pad=6)
    ax.tick_params(axis="y", which="minor", length=3, color="#666666")
    ax.set_ylabel("pages / sec  (log)", fontsize=11.5, labelpad=10)
    ax.set_xlim(-0.6, len(items) - 0.4)

    _add_header_footer(
        fig,
        "Throughput  ·  pages / sec",
        "Sustained concurrent pages/sec across local engines (log scale, higher is faster).",
        "Marker no_ocr, Marker fast, liteparse, Tesseract 5, and PP-StructureV3 run on CPU; Marker balanced, Marker vllm, docling, and MinerU use a single GPU.",
    )
    fig.savefig(OUT_DIR / "omniparsebench_throughput_bar.png", dpi=200)
    fig.savefig(OUT_DIR / "olmocr_throughput_bar.png", dpi=200)
    plt.close(fig)


def plot_marker_throughput_horizontal():
    """Horizontal bar chart of Marker's 4 modes throughput."""
    fig = plt.figure(figsize=(9.2, 4.6), dpi=200)
    ax = fig.add_axes([0.045, 0.19, 0.92, 0.56])
    _style_axes(ax)
    ax.spines["left"].set_color(SPINE_GRAY)
    ax.grid(axis="x", color=SPINE_GRAY, linewidth=0.9, zorder=0)

    # Bottom to top
    modes = [
        ("Marker  vllm  (GPU VLM)", 4.5, MARKER_BLUE, False),
        ("Marker  balanced  (GPU hybrid)", 16.9, MARKER_BLUE, False),
        ("Marker  fast  (CPU / ONNX)", 23.6, MARKER_LIGHT_BLUE, True),
        ("Marker  no_ocr  (CPU)", 57.0, MARKER_LIGHT_BLUE, True),
    ]

    ys = np.arange(len(modes))
    vals = [m[1] for m in modes]
    colors = [m[2] for m in modes]

    bars = ax.barh(
        ys,
        vals,
        height=0.56,
        color=colors,
        edgecolor="none",
        zorder=3,
    )

    for bar, (label, val, _, inside) in zip(bars, modes):
        y_center = bar.get_y() + bar.get_height() / 2
        if inside:
            ax.text(
                1.0,
                y_center,
                label,
                ha="left",
                va="center",
                fontsize=11.0,
                color="#ffffff",
                fontweight="medium",
                zorder=4,
            )
            ax.text(
                val + 0.9,
                y_center,
                f"{val:.1f} pg/s",
                ha="left",
                va="center",
                fontsize=11.5,
                color=DARK_TEXT,
                zorder=4,
            )
        else:
            ax.text(
                val + 0.9,
                y_center,
                f"{label}   {val:.1f} pg/s",
                ha="left",
                va="center",
                fontsize=11.0,
                color=DARK_TEXT,
                zorder=4,
            )

    ax.set_yticks([])
    ax.set_xlim(0, 65)
    ax.set_xticks([0, 10, 20, 30, 40, 50, 60])
    ax.tick_params(axis="x", labelsize=10.5, length=0, pad=6)
    ax.set_xlabel("pages / second  (higher is better)", fontsize=11.0, labelpad=8)

    _add_header_footer(
        fig,
        "Marker throughput  ·  by conversion mode",
        "Sustained concurrent pages/sec across Marker's four conversion modes on OmniParseBench.",
        "no_ocr and fast run in-process on CPU (pdftext + Indra ONNX layout & OCR, no VLM server); balanced and vllm share a GPU Surya 2.5 inference server.",
    )
    fig.savefig(OUT_DIR / "marker_throughput.png", dpi=200)
    plt.close(fig)


def plot_olmocr_charts():
    """Update legacy olmocr_*.png files so all mode names & scores are current."""
    # 1. olmocr_bench.png
    fig = plt.figure(figsize=(9.6, 6.6), dpi=200)
    ax = fig.add_axes([0.085, 0.13, 0.885, 0.68])
    _style_axes(ax)

    points = [
        ("MinerU", 0.54, 72.7, False, 0, -24, "center"),
        ("docling", 2.09, 50.3, False, 0, -24, "center"),
        ("liteparse", 8.9, 22.4, False, 0, -24, "center"),
        ("liteparse (no OCR)", 1721.0, 20.4, False, -20, -24, "center"),
        ("Marker  vllm", 4.5, 83.7, True, -10, 18, "center"),
        ("Marker  balanced", 16.9, 85.6, True, 0, 18, "center"),
        ("Marker  fast", 23.6, 70.8, True, 0, -24, "center"),
        ("Marker  no_ocr", 57.0, 44.0, True, 0, 18, "center"),
    ]

    for label, x, y, is_marker, dx, dy, ha in points:
        color = MARKER_BLUE if is_marker else OTHER_SLATE
        text_color = MARKER_BLUE if is_marker else DARK_TEXT
        ax.scatter(
            [x],
            [y],
            s=360,
            c=color,
            edgecolors=DARK_TEXT,
            linewidths=1.8,
            zorder=4,
        )
        ax.annotate(
            label,
            (x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            va="center",
            fontsize=11.5,
            fontweight="medium" if is_marker else "normal",
            color=text_color,
            zorder=5,
        )

    ax.set_xscale("log")
    ax.set_xlim(0.2, 4500)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.tick_params(axis="both", which="major", labelsize=10.5, length=0, pad=6)
    ax.tick_params(axis="x", which="minor", length=3, color="#666666")
    ax.set_xlabel(
        "throughput — pages / sec  ·  log scale, right is faster",
        fontsize=11.5,
        labelpad=10,
    )
    ax.set_ylabel("olmOCR-bench score", fontsize=11.5, labelpad=10)

    _add_header_footer(
        fig,
        "olmOCR-bench  ·  quality vs. throughput",
        "olmOCR-bench score vs. sustained throughput.  Marker beats MinerU and docling on both axes.",
        "Sustained pages/sec at max concurrency. Marker fast / no_ocr and liteparse run on CPU; Marker balanced / vllm, MinerU, and docling use a single GPU.",
    )
    fig.savefig(OUT_DIR / "olmocr_bench.png", dpi=200)
    plt.close(fig)

    # 2. olmocr_digital.png (CPU-only comparison)
    fig = plt.figure(figsize=(9.6, 6.6), dpi=200)
    ax = fig.add_axes([0.085, 0.13, 0.885, 0.68])
    _style_axes(ax)

    cpu_points = [
        ("Marker  fast", 23.6, 71.0, True, -10, 18, "center"),
        ("Indra OCR", 49.4, 69.5, False, 15, 18, "center"),
        ("Marker  no_ocr", 57.0, 55.8, True, 0, -24, "center"),
        ("PP-StructureV3", 0.18, 58.4, False, 10, -24, "center"),
        ("liteparse", 8.9, 27.3, False, 0, -24, "center"),
        ("Tesseract 5", 2.92, 17.6, False, 0, -24, "center"),
    ]

    for label, x, y, is_marker, dx, dy, ha in cpu_points:
        color = MARKER_BLUE if is_marker else OTHER_SLATE
        text_color = MARKER_BLUE if is_marker else DARK_TEXT
        ax.scatter(
            [x],
            [y],
            s=360,
            c=color,
            edgecolors=DARK_TEXT,
            linewidths=1.8,
            zorder=4,
        )
        ax.annotate(
            label,
            (x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            va="center",
            fontsize=11.5,
            fontweight="medium" if is_marker else "normal",
            color=text_color,
            zorder=5,
        )

    ax.set_xscale("log")
    ax.set_xlim(0.09, 150)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.tick_params(axis="both", which="major", labelsize=10.5, length=0, pad=6)
    ax.tick_params(axis="x", which="minor", length=3, color="#666666")
    ax.set_xlabel(
        "throughput — pages / sec  ·  log scale, right is faster",
        fontsize=11.5,
        labelpad=10,
    )
    ax.set_ylabel("Benchmark score", fontsize=11.5, labelpad=10)

    _add_header_footer(
        fig,
        "CPU-only document parsing  ·  quality vs. throughput",
        "Score vs. sustained pages/sec on CPU with no GPU required.",
        "Marker fast and Indra OCR use in-process ONNX layout, reading order, and OCR on CPU; Marker no_ocr uses the PDF text layer + Indra ONNX layout.",
    )
    fig.savefig(OUT_DIR / "olmocr_digital.png", dpi=200)
    plt.close(fig)

    # 3. olmocr_scores_bar.png
    fig = plt.figure(figsize=(9.6, 6.2), dpi=200)
    ax = fig.add_axes([0.085, 0.16, 0.885, 0.65])
    _style_axes(ax)

    items = [
        ("Chandra 2", 85.8, False),
        ("Marker\nvllm", 83.7, True),
        ("Gemini\nFlash 3.5", 76.4, False),
        ("Marker\nbalanced", 76.0, True),
        ("MinerU", 72.7, False),
        ("Marker\nfast", 66.6, True),
        ("docling", 50.3, False),
        ("Marker\nno_ocr", 43.6, True),
        ("liteparse", 22.4, False),
        ("liteparse\n(no OCR)", 20.4, False),
    ]

    xs = np.arange(len(items))
    scores = [it[1] for it in items]
    colors = [MARKER_BLUE if it[2] else OTHER_SLATE for it in items]

    bars = ax.bar(
        xs,
        scores,
        width=0.6,
        color=colors,
        edgecolor=DARK_TEXT,
        linewidth=1.3,
        zorder=3,
    )

    for bar, (label, score, is_marker) in zip(bars, items):
        txt_color = MARKER_BLUE if is_marker else DARK_TEXT
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            score + 1.5,
            f"{score:.1f}",
            ha="center",
            va="bottom",
            fontsize=10.2,
            fontweight="medium" if is_marker else "normal",
            color=txt_color,
        )

    ax.set_xticks(xs)
    ax.set_xticklabels([it[0] for it in items], fontsize=9.8)
    for tick_label, (_, _, is_marker) in zip(ax.get_xticklabels(), items):
        if is_marker:
            tick_label.set_color(MARKER_BLUE)
            tick_label.set_fontweight("medium")

    ax.set_ylim(0, 98)
    ax.set_yticks([0, 20, 40, 60, 80])
    ax.tick_params(axis="both", which="major", length=0, pad=6)
    ax.set_ylabel("olmOCR-bench score", fontsize=11.5, labelpad=10)
    ax.set_xlim(-0.6, len(items) - 0.4)

    _add_header_footer(
        fig,
        "olmOCR-bench  ·  score",
        "Full-bench macro-average across 8 categories.  Marker = this repo.",
        "Chandra 2 hosted; Gemini via API; MinerU / docling / liteparse are OSS pipelines. Higher is better.",
    )
    fig.savefig(OUT_DIR / "olmocr_scores_bar.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    plot_omniparsebench_scatter()
    plot_omniparsebench_scores_bar()
    plot_omniparsebench_families()
    plot_throughput_bar()
    plot_marker_throughput_horizontal()
    plot_olmocr_charts()
    print(f"Generated charts in {OUT_DIR}")
