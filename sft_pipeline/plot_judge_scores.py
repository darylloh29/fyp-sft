#!/usr/bin/env python3
import argparse
import json
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot a histogram of judge scores from a judged JSONL file."
    )
    parser.add_argument(
        "--input",
        default="data/judged_teacher.jsonl",
        help="Path to judged JSONL input.",
    )
    parser.add_argument(
        "--png-out",
        default="outputs/judge_score_histogram.png",
        help="Where to save the histogram PNG.",
    )
    parser.add_argument(
        "--svg-out",
        default="outputs/judge_score_histogram.svg",
        help="Where to save the histogram SVG.",
    )
    return parser.parse_args()


def load_scores(path: Path) -> list[float]:
    scores: list[float] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            scores.append(float(row["judge"]["score"]))
    return scores


def plot_with_matplotlib(scores: list[float], path: Path) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    counts = Counter(scores)
    xs = sorted(counts)
    ys = [counts[x] for x in xs]

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 5.5))
    plt.bar(xs, ys, width=0.22, color="#2f6db2", edgecolor="white")
    plt.title("Judge Score Histogram")
    plt.xlabel("Score")
    plt.ylabel("Count")
    plt.xticks(xs)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return True


def write_svg(scores: list[float], path: Path) -> None:
    counts = Counter(scores)
    xs = sorted(counts)
    ys = [counts[x] for x in xs]

    width = 900
    height = 540
    margin_left = 70
    margin_right = 30
    margin_top = 35
    margin_bottom = 70
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    max_y = max(ys)

    def x_pos(index: int) -> float:
        if len(xs) == 1:
            return margin_left + plot_width / 2
        return margin_left + index * (plot_width / len(xs)) + (plot_width / len(xs)) / 2

    def bar_width() -> float:
        return (plot_width / max(len(xs), 1)) * 0.7

    def y_pos(value: float) -> float:
        return margin_top + (1 - value / max_y) * plot_height

    y_ticks = [round(max_y * i / 4) for i in range(5)]
    path.parent.mkdir(parents=True, exist_ok=True)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>text{font-family:Arial,sans-serif;fill:#222}.small{font-size:12px}.label{font-size:14px}.title{font-size:20px;font-weight:bold}</style>',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="white"/>',
        f'<text class="title" x="{width / 2}" y="24" text-anchor="middle">Judge Score Histogram</text>',
        f'<line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - margin_right}" y2="{height - margin_bottom}" stroke="#333" stroke-width="1"/>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height - margin_bottom}" stroke="#333" stroke-width="1"/>',
    ]

    for tick in y_ticks:
        y = y_pos(tick)
        parts.append(
            f'<line x1="{margin_left}" y1="{y:.2f}" x2="{width - margin_right}" y2="{y:.2f}" stroke="#dddddd" stroke-width="1"/>'
        )
        parts.append(
            f'<text class="small" x="{margin_left - 8}" y="{y + 4:.2f}" text-anchor="end">{tick}</text>'
        )

    bw = bar_width()
    for index, (score, count) in enumerate(zip(xs, ys)):
        center_x = x_pos(index)
        top_y = y_pos(count)
        rect_x = center_x - bw / 2
        rect_height = height - margin_bottom - top_y
        parts.append(
            f'<rect x="{rect_x:.2f}" y="{top_y:.2f}" width="{bw:.2f}" height="{rect_height:.2f}" fill="#2f6db2"/>'
        )
        parts.append(
            f'<text class="small" x="{center_x:.2f}" y="{height - margin_bottom + 18}" text-anchor="middle">{score:g}</text>'
        )
        parts.append(
            f'<text class="small" x="{center_x:.2f}" y="{top_y - 8:.2f}" text-anchor="middle">{count}</text>'
        )

    parts.extend(
        [
            f'<text class="label" x="{width / 2}" y="{height - 20}" text-anchor="middle">Score</text>',
            f'<text class="label" x="18" y="{height / 2}" transform="rotate(-90 18 {height / 2})" text-anchor="middle">Count</text>',
            "</svg>",
        ]
    )

    path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    scores = load_scores(input_path)
    if not scores:
        raise ValueError(f"No judge scores found in {input_path}")

    png_out = Path(args.png_out)
    svg_out = Path(args.svg_out)
    plotted = plot_with_matplotlib(scores, png_out)
    write_svg(scores, svg_out)

    print(
        json.dumps(
            {
                "input": str(input_path),
                "png_out": str(png_out) if plotted else None,
                "svg_out": str(svg_out),
                "samples": len(scores),
                "min_score": min(scores),
                "max_score": max(scores),
                "note": None if plotted else "matplotlib not installed; SVG was still generated",
            }
        )
    )


if __name__ == "__main__":
    main()
