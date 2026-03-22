#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract and plot training loss points from a Hugging Face trainer_state.json file."
    )
    parser.add_argument(
        "--trainer-state",
        required=True,
        help="Path to trainer_state.json, e.g. outputs/checkpoint-168/trainer_state.json",
    )
    parser.add_argument(
        "--csv-out",
        default="outputs/loss_points.csv",
        help="Where to write the extracted loss points as CSV.",
    )
    parser.add_argument(
        "--png-out",
        default="outputs/loss_curve.png",
        help="Where to save the plotted loss curve PNG.",
    )
    parser.add_argument(
        "--svg-out",
        default="outputs/loss_curve.svg",
        help="Where to save a dependency-free SVG version of the loss curve.",
    )
    return parser.parse_args()


def load_trainer_state(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_loss_rows(trainer_state: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entry in trainer_state.get("log_history", []):
        step = entry.get("step")
        epoch = entry.get("epoch")
        if step is None:
            continue

        if "loss" in entry:
            rows.append(
                {
                    "step": step,
                    "epoch": epoch,
                    "split": "train",
                    "loss": entry["loss"],
                }
            )
    return rows


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "epoch", "split", "loss"])
        writer.writeheader()
        writer.writerows(rows)


def maybe_plot(rows: list[dict[str, Any]], path: Path) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))

    plt.plot(
        [row["step"] for row in rows],
        [row["loss"] for row in rows],
        marker="o",
        linewidth=1.8,
        label="Training loss",
    )

    plt.title("Training Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return True


def write_svg(rows: list[dict[str, Any]], path: Path) -> None:
    width = 900
    height = 540
    margin_left = 70
    margin_right = 20
    margin_top = 30
    margin_bottom = 55
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    all_steps = [row["step"] for row in rows]
    all_losses = [row["loss"] for row in rows]

    min_step = min(all_steps)
    max_step = max(all_steps)
    min_loss = min(all_losses)
    max_loss = max(all_losses)

    if min_step == max_step:
        max_step += 1
    if min_loss == max_loss:
        max_loss += 1

    def x_pos(step: float) -> float:
        return margin_left + ((step - min_step) / (max_step - min_step)) * plot_width

    def y_pos(loss: float) -> float:
        return margin_top + (1 - (loss - min_loss) / (max_loss - min_loss)) * plot_height

    def polyline_points(points: list[dict[str, Any]]) -> str:
        return " ".join(f"{x_pos(row['step']):.2f},{y_pos(row['loss']):.2f}" for row in points)

    x_ticks = sorted(set(all_steps))
    y_ticks = [min_loss + (max_loss - min_loss) * i / 4 for i in range(5)]
    path.parent.mkdir(parents=True, exist_ok=True)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>text{font-family:Arial,sans-serif;fill:#222} .small{font-size:12px} .label{font-size:14px} .title{font-size:20px;font-weight:bold}</style>',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="white"/>',
        f'<text class="title" x="{width / 2}" y="22" text-anchor="middle">Training Loss</text>',
        f'<line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - margin_right}" y2="{height - margin_bottom}" stroke="#333" stroke-width="1"/>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height - margin_bottom}" stroke="#333" stroke-width="1"/>',
    ]

    for loss_tick in y_ticks:
        y = y_pos(loss_tick)
        parts.append(
            f'<line x1="{margin_left}" y1="{y:.2f}" x2="{width - margin_right}" y2="{y:.2f}" stroke="#dddddd" stroke-width="1"/>'
        )
        parts.append(
            f'<text class="small" x="{margin_left - 8}" y="{y + 4:.2f}" text-anchor="end">{loss_tick:.3f}</text>'
        )

    for step_tick in x_ticks:
        x = x_pos(step_tick)
        parts.append(
            f'<line x1="{x:.2f}" y1="{margin_top}" x2="{x:.2f}" y2="{height - margin_bottom}" stroke="#f0f0f0" stroke-width="1"/>'
        )
        parts.append(
            f'<text class="small" x="{x:.2f}" y="{height - margin_bottom + 18}" text-anchor="middle">{step_tick}</text>'
        )

    parts.append(
        f'<polyline fill="none" stroke="#1f77b4" stroke-width="2.5" points="{polyline_points(rows)}"/>'
    )
    for row in rows:
        parts.append(
            f'<circle cx="{x_pos(row["step"]):.2f}" cy="{y_pos(row["loss"]):.2f}" r="3.5" fill="#1f77b4"/>'
        )

    legend_x = width - 180
    legend_y = margin_top + 10
    parts.extend(
        [
            f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 24}" y2="{legend_y}" stroke="#1f77b4" stroke-width="2.5"/>',
            f'<circle cx="{legend_x + 12}" cy="{legend_y}" r="3.5" fill="#1f77b4"/>',
            f'<text class="label" x="{legend_x + 32}" y="{legend_y + 5}">Training loss</text>',
            f'<text class="label" x="{width / 2}" y="{height - 12}" text-anchor="middle">Step</text>',
            f'<text class="label" x="18" y="{height / 2}" transform="rotate(-90 18 {height / 2})" text-anchor="middle">Loss</text>',
            "</svg>",
        ]
    )

    path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    args = parse_args()
    trainer_state_path = Path(args.trainer_state)
    trainer_state = load_trainer_state(trainer_state_path)
    rows = extract_loss_rows(trainer_state)

    if not rows:
        raise ValueError(f"No train/eval loss rows found in {trainer_state_path}")

    csv_out = Path(args.csv_out)
    png_out = Path(args.png_out)
    svg_out = Path(args.svg_out)
    write_csv(rows, csv_out)
    plotted = maybe_plot(rows, png_out)
    write_svg(rows, svg_out)

    print(
        json.dumps(
            {
                "trainer_state": str(trainer_state_path),
                "csv_out": str(csv_out),
                "png_out": str(png_out) if plotted else None,
                "svg_out": str(svg_out),
                "train_points": len(rows),
                "note": None if plotted else "matplotlib not installed; CSV was still generated",
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
