#!/usr/bin/env python
"""
rys_analyze.py — Generate heatmaps and Pareto analysis from RYS sweep results.

Usage:
    python rys_analyze.py                              # analyze latest results
    python rys_analyze.py --results path/to/sweep.json # specific results file
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

RESULTS_DIR = str(Path(__file__).parent / "results")
RESULTS_FILE = os.path.join(RESULTS_DIR, "rys_sweep.json")
NUM_LAYERS = 36


def load_results(path):
    with open(path) as f:
        return json.load(f)


def build_heatmaps(data, num_layers):
    """Build (i,j) heatmap matrices for math, EQ, and combined deltas."""
    baseline = data.get("0,0", {})
    bl_math = baseline.get("math_score", 0)
    bl_eq = baseline.get("eq_score", 0)
    bl_combined = bl_math + bl_eq

    # Initialize with NaN (missing configs)
    math_map = np.full((num_layers + 1, num_layers + 1), np.nan)
    eq_map = np.full((num_layers + 1, num_layers + 1), np.nan)
    combined_map = np.full((num_layers + 1, num_layers + 1), np.nan)

    for key, val in data.items():
        parts = key.split(",")
        if len(parts) != 2:
            continue
        i, j = int(parts[0]), int(parts[1])
        math_map[i, j] = val["math_score"] - bl_math
        eq_map[i, j] = val["eq_score"] - bl_eq
        combined_map[i, j] = val["combined"] - bl_combined

    return math_map, eq_map, combined_map, bl_math, bl_eq, bl_combined


def find_pareto_frontier(data, num_layers):
    """Find Pareto-optimal configs (best combined at each overhead level)."""
    baseline_combined = data.get("0,0", {}).get("combined", 0)
    candidates = []

    for key, val in data.items():
        if key == "0,0":
            continue
        if val["combined"] <= baseline_combined:
            continue
        candidates.append({
            "config": key,
            "combined": val["combined"],
            "delta": val["combined"] - baseline_combined,
            "math": val["math_score"],
            "eq": val["eq_score"],
            "extra_layers": val["extra_layers"],
            "overhead_pct": val["overhead_pct"],
        })

    # Sort by overhead
    candidates.sort(key=lambda x: x["overhead_pct"])

    # Pareto filter: keep only if combined is strictly better than all cheaper configs
    pareto = []
    best_so_far = baseline_combined
    for c in candidates:
        if c["combined"] > best_so_far:
            pareto.append(c)
            best_so_far = c["combined"]

    return pareto


def plot_heatmaps(math_map, eq_map, combined_map, num_layers, out_dir):
    """Generate and save heatmap images."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import TwoSlopeNorm
    except ImportError:
        print("matplotlib not installed — skipping heatmap plots.")
        return

    os.makedirs(out_dir, exist_ok=True)

    for name, data_map, title in [
        ("math_delta", math_map, "Math Score Delta vs Baseline"),
        ("eq_delta", eq_map, "EQ Score Delta vs Baseline"),
        ("combined_delta", combined_map, "Combined Score Delta vs Baseline"),
    ]:
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))

        # Center colormap at 0 (red=improvement, blue=degradation)
        valid = data_map[~np.isnan(data_map)]
        if len(valid) == 0:
            continue

        vmin, vmax = np.nanmin(valid), np.nanmax(valid)
        # Ensure we have both negative and positive for diverging colormap
        if vmin >= 0:
            vmin = -0.01
        if vmax <= 0:
            vmax = 0.01

        norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

        im = ax.imshow(
            data_map,
            cmap="RdBu_r",  # Red=positive (good), Blue=negative (bad)
            norm=norm,
            origin="upper",
            aspect="equal",
            interpolation="nearest",
        )

        ax.set_xlabel("j (repeat end)", fontsize=12)
        ax.set_ylabel("i (repeat start)", fontsize=12)
        ax.set_title(f"Qwen3-4B RYS Scan: {title}", fontsize=14)

        # Tick every 4 layers
        ticks = list(range(0, num_layers + 1, 4))
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)

        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("Score delta vs baseline", fontsize=10)

        plt.tight_layout()
        path = os.path.join(out_dir, f"{name}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")


def plot_triptych(math_map, eq_map, combined_map, num_layers, out_dir):
    """Single figure with math / EQ / combined delta heatmaps side by side."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import TwoSlopeNorm
    except ImportError:
        print("matplotlib not installed — skipping triptych plot.")
        return

    os.makedirs(out_dir, exist_ok=True)

    panels = [
        (math_map, "Math Score Delta"),
        (eq_map, "EQ Score Delta"),
        (combined_map, "Combined Delta"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(24, 8))

    for ax, (data_map, title) in zip(axes, panels):
        valid = data_map[~np.isnan(data_map)]
        if len(valid) == 0:
            continue

        vmin, vmax = np.nanmin(valid), np.nanmax(valid)
        if vmin >= 0:
            vmin = -0.01
        if vmax <= 0:
            vmax = 0.01

        norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        im = ax.imshow(
            data_map,
            cmap="RdBu_r",
            norm=norm,
            origin="upper",
            aspect="equal",
            interpolation="nearest",
        )

        ax.set_xlabel("j (repeat end)", fontsize=11)
        ax.set_ylabel("i (repeat start)", fontsize=11)
        ax.set_title(title, fontsize=13, fontweight="bold")

        ticks = list(range(0, num_layers + 1, 4))
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)

        fig.colorbar(im, ax=ax, shrink=0.75)

    fig.suptitle("Qwen3-4B RYS Heatmap Scan (36 layers, 667 configs)",
                 fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = os.path.join(out_dir, "rys_triptych.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_pareto(pareto, baseline_combined, out_dir):
    """Plot Pareto frontier: overhead vs combined score."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    if not pareto:
        print("No Pareto-optimal configs found.")
        return

    os.makedirs(out_dir, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    overheads = [p["overhead_pct"] for p in pareto]
    scores = [p["combined"] for p in pareto]
    labels = [p["config"] for p in pareto]

    ax.axhline(y=baseline_combined, color="gray", linestyle="--", alpha=0.7, label="Baseline")
    ax.plot(overheads, scores, "ro-", markersize=8, label="Pareto frontier")

    for x, y, label in zip(overheads, scores, labels):
        ax.annotate(f"({label})", (x, y), textcoords="offset points",
                    xytext=(5, 8), fontsize=8)

    ax.set_xlabel("Overhead (%)", fontsize=12)
    ax.set_ylabel("Combined Score (Math + EQ)", fontsize=12)
    ax.set_title("Qwen3-4B RYS: Pareto Frontier", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, "pareto_frontier.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def write_summary(data, pareto, bl_math, bl_eq, bl_combined, out_dir):
    """Write markdown summary of results."""
    os.makedirs(out_dir, exist_ok=True)

    ranked = sorted(data.items(), key=lambda x: x[1]["combined"], reverse=True)
    beating = [
        (k, v) for k, v in ranked
        if v["combined"] > bl_combined and k != "0,0"
    ]

    lines = [
        "# RYS Sweep Results — Qwen3-4B (36 layers)\n",
        f"## Baseline Scores\n",
        f"- Math: {bl_math:.4f}",
        f"- EQ: {bl_eq:.4f}",
        f"- Combined: {bl_combined:.4f}\n",
        f"## Summary\n",
        f"- Total configs evaluated: {len(data)}",
        f"- Configs beating baseline: {len(beating)}/{len(data) - 1}",
        f"- Early-stopped (garbage): {sum(1 for v in data.values() if v.get('early_stopped', False))}\n",
        f"## Top 20 Configs\n",
        f"| Config | Math | EQ | Combined | Delta | Extra Layers | Overhead % |",
        f"|--------|------|-----|----------|-------|-------------|-----------|",
    ]

    for key, val in ranked[:20]:
        delta = val["combined"] - bl_combined
        marker = " **" if key == "0,0" else ""
        lines.append(
            f"| {key}{marker} | {val['math_score']:.4f} | {val['eq_score']:.4f} | "
            f"{val['combined']:.4f} | {delta:+.4f} | {val['extra_layers']} | "
            f"{val['overhead_pct']:.1f} |"
        )

    if pareto:
        lines.extend([
            f"\n## Pareto Frontier\n",
            f"| Size | Config | Extra Layers | Overhead % | Math | EQ | Combined | Delta |",
            f"|------|--------|-------------|-----------|------|-----|----------|-------|",
        ])
        sizes = ["S", "M", "L", "XL", "XXL"]
        for idx, p in enumerate(pareto):
            size = sizes[idx] if idx < len(sizes) else f"P{idx}"
            lines.append(
                f"| {size} | ({p['config']}) | {p['extra_layers']} | "
                f"{p['overhead_pct']:.1f} | {p['math']:.4f} | {p['eq']:.4f} | "
                f"{p['combined']:.4f} | {p['delta']:+.4f} |"
            )

    summary_path = os.path.join(out_dir, "rys_summary.md")
    with open(summary_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved: {summary_path}")

    # Also save as JSON
    json_path = os.path.join(out_dir, "pareto.json")
    with open(json_path, "w") as f:
        json.dump(pareto, f, indent=2)
    print(f"Saved: {json_path}")


def main():
    parser = argparse.ArgumentParser(description="RYS sweep analysis")
    parser.add_argument("--results", default=RESULTS_FILE)
    parser.add_argument("--out-dir", default=os.path.join(RESULTS_DIR, "analysis"))
    parser.add_argument("--num-layers", type=int, default=NUM_LAYERS)
    args = parser.parse_args()

    print(f"Loading results from {args.results}...")
    data = load_results(args.results)
    print(f"Loaded {len(data)} configs.")

    math_map, eq_map, combined_map, bl_math, bl_eq, bl_combined = build_heatmaps(
        data, args.num_layers
    )

    print("\nGenerating heatmaps...")
    plot_heatmaps(math_map, eq_map, combined_map, args.num_layers, args.out_dir)
    plot_triptych(math_map, eq_map, combined_map, args.num_layers, args.out_dir)

    print("\nFinding Pareto frontier...")
    pareto = find_pareto_frontier(data, args.num_layers)
    plot_pareto(pareto, bl_combined, args.out_dir)

    print("\nWriting summary...")
    write_summary(data, pareto, bl_math, bl_eq, bl_combined, args.out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
