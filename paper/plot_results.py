#!/usr/bin/env python3
"""Regenerate the README figure from unchanged paper CSV inputs."""

from pathlib import Path
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    root = Path(__file__).resolve().parent
    with (root / "data/gpu_24_per_kernel.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "svg.fonttype": "none", "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(12, 6.2), layout="constrained")
    x = np.arange(len(rows))
    ax.bar(x - .19, [float(r["rtx5090_source_over_full"]) for r in rows],
           width=.36, color="#276FBF", label="RTX 5090 D · geomean 1.305×")
    ax.bar(x + .19, [float(r["h100_source_over_full"]) for r in rows],
           width=.36, color="#E09540", label="H100 · geomean 1.440×")
    ax.axhline(1, color="#555555", lw=1, linestyle="--")
    ax.set_xticks(x, [r["kernel"] for r in rows], rotation=48, ha="right", fontsize=9)
    ax.set_ylabel("Speedup over original Triton (×)")
    ax.set_title("One recovered algorithm, two GPU targets", loc="left", pad=14, weight="bold")
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#e6e8eb", linewidth=.7)
    ax.set_xlim(-.7, len(rows)-.3)
    ax.legend(loc="upper left", frameon=False)
    root.joinpath("figures").mkdir(exist_ok=True)
    for suffix in ("png", "svg"):
        fig.savefig(root / f"figures/gpu_speedups.{suffix}", dpi=180, facecolor="white")
    plt.close(fig)
    print("Wrote paper/figures/gpu_speedups.{png,svg}")


if __name__ == "__main__":
    main()
