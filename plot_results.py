"""Plot learning curves from results/eval.json -> results/learning_curve.png."""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default="results")
    args = p.parse_args()

    with open(os.path.join(args.out, "eval.json")) as f:
        rows = json.load(f)
    steps = [r["steps"] for r in rows]
    mean = [r["mean_reward"] for r in rows]
    std = [r["std_reward"] for r in rows]
    succ = [r["success_rate"] for r in rows]

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))

    ax[0].plot(steps, mean, marker="o", color="#1f77b4")
    ax[0].fill_between(steps,
                       [m - s for m, s in zip(mean, std)],
                       [m + s for m, s in zip(mean, std)],
                       alpha=0.2, color="#1f77b4")
    ax[0].set_xlabel("environment steps")
    ax[0].set_ylabel("mean episode return")
    ax[0].set_title("PPO learning curve: episode return")
    ax[0].grid(alpha=0.3)

    ax[1].plot(steps, [s * 100 for s in succ], marker="o", color="#2ca02c")
    ax[1].set_xlabel("environment steps")
    ax[1].set_ylabel("landing success rate (%)")
    ax[1].set_title("PPO learning curve: successful landings")
    ax[1].set_ylim(-5, 105)
    ax[1].grid(alpha=0.3)

    fig.suptitle("Falcon 9 propulsive landing - PPO training progress")
    fig.tight_layout()
    out = os.path.join(args.out, "learning_curve.png")
    fig.savefig(out, dpi=120)
    print("wrote", out)


if __name__ == "__main__":
    main()
