"""Render a side-by-side comparison video of trajectories from three
checkpoints (early / mid / final) -> results/landing_comparison.mp4.

The booster is drawn enlarged (~2.5x) so it stays visible at the wide
camera framing; physics itself is true-scale.
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Polygon, Rectangle
import numpy as np

PAD_HALF = 12.0
ROCKET_SCALE = 2.5   # drawing exaggeration only
BODY_L, BODY_W = 42.0 * ROCKET_SCALE, 3.7 * ROCKET_SCALE


def draw_rocket(ax, x, y, theta, throttle):
    # body corners (nose at +L/2 along body up vector)
    ux, uy = np.sin(theta), np.cos(theta)
    px, py = uy, -ux  # perpendicular
    cx, cy = x, y
    nose = (cx + ux * BODY_L / 2, cy + uy * BODY_L / 2)
    tail = (cx - ux * BODY_L / 2, cy - uy * BODY_L / 2)
    hw = BODY_W / 2
    corners = [
        (nose[0] + px * hw, nose[1] + py * hw),
        (nose[0] - px * hw, nose[1] - py * hw),
        (tail[0] - px * hw, tail[1] - py * hw),
        (tail[0] + px * hw, tail[1] + py * hw),
    ]
    ax.add_patch(Polygon(corners, closed=True, fc="white", ec="black", lw=1.2, zorder=5))
    # engine flame, length ~ throttle
    if throttle > 0.02:
        fl = throttle * 90.0
        tip = (tail[0] - ux * fl, tail[1] - uy * fl)
        fw = BODY_W * 0.35 * throttle + 2
        flame = [(tail[0] + px * fw, tail[1] + py * fw),
                 (tail[0] - px * fw, tail[1] - py * fw), tip]
        ax.add_patch(Polygon(flame, closed=True, fc="orange", ec="red", alpha=0.85, zorder=4))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default="results")
    p.add_argument("--labels", nargs=3, default=None,
                   help="titles for the three panels")
    args = p.parse_args()

    traj_dir = os.path.join(args.out, "trajectories")
    files = sorted(os.listdir(traj_dir))
    assert len(files) >= 3, "need >=3 trajectories"
    # pick the BC init, the PPO-best, and a mid-training checkpoint
    # (first/middle/last would include the collapsed hover policy)
    want = ["traj_0_steps.npz", "traj_25000_steps.npz", "traj_50000_steps.npz"]
    picks = [f for f in want if f in files]
    assert len(picks) == 3, f"missing trajectories: {want}"
    trajs = [np.load(os.path.join(traj_dir, f)) for f in picks]
    labels = args.labels or [f.replace("traj_", "").replace("_steps.npz", "").replace("_", " ") + " steps"
                             for f in picks]

    states = [t["states"] for t in trajs]
    actions = [t["actions"] for t in trajs]
    n_frames = max(len(s) for s in states)
    stride = 6  # ~6x playback speed at 30 fps (keeps render time sane)

    fig, axes = plt.subplots(1, 3, figsize=(12, 5), sharex=True, sharey=True)
    writer = FFMpegWriter(fps=30, metadata={"title": "Falcon 9 landing: training progress"})
    out_path = os.path.join(args.out, "landing_comparison.mp4")

    # static scenery extents
    xmin, xmax, ymin, ymax = -520, 520, -40, 1650

    with writer.saving(fig, out_path, dpi=100):
        for k in range(0, n_frames, stride):
            for ax, st, ac, lab in zip(axes, states, actions, labels):
                ax.clear()
                ax.set_xlim(xmin, xmax)
                ax.set_ylim(ymin, ymax)
                ax.set_aspect("equal")
                ax.set_title(lab, fontsize=11)
                ax.set_xlabel("x (m)")
                # ground + pad
                ax.axhline(0, color="saddlebrown", lw=3)
                ax.add_patch(Rectangle((-PAD_HALF, 0), 2 * PAD_HALF, 6,
                                      fc="dimgray", ec="black", zorder=3))
                ax.text(0, 14, "LZ-1", ha="center", fontsize=8, color="dimgray")
                # trail
                j = min(k, len(st) - 1)
                past = st[max(0, j - 120):j + 1]
                ax.plot(past[:, 0], past[:, 1], color="royalblue", lw=1.2, alpha=0.8, zorder=2)
                x, y, vx, vy, th, om, fuel = st[j]
                thr = float((ac[min(j, len(ac) - 1)][0] + 1) / 2)
                draw_rocket(ax, x, max(y, 0), th, thr)
                speed = np.hypot(vx, vy)
                ax.text(0.02, 0.98,
                        f"t={j * 0.02:5.1f}s  alt={max(y, 0):6.0f} m\n"
                        f"spd={speed:5.1f} m/s  fuel={fuel:4.0%}\n"
                        f"thr={thr:4.0%}  tilt={np.degrees(th):5.1f} deg",
                        transform=ax.transAxes, va="top", fontsize=8,
                        family="monospace",
                        bbox=dict(fc="white", alpha=0.75, ec="none"))
            axes[0].set_ylabel("altitude (m)")
            fig.suptitle("Falcon 9 first-stage landing - PPO policy improving with training",
                         fontsize=13)
            fig.tight_layout()
            writer.grab_frame()
            if k % 60 == 0:
                print(f"frame {k}/{n_frames}")
    print("wrote", out_path)


if __name__ == "__main__":
    main()
