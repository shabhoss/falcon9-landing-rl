"""Evaluate every saved PPO checkpoint.

For each checkpoint: run N deterministic episodes, record mean reward and
success rate -> results/eval.json. Also saves one representative trajectory
(median-return episode) per checkpoint -> results/trajectories/*.npz, used
by render_videos.py.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

import numpy as np
from stable_baselines3 import PPO

from falcon9_landing import Falcon9LandingEnv


def steps_of(path: str) -> int:
    m = re.search(r"ppo_(\d+)_steps", path)
    return int(m.group(1)) if m else -1


def run_episode(model, seed: int, difficulty: float = 1.0):
    env = Falcon9LandingEnv()
    env.set_difficulty(difficulty)
    obs, _ = env.reset(seed=seed)
    states, actions = [env.state.copy()], []
    total, success = 0.0, False
    done = False
    while not done:
        act, _ = model.predict(obs, deterministic=True)
        actions.append(act.copy())
        obs, r, term, trunc, info = env.step(act)
        states.append(env.state.copy())
        total += r
        done = term or trunc
        if term:
            success = bool(info.get("is_success", False))
    return np.array(states), np.array(actions), total, success


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=24)
    p.add_argument("--difficulty", type=float, default=1.0)
    p.add_argument("--out", type=str, default="results")
    args = p.parse_args()

    ckpt_dir = os.path.join(args.out, "checkpoints")
    traj_dir = os.path.join(args.out, "trajectories")
    os.makedirs(traj_dir, exist_ok=True)

    ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "ppo_*_steps.zip")),
                   key=steps_of)
    assert ckpts, f"no checkpoints in {ckpt_dir}"

    summary = []
    for ckpt in ckpts:
        n = steps_of(ckpt)
        model = PPO.load(ckpt)
        rewards, successes, trajs = [], [], []
        for ep in range(args.episodes):
            st, ac, total, ok = run_episode(model, seed=1000 + ep,
                                            difficulty=args.difficulty)
            rewards.append(total)
            successes.append(ok)
            trajs.append((total, st, ac, ok))
        rewards = np.array(rewards)
        # representative trajectory: median-return episode
        trajs.sort(key=lambda t: t[0])
        _, st, ac, ok = trajs[len(trajs) // 2]
        np.savez(os.path.join(traj_dir, f"traj_{n}_steps.npz"),
                 states=st, actions=ac, success=ok)
        row = {"steps": n,
               "mean_reward": float(rewards.mean()),
               "std_reward": float(rewards.std()),
               "success_rate": float(np.mean(successes))}
        summary.append(row)
        print(f"steps={n:>7d}  mean_reward={row['mean_reward']:8.1f}  "
              f"success={row['success_rate']:.0%}")

    with open(os.path.join(args.out, "eval.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("wrote", os.path.join(args.out, "eval.json"))


if __name__ == "__main__":
    main()
