"""Collect successful PD-controller demo episodes -> <out>/demos.npz.

Stores flat arrays: obs (N,7), actions (N,2), returns (N,) with gamma=0.99,
keeping only episodes that end in a successful landing.
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from falcon9_landing import Falcon9LandingEnv
from pd_controller import pd_action

GAMMA = 0.99


def collect(difficulty: float, n_episodes: int, noise: float, seed0: int):
    rng = np.random.default_rng(seed0)
    obs_all, act_all, ret_all = [], [], []
    kept = 0
    for ep in range(n_episodes):
        env = Falcon9LandingEnv()
        env.set_difficulty(difficulty)
        obs, _ = env.reset(seed=seed0 + ep)
        ep_obs, ep_act, ep_rew = [], [], []
        done = False
        while not done:
            a = pd_action(env, env.state) + rng.normal(0.0, noise, 2)
            a = np.clip(a, -1.0, 1.0).astype(np.float32)
            ep_obs.append(obs)
            ep_act.append(a)
            obs, r, term, trunc, info = env.step(a)
            ep_rew.append(r)
            done = term or trunc
        if term and info.get("is_success", False):
            kept += 1
            # discounted returns
            g, rets = 0.0, []
            for r in reversed(ep_rew):
                g = r + GAMMA * g
                rets.append(g)
            rets.reverse()
            obs_all.extend(ep_obs)
            act_all.extend(ep_act)
            ret_all.extend(rets)
    return kept, n_episodes, obs_all, act_all, ret_all


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default="results")
    p.add_argument("--config", type=str, default="1.0:400,0.3:200",
                   help="difficulty:n_episodes,...")
    p.add_argument("--noise", type=float, default=0.03)
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    obs_all, act_all, ret_all = [], [], []
    total_kept, total_eps = 0, 0
    for i, chunk in enumerate(args.config.split(",")):
        d, n = chunk.split(":")
        kept, n_eps, o, a, r_ = collect(float(d), int(n), args.noise,
                                       seed0=10000 + 1000 * i)
        total_kept += kept
        total_eps += n_eps
        obs_all.extend(o)
        act_all.extend(a)
        ret_all.extend(r_)
        print(f"d={d}: kept {kept}/{n_eps}", flush=True)

    obs_all = np.array(obs_all, dtype=np.float32)
    act_all = np.array(act_all, dtype=np.float32)
    ret_all = np.array(ret_all, dtype=np.float32)
    path = os.path.join(args.out, "demos.npz")
    np.savez(path, obs=obs_all, actions=act_all, returns=ret_all)
    print(f"kept {total_kept}/{total_eps} episodes, {len(obs_all)} steps -> {path}")


if __name__ == "__main__":
    main()
