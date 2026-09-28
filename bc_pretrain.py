"""Behavioral-clone PD demos into a fresh PPO policy (actor + value).

Trains the SB3 policy's action head (MSE vs demo actions) and value head
(MSE vs demo discounted returns) with plain supervised learning, then saves
the model as a PPO warm start: <out>/ppo_bc_init.zip
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch as th
import torch.nn.functional as F
from stable_baselines3 import PPO

from falcon9_landing import Falcon9LandingEnv  # noqa: F401 (registers env)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--demos", type=str, default="results/demos.npz")
    p.add_argument("--out", type=str, default="results")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    args = p.parse_args()

    data = np.load(args.demos)
    obs, act, ret = data["obs"], data["actions"], data["returns"]
    print(f"demos: {len(obs)} steps")
    ret_mean, ret_std = float(ret.mean()), float(ret.std())
    print(f"return mean={ret_mean:.1f} std={ret_std:.1f}")
    # normalize returns for stable value pretraining; rescale the value head
    # back to raw units afterwards so PPO advantages are well-scaled
    ret = (ret - ret_mean) / (ret_std + 1e-8)

    env = Falcon9LandingEnv()
    model = PPO("MlpPolicy", env,
                policy_kwargs=dict(net_arch=[128, 128],
                                   activation_fn=th.nn.Tanh),
                verbose=0)
    policy = model.policy
    opt = th.optim.Adam(policy.parameters(), lr=args.lr)
    n = len(obs)

    for epoch in range(args.epochs):
        perm = np.random.permutation(n)
        tot = 0.0
        for i in range(0, n, args.batch):
            b = perm[i:i + args.batch]
            ob = th.as_tensor(obs[b])
            ac = th.as_tensor(act[b])
            rt = th.as_tensor(ret[b])
            feats = policy.extract_features(ob)
            latent_pi, latent_vf = policy.mlp_extractor(feats)
            mu = policy.action_net(latent_pi)
            vals = policy.value_net(latent_vf).flatten()
            loss = F.mse_loss(mu, ac) + 0.5 * F.mse_loss(vals, rt)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"epoch {epoch + 1}/{args.epochs}  loss={tot / n:.4f}", flush=True)

    # quick check: action MSE on a holdout slice
    with th.no_grad():
        feats = policy.extract_features(th.as_tensor(obs[:2000]))
        mu = policy.action_net(policy.mlp_extractor(feats)[0])
        mse = F.mse_loss(mu, th.as_tensor(act[:2000])).item()
    print(f"train action MSE: {mse:.4f}")

    # rescale the value head's final linear layer back to raw return units:
    # V_raw(s) = std * V_norm(s) + mean  (exact for a final linear layer)
    with th.no_grad():
        lin = policy.value_net
        assert isinstance(lin, th.nn.Linear) and lin.out_features == 1
        lin.weight.mul_(ret_std)
        lin.bias.mul_(ret_std).add_(ret_mean)
    with th.no_grad():
        v = policy.predict_values(th.as_tensor(obs[:256])).flatten()
        print(f"value head check: mean={v.mean().item():.1f} "
              f"(demo raw mean={ret_mean:.1f})")

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "ppo_bc_init")
    model.save(path)
    print("saved", path + ".zip")


if __name__ == "__main__":
    main()
