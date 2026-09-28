"""Train PPO on the Falcon 9 landing environment.

Saves a model checkpoint every --checkpoint-every steps plus an untrained
(step-0) snapshot, so evaluate.py can show the policy improving over time.
"""
from __future__ import annotations

import argparse
import os

import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from falcon9_landing import Falcon9LandingEnv


class CheckpointCallback(BaseCallback):
    """Save checkpoints and ramp env difficulty 0 -> 1 over `ramp_steps`."""

    def __init__(self, every: int, ckpt_dir: str, ramp_steps: int,
                 fixed_difficulty: float | None = None):
        super().__init__()
        self.every = every
        self.ckpt_dir = ckpt_dir
        self.ramp_steps = ramp_steps
        self.fixed_difficulty = fixed_difficulty

    def _on_step(self) -> bool:
        if self.num_timesteps % 4096 == 0:
            if self.fixed_difficulty is not None:
                d = self.fixed_difficulty
            else:
                d = min(1.0, self.num_timesteps / self.ramp_steps)
            self.model.get_env().env_method("set_difficulty", d)
        if self.num_timesteps % self.every == 0:
            path = os.path.join(self.ckpt_dir, f"ppo_{self.num_timesteps}_steps")
            self.model.save(path)
        return True


def make_env(seed: int):
    def _thunk():
        env = Monitor(Falcon9LandingEnv())
        env.reset(seed=seed)
        return env
    return _thunk


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=600_000)
    p.add_argument("--checkpoint-every", type=int, default=50_000)
    p.add_argument("--n-envs", type=int, default=4)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--ramp-steps", type=int, default=600_000,
                   help="env difficulty 0 -> 1 over this many steps")
    p.add_argument("--out", type=str, default="results")
    p.add_argument("--init-model", type=str, default=None,
                   help="zip path of a PPO model to warm-start from (e.g. BC init)")
    p.add_argument("--difficulty", type=float, default=None,
                   help="fixed difficulty in [0,1]; if unset, ramp 0->1 over --ramp-steps")
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--log-std-init", type=float, default=None,
                   help="if set with --init-model, fill policy log_std with this value")
    args = p.parse_args()

    ckpt_dir = os.path.join(args.out, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    vec_env = DummyVecEnv([make_env(args.seed + i) for i in range(args.n_envs)])

    policy_kwargs = dict(net_arch=[128, 128], activation_fn=th.nn.Tanh)
    if args.init_model:
        print("warm-starting from", args.init_model)
        model = PPO.load(args.init_model, env=vec_env)
        model.ent_coef = args.ent_coef
        model.verbose = 1
        # PPO.load restores the saved lr schedule; override both
        model.learning_rate = args.lr
        model.lr_schedule = lambda _progress_remaining: args.lr
        if args.log_std_init is not None:
            # keep early rollouts near the demonstrator
            with th.no_grad():
                model.policy.log_std.fill_(args.log_std_init)
    else:
        model = PPO(
            "MlpPolicy",
            vec_env,
            n_steps=1024,
            batch_size=256,
            n_epochs=3,
            gamma=0.99,
            gae_lambda=0.95,
            learning_rate=args.lr,
            clip_range=0.2,
            ent_coef=args.ent_coef,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=policy_kwargs,
            verbose=1,
            seed=args.seed,
        )
    # untrained snapshot = the "episode 0" policy for the comparison video
    model.save(os.path.join(ckpt_dir, "ppo_0_steps"))

    model.learn(
        total_timesteps=args.timesteps,
        callback=CheckpointCallback(args.checkpoint_every, ckpt_dir,
                                    args.ramp_steps,
                                    fixed_difficulty=args.difficulty),
    )
    model.save(os.path.join(args.out, "ppo_final"))
    print("done ->", os.path.join(args.out, "ppo_final"))


if __name__ == "__main__":
    main()
