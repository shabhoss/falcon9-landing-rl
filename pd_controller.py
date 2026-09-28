"""Hand-tuned PD controller for the Falcon 9 landing env (demo generation)."""
import math
import numpy as np


def pd_action(env, s):
    x, y, vx, vy, th, om, fuel = s
    m = env.DRY_MASS + fuel * env.FUEL_MASS
    vy_des = -max(6.0, math.sqrt(5.0 * max(y, 0.0)))
    F = m * (env.G + 1.2 * (vy_des - vy)) / max(np.cos(th), 0.5)
    throttle = np.clip(F / env.MAX_THRUST, 0, 1)
    if y < 200:  # precision mode: gentle, direct position feedback
        th_des = np.clip(-(0.004 * x + 0.030 * vx), -0.10, 0.10)
    else:
        vx_des = np.clip(-0.12 * x, -22.0, 22.0)
        th_des = np.clip(0.04 * (vx_des - vx), -0.25, 0.25)
    # NOTE: positive gimbal DECREASES theta
    gimbal = np.clip(-12.0 * (th_des - th) + 3.0 * om, -1, 1)
    return np.array([throttle * 2 - 1, gimbal], dtype=np.float32)
