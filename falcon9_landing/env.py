"""Gymnasium environment: 2D Falcon 9 first-stage propulsive landing.

The agent controls the Merlin engine throttle and gimbal angle to land the
booster on the pad at x = 0. Physics is a planar rigid body: point-mass
translation with gravity, thrust, quadratic drag and wind, plus rotation
driven by gimballed-thrust torque.

State  (7,): x, y, vx, vy, theta, omega, fuel_frac
Action (2,): [throttle_cmd, gimbal_cmd] in [-1, 1]^2
  throttle = (throttle_cmd + 1) / 2 in [0, 1]
  gimbal   = gimbal_cmd * MAX_GIMBAL (radians)

Reward: potential-based shaping toward a guided descent profile, a fuel
penalty, and terminal bonuses/penalties for touchdown quality.
"""

from __future__ import annotations

import math

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class Falcon9LandingEnv(gym.Env):
    metadata = {"render_modes": []}

    # ---- vehicle / physics constants (Falcon 9 first stage, sea level) ----
    G = 9.81
    DRY_MASS = 22_000.0          # kg
    FUEL_MASS = 6_000.0          # kg
    ISP = 282.0                  # s, Merlin 1D sea level
    MAX_THRUST = 1.5 * (DRY_MASS + FUEL_MASS) * G  # N (~412 kN)
    LENGTH = 42.0                # m, for inertia
    GIMBAL_LEVER = 14.0          # m, engine gimbal moment arm about COM
    MAX_GIMBAL = 0.12            # rad (~7 deg)
    RHO = 1.225                  # kg/m^3
    CD_A = 8.0                   # m^2, drag area
    ANGULAR_DAMPING = 2.0e6      # N*m*s, grid-fin-ish passive damping

    # ---- episode / task constants ----
    DT = 0.02                    # s, 50 Hz sim
    MAX_STEPS = 1500             # 30 s
    PAD_HALF_WIDTH = 12.0        # m

    # touchdown tolerances for a successful landing
    VX_TOL = 6.0                 # m/s lateral
    VY_TOL = 10.0                # m/s vertical
    TH_TOL = 0.12                # rad tilt
    OM_TOL = 0.25                # rad/s tip rate

    GAMMA_SHAPE = 0.99           # discount used in potential shaping

    def __init__(self):
        super().__init__()
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-3.0, high=3.0, shape=(7,), dtype=np.float32)
        self._mdot_max = self.MAX_THRUST / (self.G * self.ISP)
        self.state = np.zeros(7, dtype=np.float64)
        self.wind = 0.0
        self.steps = 0
        self.difficulty = 1.0  # 0 = easy (low/slow starts), 1 = full task

    # ------------------------------------------------------------ curriculum
    def set_difficulty(self, d: float):
        """0 -> gentle low-altitude starts; 1 -> full task. Called by training."""
        self.difficulty = float(np.clip(d, 0.0, 1.0))

    @staticmethod
    def _lerp(a, b, d):
        return a + (b - a) * d

    # ------------------------------------------------------------------ api
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        r = self.np_random
        d = self.difficulty
        x = r.uniform(-self._lerp(80.0, 400.0, d), self._lerp(80.0, 400.0, d))
        y = r.uniform(self._lerp(400.0, 1200.0, d), self._lerp(700.0, 1600.0, d))
        vx = r.normal(0.0, self._lerp(6.0, 25.0, d))
        vy = r.normal(self._lerp(-15.0, -55.0, d), self._lerp(5.0, 12.0, d))
        theta = r.normal(0.0, self._lerp(0.05, 0.12, d))
        omega = r.normal(0.0, 0.05)
        self.state = np.array([x, y, vx, vy, theta, omega, 1.0], dtype=np.float64)
        w = self._lerp(3.0, 8.0, d)
        self.wind = float(r.uniform(-w, w))
        self.steps = 0
        return self._obs(), {}

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        throttle = float((a[0] + 1.0) / 2.0)
        gimbal = float(a[1] * self.MAX_GIMBAL)

        x, y, vx, vy, theta, omega, fuel = self.state
        phi_prev = self._potential(self.state)

        mass = self.DRY_MASS + fuel * self.FUEL_MASS
        if fuel <= 0.0:
            throttle = 0.0
        thrust = throttle * self.MAX_THRUST
        fuel = max(0.0, fuel - throttle * self._mdot_max * self.DT / self.FUEL_MASS)

        # --- translation: thrust + gravity + drag (with wind) ---
        ang = theta + gimbal
        fx = thrust * math.sin(ang)
        fy = thrust * math.cos(ang) - mass * self.G
        vrx, vry = vx - self.wind, vy
        vr = math.hypot(vrx, vry)
        fd = 0.5 * self.RHO * self.CD_A * vr
        fx -= fd * vrx
        fy -= fd * vry
        vx += fx / mass * self.DT
        vy += fy / mass * self.DT
        x += vx * self.DT
        y += vy * self.DT

        # --- rotation: gimbal torque (nose swings away from gimbal side) ---
        inertia = mass * self.LENGTH ** 2 / 12.0
        alpha = -(self.GIMBAL_LEVER * thrust * math.sin(gimbal)
                  + self.ANGULAR_DAMPING * omega) / inertia
        omega += alpha * self.DT
        theta += omega * self.DT
        theta = (theta + math.pi) % (2.0 * math.pi) - math.pi

        self.steps += 1
        self.state = np.array([x, y, vx, vy, theta, omega, fuel], dtype=np.float64)

        terminated, truncated = False, False
        terminal_reward, info = 0.0, {}

        if y <= 0.0:  # touchdown
            terminated = True
            self.state[1] = 0.0
            ok = (abs(x) <= self.PAD_HALF_WIDTH
                  and abs(vx) <= self.VX_TOL
                  and -self.VY_TOL <= vy <= 1.0
                  and abs(theta) <= self.TH_TOL
                  and abs(omega) <= self.OM_TOL)
            info = {"is_success": bool(ok)}
            if ok:
                terminal_reward = (150.0
                                   + 40.0 * (1.0 - abs(vy) / self.VY_TOL)
                                   + 20.0 * (1.0 - abs(vx) / self.VX_TOL)
                                   + 20.0 * (1.0 - abs(theta) / self.TH_TOL)
                                   + 30.0 * fuel)
            else:
                # graduated: partial credit for getting close to the pad,
                # so PPO sees a gradient even when it can't land perfectly
                ax = abs(x)
                if ax <= self.PAD_HALF_WIDTH:
                    terminal_reward = 80.0  # on pad, but too hard/tilted
                elif ax <= 50.0:
                    terminal_reward = 40.0  # near pad
                elif ax <= 120.0:
                    terminal_reward = 10.0  # in the vicinity
                else:
                    impact = math.hypot(vx, vy)
                    terminal_reward = -(60.0
                                        + min(120.0, 6.0 * impact)
                                        + 30.0
                                        + 80.0 * min(1.0, abs(theta) / 0.6))
        elif abs(x) > 1200.0 or y > 2600.0:
            terminated = True
            terminal_reward = -80.0
        elif self.steps >= self.MAX_STEPS:
            truncated = True
            # must be WORSE than any crash: hovering out the clock is the
            # local optimum we are trying to kill
            terminal_reward = -400.0

        # potential-based shaping toward the guided descent profile
        shaping = self.GAMMA_SHAPE * self._potential(self.state) - phi_prev
        reward = shaping - 1.0 * throttle * self.DT + terminal_reward
        # continuous anti-loitering penalty: descending slower than the guided
        # profile (or ascending) costs every step. A thresholded penalty has
        # a loophole (hover just under the threshold); this has none. A pure
        # terminal timeout penalty is invisible to gamma=0.99 over 1500 steps
        # (0.99^1500 ~ 3e-7), so hovering must hurt NOW, not at the deadline.
        vy_des = -max(6.0, 0.06 * y)
        if vy > vy_des and not (terminated or truncated):
            reward -= 0.03 * (vy - vy_des)
        return self._obs(), float(reward), terminated, truncated, info

    # -------------------------------------------------------------- helpers
    def _obs(self):
        x, y, vx, vy, theta, omega, fuel = self.state
        obs = np.array([x / 500.0, y / 1500.0, vx / 80.0, vy / 80.0,
                        theta / 0.6, omega / 1.5, fuel], dtype=np.float32)
        return np.clip(obs, -3.0, 3.0)

    def _potential(self, s):
        """Negative weighted distance to the desired descent profile."""
        x, y, vx, vy, theta, omega, _ = s
        vy_des = -max(6.0, 0.06 * y)          # slow as we near the ground
        px = min(abs(x), 600.0) / 600.0
        py = max(y, 0.0) / 1800.0
        pvx = min(abs(vx), 60.0) / 60.0
        pvy = min(abs(vy - vy_des), 90.0) / 90.0
        pth = min(abs(theta), 0.6) / 0.6
        pom = min(abs(omega), 1.5) / 1.5
        pot = -(1.2 * px + 1.5 * py + 0.6 * pvx + 0.8 * pvy + 1.0 * pth + 0.4 * pom)
        # final approach: strong lateral pull when low (must be over the pad
        # BEFORE descending into the zone where tilting is dangerous)
        if y < 150.0:
            pot -= 3.0 * min(abs(x), 150.0) / 150.0
        return pot
