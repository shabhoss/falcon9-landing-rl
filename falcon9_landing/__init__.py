from gymnasium.envs.registration import register

from .env import Falcon9LandingEnv

register(id="Falcon9Landing-v0", entry_point="falcon9_landing.env:Falcon9LandingEnv")

__all__ = ["Falcon9LandingEnv"]
