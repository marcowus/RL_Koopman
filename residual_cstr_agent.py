"""Residual RL example for the CSTR environment.

This script demonstrates how to train a base SAC policy on the default
``CSTR1Env`` and then adapt to a modified environment using a residual
policy.  The residual policy learns corrections on top of the frozen base
policy.  The example is intentionally lightweight – the reward is zero and
the training steps are small – but it shows how to compose actions from a
base policy with residual actions.
"""

from typing import Optional, Dict

import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC

from pse_environments import CSTR1Env


def make_env(param_overrides: Optional[Dict[str, float]] = None) -> CSTR1Env:
    """Construct a ``CSTR1Env`` with optional parameter overrides."""

    delta_t = {"timestep": 15 * 60.0, "control": 60 * 60.0}
    env = CSTR1Env(delta_t, episode_length=50, param_overrides=param_overrides)
    return env


def train_base(total_timesteps: int = 100) -> SAC:
    """Train a base SAC policy on the nominal environment."""

    env = make_env()
    model = SAC("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=total_timesteps)
    return model


class ResidualWrapper(gym.Env):
    """Environment wrapper that applies a base policy before residual actions."""

    def __init__(self, env: gym.Env, base_policy: SAC):
        super().__init__()
        self.env = env
        self.base_policy = base_policy
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self._last_obs = None

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._last_obs = obs
        return obs, info

    def step(self, residual_action):
        base_action, _ = self.base_policy.predict(self._last_obs, deterministic=True)
        final_action = np.clip(
            base_action + residual_action,
            self.env.action_space.low,
            self.env.action_space.high,
        )
        obs, reward, terminated, truncated, info = self.env.step(final_action)
        self._last_obs = obs
        return obs, reward, terminated, truncated, info


def train_residual(base_model: SAC, total_timesteps: int = 100) -> SAC:
    """Train a residual SAC policy on a modified environment."""

    # Modify the coolant temperature parameter to create a new environment
    env = make_env(param_overrides={"T_c": 0.40})
    residual_env = ResidualWrapper(env, base_model)
    model = SAC("MlpPolicy", residual_env, verbose=0)
    model.learn(total_timesteps=total_timesteps)
    return model


if __name__ == "__main__":
    base = train_base(total_timesteps=100)
    residual = train_residual(base, total_timesteps=100)

    test_env = ResidualWrapper(make_env(param_overrides={"T_c": 0.40}), base)
    obs, _ = test_env.reset()
    for _ in range(5):
        res_action, _ = residual.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = test_env.step(res_action)
        if terminated or truncated:
            obs, _ = test_env.reset()

    base_action, _ = base.predict(obs, deterministic=True)
    final_action = np.clip(
        base_action + res_action,
        test_env.action_space.low,
        test_env.action_space.high,
    )
    print("Residual action:", res_action)
    print("Final combined action:", final_action)

