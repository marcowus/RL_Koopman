"""Residual RL example for the CSTR environment.

This script trains a base SAC policy that regulates the reactor
concentration toward ``c = 0.6`` and then adapts to a perturbed
environment (different coolant temperature) using a residual policy.  The
resulting trajectories for the base-only and base+residual controllers are
plotted for comparison.
"""

from typing import Optional, Dict, List

import numpy as np
import torch
import matplotlib.pyplot as plt
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
    """Environment wrapper that applies a base policy before residual actions.

    Parameters
    ----------
    env : gym.Env
        Wrapped environment.
    base_policy : SAC
        Frozen policy providing baseline actions.
    residual_policies : list[SAC], optional
        Previously learned residual policies whose actions are composed before
        the new residual action.  This allows progressive stacking of residual
        controllers when learning a sequence of tasks.
    """

    def __init__(
        self,
        env: gym.Env,
        base_policy: SAC,
        residual_policies: Optional[List[SAC]] = None,
    ):
        super().__init__()
        self.env = env
        self.base_policy = base_policy
        self.residual_policies = residual_policies or []
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self._last_obs = None

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._last_obs = obs
        return obs, info

    def step(self, residual_action):
        base_action, _ = self.base_policy.predict(self._last_obs, deterministic=True)

        for pol in self.residual_policies:
            corr, _ = pol.predict(self._last_obs, deterministic=True)
            base_action = np.clip(
                base_action + corr,
                self.env.action_space.low,
                self.env.action_space.high,
            )

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
    base = train_base(total_timesteps=200)
    residual = train_residual(base, total_timesteps=200)

    # Evaluate in a perturbed environment
    eval_env_base = make_env(param_overrides={"T_c": 0.40})
    eval_env_res = ResidualWrapper(make_env(param_overrides={"T_c": 0.40}), base)

    def rollout(env, policy, steps=50):
        obs, _ = env.reset()
        traj = []
        for _ in range(steps):
            action, _ = policy.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            c_val = env.model.state_scaler.unscale(torch.tensor(obs))[0].item() if isinstance(env, CSTR1Env) else env.env.model.state_scaler.unscale(torch.tensor(obs))[0].item()
            traj.append(c_val)
            if terminated or truncated:
                break
        return traj

    base_traj = rollout(eval_env_base, base)
    res_traj = rollout(eval_env_res, residual)

    plt.plot(base_traj, label="Base only")
    plt.plot(res_traj, label="Base + residual")
    plt.axhline(eval_env_base.c_target, color="r", linestyle="--", label="target")
    plt.xlabel("Step")
    plt.ylabel("Concentration c")
    plt.legend()
    plt.tight_layout()
    plt.savefig("residual_cstr_trajectory.png")
    print("Trajectory plot saved to residual_cstr_trajectory.png")
