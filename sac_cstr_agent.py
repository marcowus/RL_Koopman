"""Basic SAC training script for CSTR1Env.

This script demonstrates how to create the environment and train a
Soft Actor-Critic agent from Stable-Baselines3 for a small number of
steps.  It can be used as a starting point for more advanced
experiments.
"""

from stable_baselines3 import SAC
import gymnasium as gym
import torch
import matplotlib.pyplot as plt
from pse_environments import CSTR1Env


def make_env():
    """Construct a single instance of :class:`CSTR1Env`.

    Returns
    -------
    env : CSTR1Env
        Environment with default sampling intervals and a short
        episode length for quick experiments.
    """
    delta_t = {"timestep": 15 * 60.0, "control": 60 * 60.0}
    env = CSTR1Env(delta_t, episode_length=50)
    return env


def train_example(total_timesteps: int = 100) -> SAC:
    """Train a minimal SAC agent on the CSTR1Env.

    Parameters
    ----------
    total_timesteps: int
        Number of training timesteps.  Keep this low for quick
        demonstrations; increase for better policies.

    Returns
    -------
    model : SAC
        Trained SAC model.
    """
    env = make_env()
    model = SAC("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=total_timesteps)
    return model


if __name__ == "__main__":
    model = train_example(total_timesteps=200)
    env = make_env()
    obs, _ = env.reset()
    concentrations = []
    for _ in range(50):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        c_val = env.model.state_scaler.unscale(torch.tensor(obs))[0].item()
        concentrations.append(c_val)
        if terminated or truncated:
            break

    plt.plot(concentrations, label="SAC policy")
    plt.axhline(env.c_target, color="r", linestyle="--", label="target")
    plt.xlabel("Step")
    plt.ylabel("Concentration c")
    plt.legend()
    plt.tight_layout()
    plt.savefig("sac_cstr_trajectory.png")
    print("Trajectory plot saved to sac_cstr_trajectory.png")
