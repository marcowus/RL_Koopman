"""Basic SAC training script for CSTR1Env.

This script demonstrates how to create the environment and train a
Soft Actor-Critic agent from Stable-Baselines3 for a small number of
steps.  It can be used as a starting point for more advanced
experiments.
"""

from stable_baselines3 import SAC
import gymnasium as gym
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
    model = train_example(total_timesteps=100)
    env = make_env()
    obs, _ = env.reset()
    for _ in range(5):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()
    print("Sampled action after training:", action)
