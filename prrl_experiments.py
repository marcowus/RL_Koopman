"""Lightweight PRRL experiment suite.

This script runs a minimal set of experiments illustrating the workflow
outlined in the project notes:

1. Train a base SAC policy on the nominal CSTR1Env environment.
2. Compare different strategies on a perturbed environment:
   - Tabula rasa (train from scratch)
   - Base policy only
   - Base + Residual policy
3. Demonstrate progressive residual learning across multiple tasks and
   evaluate retention of earlier task performance.

The experiments are intentionally small to keep runtime short.  They do
not aim to produce strong control performance; instead they provide code
structure that researchers can extend with longer training runs and
richer metrics/plots.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy

from residual_cstr_agent import make_env, ResidualWrapper, train_base


@dataclass
class EvalResult:
    label: str
    mean_reward: float


def train_sac(env, total_timesteps: int = 100) -> SAC:
    model = SAC("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=total_timesteps)
    return model


def evaluate_wrapper(wrapper, episodes: int = 5) -> EvalResult:
    class _ZeroPolicy:
        def __init__(self, action_space):
            self.action_space = action_space

        def predict(self, obs, state=None, episode_start=None, deterministic=True):
            return np.zeros(self.action_space.shape), state

    policy = _ZeroPolicy(wrapper.action_space)
    mean, _ = evaluate_policy(policy, wrapper, n_eval_episodes=episodes, warn=False)
    return EvalResult("wrapper", mean)


def run_base_training(timesteps: int = 100) -> SAC:
    base = train_base(total_timesteps=timesteps)
    env = make_env()
    mean, _ = evaluate_policy(base, env, n_eval_episodes=5, warn=False)
    print(f"Base policy mean reward: {mean:.2f}")
    return base


def rrl_comparison(base_model: SAC, timesteps: int = 100):
    task_env = make_env(param_overrides={"T_c": 0.40})

    tabula_model = train_sac(task_env, timesteps)
    tabula_reward, _ = evaluate_policy(tabula_model, task_env, n_eval_episodes=5, warn=False)

    base_reward, _ = evaluate_policy(base_model, task_env, n_eval_episodes=5, warn=False)

    residual_env = ResidualWrapper(task_env, base_model)
    residual_model = train_sac(residual_env, timesteps)
    residual_reward = evaluate_wrapper(ResidualWrapper(task_env, base_model, [residual_model])).mean_reward

    print("RRL comparison (mean reward over 5 episodes):")
    print(f"  Tabula Rasa     : {tabula_reward:.2f}")
    print(f"  Base Only       : {base_reward:.2f}")
    print(f"  Base + Residual : {residual_reward:.2f}")
    return residual_model


def progressive_learning(base_model: SAC, overrides: List[Dict[str, float]], timesteps: int = 100):
    residual_models: List[SAC] = []
    for idx, params in enumerate(overrides, start=1):
        env = make_env(param_overrides=params)
        wrapper = ResidualWrapper(env, base_model, residual_models.copy())
        model = train_sac(wrapper, timesteps)
        residual_models.append(model)
        print(f"Trained residual for task {idx} with params {params}")

        # Evaluate retention on all seen tasks
        for j, prev_params in enumerate(overrides[:idx], start=1):
            eval_env = make_env(param_overrides=prev_params)
            eval_wrapper = ResidualWrapper(eval_env, base_model, residual_models[:j])
            result = evaluate_wrapper(eval_wrapper)
            print(f"  Eval task {j}: mean reward {result.mean_reward:.2f}")
    return residual_models


def ablation_remove_pnn(base_model: SAC, overrides: List[Dict[str, float]], timesteps: int = 100):
    model: Optional[SAC] = None
    for idx, params in enumerate(overrides, start=1):
        env = make_env(param_overrides=params)
        wrapper = ResidualWrapper(env, base_model, [model] if model else [])
        if model is None:
            model = train_sac(wrapper, timesteps)
        else:
            model.set_env(wrapper)
            model.learn(total_timesteps=timesteps)
        base_eval_env = make_env(param_overrides=overrides[0])
        base_eval_wrapper = ResidualWrapper(base_eval_env, base_model, [model])
        result = evaluate_wrapper(base_eval_wrapper)
        print(f"After training task {idx}, reward on first task: {result.mean_reward:.2f}")


def main():
    base_model = run_base_training(timesteps=100)
    print("\n--- Residual RL comparison ---")
    residual_model = rrl_comparison(base_model, timesteps=100)

    print("\n--- Progressive learning over tasks ---")
    task_params = [
        {"T_c": 0.35},
        {"T_c": 0.40},
        {"T_c": 0.45},
    ]
    progressive_learning(base_model, task_params, timesteps=100)

    print("\n--- Ablation: remove PNN (fine-tune single residual) ---")
    ablation_remove_pnn(base_model, task_params, timesteps=100)


if __name__ == "__main__":
    main()
