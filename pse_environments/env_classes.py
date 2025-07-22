# NOTE: If you load, deepcopy, or assign a model/ensemble, always call move_model_to_device(model, device) on both the ensemble and each submodel.
import numpy as np
import gymnasium as gym
import time 
from common import get_default_device
import torch

if __name__ == "__main__":
    import common
else:
    try: # TODO: find a real solution for this
        from . import common
    except:
        import common

class DemandResponseEnvironment(gym.Env):
    def __init__(self,
                 process_model,
                 episode_length = 24*3,
                 price_prediction_horizon = 9,
                 storage_size = 6,
                 default_price_reset_type = 'train',
                 default_state_reset_type = 'steady_state',
                 default_use_discrete_model = False,
                 default_idx_discrete_model = None,
                 constant_constr_viol_penalty = -1.0,
                 step_reward = 2.0,
                 cost_reward_scaling_factor = 5e-5,
                 reward_range = (-10, np.inf),
                 terminate_episode_at_relative_constr_viol =  np.inf,
                 device=None
                 ):
        self.device = device if device is not None else get_default_device()  # Store device
        self.process_model = process_model
        self.delta_t = process_model.delta_t
        self.ctrl_steps_per_hour = int(60*60 / self.delta_t['control'])
        self.episode_length = {
            'hours': episode_length,
            'ctrl_steps': int(episode_length * self.ctrl_steps_per_hour)}
        self.price_prediction_horizon = {
            'hours': price_prediction_horizon,
            'ctrl_steps': int(price_prediction_horizon * self.ctrl_steps_per_hour)}
        self.storage_size = storage_size
        self.price_generator = common.Prices_generator()

        self.default_price_reset_type = default_price_reset_type
        self.default_state_reset_type = default_state_reset_type
        self.default_use_discrete_model = default_use_discrete_model
        self.default_idx_discrete_model = default_idx_discrete_model

        self.observation_space = gym.spaces.Box(
            low = -np.inf, high = np.inf,
            shape = (process_model.n_states + 1
                     + self.price_prediction_horizon['ctrl_steps'],),
            dtype = np.float32)
        self.action_space = process_model.action_space
        self.max_state = process_model.x_bounds['upper']
        self.min_state = process_model.x_bounds['lower']
        self.state_mid = 0.5 * (self.max_state + self.min_state)
        self.state_range = self.max_state - self.min_state

        # reward shaping
        self.constant_constr_viol_penalty = constant_constr_viol_penalty
        self.step_reward = step_reward
        self.cost_reward_scaling_factor = cost_reward_scaling_factor
        self.reward_range = reward_range
        self.terminate_episode_at_relative_constr_viol = terminate_episode_at_relative_constr_viol

    def step(self, action,
             use_discrete_model = None,
             idx_discrete_model = None):
        # Convert action to tensor on the correct device
        if isinstance(action, np.ndarray):
            action = torch.tensor(action, dtype=torch.float32, device=self.device)
        price = self.electricity_price[self.num_steps]
        use_discrete_model = self.default_use_discrete_model\
            if use_discrete_model is None\
            else use_discrete_model
        idx_discrete_model = self.default_idx_discrete_model\
            if idx_discrete_model is None\
            else idx_discrete_model

        # solve initial value problem
        x, storage, cost, steady_state_cost =\
            self.process_model.forward(
                self.x, action, 
                initial_storage=self.storage,
                electricity_price=price, 
                use_discrete_model=use_discrete_model,
                idx_discrete_model=idx_discrete_model)

        # update current state
        self.x = x[-1,:]
        self.storage = storage[-1]
        self.num_steps += 1

        # get observation
        x_np = self.x.cpu().numpy() if isinstance(self.x, torch.Tensor) else self.x
        storage_np = self.storage.cpu().numpy() if isinstance(self.storage, torch.Tensor) else np.array(self.storage)
        price_slice = self.electricity_price[self.num_steps:self.num_steps+self.price_prediction_horizon['ctrl_steps']]
        if isinstance(price_slice, torch.Tensor):
            price_slice = price_slice.cpu().numpy()
        observation = np.concatenate((
            x_np,
            storage_np.reshape(1),
            price_slice
        ))

        # calculate cost reward
        cost_reward = (steady_state_cost - cost) * self.cost_reward_scaling_factor

        # check for constraint violations
        # calculates the total penalty based on how much self.state and self.storage deviate from their allowed ranges
        # as a quadratic function =-1 when in the middle of the range and =0 at the bounds.
        # Ensure all operands are numpy arrays for state_violation calculation
        x_np = self.x.cpu().numpy() if isinstance(self.x, torch.Tensor) else self.x
        state_mid_np = self.state_mid.cpu().numpy() if hasattr(self.state_mid, 'cpu') else self.state_mid
        min_state_np = self.min_state.cpu().numpy() if hasattr(self.min_state, 'cpu') else self.min_state
        state_violation   = (x_np - state_mid_np)**2 / (state_mid_np - min_state_np)**2 - 1
        storage_violation = (self.storage - 0.5*self.storage_size)**2 / (0.5*self.storage_size)**2 - 1
        # Ensure violations are numpy arrays (on CPU)
        if isinstance(state_violation, torch.Tensor):
            state_violation = state_violation.cpu().numpy()
        if isinstance(storage_violation, torch.Tensor):
            storage_violation = storage_violation.cpu().numpy()
        penalty = np.mean(np.maximum(
            np.concatenate([
                state_violation.reshape(-1),
                storage_violation.reshape(-1)]),
            0))
        violation_bool = penalty > 0.0

        # calculate total reward
        reward  = cost_reward - penalty\
            - violation_bool * self.constant_constr_viol_penalty\
            + self.step_reward
        if isinstance(reward, torch.Tensor):
            reward = reward.cpu().item()
        reward = np.clip(reward, self.reward_range[0], self.reward_range[1])

        # check if episode should end
        
        # check if the allowed state violation is exceeded
        
        # check if max penalty has been reached
        terminated = True\
            if max(state_violation) > self.terminate_episode_at_relative_constr_viol\
            else False
        # check if max episode length has been reached
        truncated = True if self.num_steps >= self.episode_length['ctrl_steps'] else False
        self.done = terminated or truncated


        info = {'reward': {'total': reward,'cost': cost_reward},
                'constraint_violation': violation_bool,
                'penalty': penalty,
                'true_cost': cost,
                'steady_state_cost': steady_state_cost,
                'X': x,
                'storage': storage,}
        return observation, reward, terminated, truncated, info

    def reset(self,
              x_initialization = None,
              storage_initialization = 'random',
              train_or_test = None,
              seed = None
              ):
        # set seed
        super().reset(seed=seed)

        # get reset type
        train_or_test = self.default_price_reset_type\
            if train_or_test is None\
            else train_or_test
        x_initialization = self.default_state_reset_type\
            if x_initialization is None\
            else x_initialization

        # reset environment
        self.num_steps = 0
        self.storage = self.process_model.get_initial_storage(storage_initialization)
        self.electricity_price = self.price_generator.generate_timeseries(
            length=self.episode_length['hours']+self.price_prediction_horizon['hours'],
            train_or_test=train_or_test).repeat(self.ctrl_steps_per_hour)
        self.done = False
        if x_initialization == 'steady_state' or self.process_model.initial_state_memory is None: # TODO Daniel: this is really ugly, find a better solution
            self.x = self.process_model.get_initial_state(x_initialization)
        elif x_initialization == 'from_state_memory':
            self.x, self.storage = self.process_model.get_initial_state(x_initialization)
        # Convert state and storage to tensor on the correct device if needed
        if isinstance(self.x, np.ndarray):
            self.x = torch.tensor(self.x, dtype=torch.float32, device=self.device)
        if isinstance(self.storage, np.ndarray):
            self.storage = torch.tensor(self.storage, dtype=torch.float32, device=self.device)
        # get observation and info
        x_np = self.x.cpu().numpy() if isinstance(self.x, torch.Tensor) else self.x
        storage_np = self.storage.cpu().numpy() if isinstance(self.storage, torch.Tensor) else np.array(self.storage)
        price_slice = self.electricity_price[self.num_steps:self.num_steps+self.price_prediction_horizon['ctrl_steps']]
        if isinstance(price_slice, torch.Tensor):
            price_slice = price_slice.cpu().numpy()
        observation = np.concatenate((
            x_np,
            storage_np.reshape(1),
            price_slice
        ))
        info = {}
        return observation, info