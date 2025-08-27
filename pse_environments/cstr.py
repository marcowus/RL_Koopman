# NOTE: If you load, deepcopy, or assign a model/ensemble, always call move_model_to_device(model, device) on both the ensemble and each submodel.
import numpy as np
from scipy.integrate import solve_ivp
import math
import gymnasium as gym
import random
import warnings
import torch as torch_
from common import get_default_device

if __name__ == "__main__":
    import common
else:
    try: # TODO: find a real solution for this
        from . import common
    except:
        import common

# CSTR model based on: Baader et al. “Dynamic Ramping for Demand Response of Processes and Energy Systems Based on Exact Linearization.”
# simplifying assumption: coolant flowrate Fc is a control input and is proportional to consumed electricity

def cstr1_ode(t, x0, p, control_input):
    c, T = x0
    roh, Fc = control_input
    dcdt = (1-c)*roh/p['V'] - c*p['k']*math.exp(-p['N']/T)
    dTdt = (p['T_f']-T)*roh/p['V'] + c*p['k']*math.exp(-p['N']/T) - Fc*p['alpha_c']*(T-p['T_c'])
    return [dcdt, dTdt]

def solve_IVP1(delta_t, x0, control_input, p):
    # solve IVP
    t_final = delta_t['control']
    t_eval = np.arange(start=0, stop=t_final+delta_t['timestep'], step=delta_t['timestep'])
    # Ensure x0 and control_input are numpy arrays on CPU
    if isinstance(x0, torch_.Tensor):
        x0 = x0.cpu().numpy()
    if isinstance(control_input, torch_.Tensor):
        control_input = control_input.cpu().numpy()
    solution = solve_ivp(cstr1_ode, t_span=[0,t_final], y0=x0, t_eval=t_eval, args=(p, control_input))
    assert solution.success

    # retrieve solution
    c, T = solution.y[0], solution.y[1]
    x = np.concatenate((c.reshape(-1,1), T.reshape(-1,1)), axis=1)
    return x


class CSTR1():
    def __init__(self,
                 delta_t,
                 normalize_state = True,
                 normalize_action = True,
                 discrete_model = None,
                 device=None,
                 param_overrides=None
                 ):
        self.delta_t = delta_t
        self.normalize_state = normalize_state
        self.normalize_action = normalize_action
        self.device = device if device is not None else get_default_device()  # Store device

        self.state_names = ['c', 'T']
        self.control_names = ['roh', 'Fc']
        self.n_states = 2
        self.n_actions = 2

        if normalize_action:
            low = np.array([-1.0, -1.0])
            high = np.array([1.0, 1.0])
            self.action_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
            self.action_scaler = common.ElementwiseScaler(
                min_unscaled = torch_.tensor([0.8/(60*60), 0.0/(60*60)], device=self.device),
                max_unscaled = torch_.tensor([1.2/(60*60), 700.0/(60*60)], device=self.device),
                min_scaled = torch_.tensor([-1.0, -1.0], device=self.device),
                max_scaled = torch_.tensor([1.0, 1.0], device=self.device),
                device=self.device)  # Pass device
        else:
            low = np.array([0.8/(60*60), 0.0/(60*60)])
            high = np.array([1.2/(60*60), 700.0/(60*60)])
            self.action_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

        if normalize_state:
            self.state_scaler = common.ElementwiseScaler(
                min_unscaled = torch_.tensor([0.9*0.1367, 0.8*0.7293], device=self.device),  # c_lower, T_lower
                max_unscaled = torch_.tensor([1.1*0.1367, 1.2*0.7293], device=self.device),  # c_upper, T_upper
                min_scaled = torch_.tensor([-1.0, -1.0], device=self.device),
                max_scaled = torch_.tensor([1.0, 1.0], device=self.device),
                device=self.device)  # Pass device
            self.x_bounds = {'lower':torch_.tensor([-1.0, -1.0], device=self.device),        # c_lower, T_lower
                             'upper':torch_.tensor([1.0, 1.0], device=self.device)}          # c_upper, T_upper
        else:
            self.x_bounds = {'lower':torch_.tensor([0.9*0.1367, 0.8*0.7293], device=self.device),      # c_lower, T_lower
                             'upper':torch_.tensor([1.1*0.1367, 1.2*0.7293], device=self.device)}      # c_upper, T_upper
        self.x_range = [self.x_bounds['upper'][0]-self.x_bounds['lower'][0],
                        self.x_bounds['upper'][1]-self.x_bounds['lower'][1]]

        # mechanistic model
        self.cstr1_ode = cstr1_ode
        self.solve_IVP = solve_IVP1
        self.cstr_parameters = {
            'V': 20.0,                     # constant parameters
            'k': 300.0/(60.0*60.0),
            'N': 5.0,
            'T_f': 0.3947,
            'alpha_c': 1.95e-04,
            'T_c': 0.3816,
            'x_SS': {'c': 0.1367,          # steady state point
                     'roh': 1.0/(60.0*60.0),
                     'T': 0.7293,
                     'Fc': 390.0/(60.0*60.0)}}
        if param_overrides:
            self.cstr_parameters.update(param_overrides)
        
        # stuff needed for model-based RL algorithms
        self.discrete_model = discrete_model
        self.initial_state_memory = None

        # settings for StabilizeStateEnvironment
        self.settingsStabilizeStateEnvironment =\
            {'index_externally_forced_control': [0], # roh is the externally forced control
             'new_external_control_every_n_steps': 8,
             'x_setpoint': torch_.tensor([0.0,0.0], device=self.device)\
                if self.normalize_state\
                else torch_.tensor([0.1367,0.7293], device=self.device),}
        # settings for SetPointTrackingEnvironment
        self.settingsSetPointTrackingEnvironment =\
            {"index_tracked_state":0,
             "new_setpoint_value_every_n_steps": 8,
             "x_setpoint_range": torch_.tensor([-0.25,0.25], device=self.device)\
                 if self.normalize_state\
                 else torch_.tensor([0.1332825,0.1401175], device=self.device),}
    def forward(self, x, action, 
                initial_storage=None, electricity_price=None,
                use_discrete_model=False, idx_discrete_model=None):
        # unscale state
        if self.normalize_state and not use_discrete_model:
            x = self.state_scaler.unscale(x)

        # unscale action
        if self.normalize_action and not use_discrete_model:
            action = self.action_scaler.unscale(action)

        # solve the initial value problem given by the current state and the chosen action
        if use_discrete_model:
            assert self.discrete_model is not None
            # Ensure x is a torch tensor on the correct device for dream model
            if isinstance(x, np.ndarray):
                x = torch_.tensor(x, dtype=torch_.float32, device=self.device)
            elif isinstance(x, torch_.Tensor) and x.device != self.device:
                x = x.to(self.device)
            x = self.discrete_model.env_step(self.delta_t, x, action, idx_discrete_model)
        else:
            x = self.solve_IVP(self.delta_t, x, action, self.cstr_parameters)

        # unscale action in case this was not done before for storage and cost calculation
        if not (self.normalize_action and not use_discrete_model):
            action = self.action_scaler.unscale(action)

        # get new storage level (unit: hours of steady-state production)
        if initial_storage is None:
            storage = None
        else:
            storage_increase_per_hour = (action[0] - self.cstr_parameters['x_SS']['roh']) / self.cstr_parameters['x_SS']['roh']
            storage_increase_per_control = storage_increase_per_hour * self.delta_t['control']/3600.0
            start = initial_storage
            stop = initial_storage + storage_increase_per_control
            if isinstance(start, torch_.Tensor):
                start = start.cpu().numpy()
            if isinstance(stop, torch_.Tensor):
                stop = stop.cpu().numpy()
            storage = np.linspace(start=start,
                                  stop=stop,
                                  num=int(self.delta_t['control']/self.delta_t['timestep']+1),
                                  endpoint=True)

        # calculate cost
        if electricity_price is None:
            cost, steady_state_cost = None, None
        else:
            cost = electricity_price * action[1] * self.delta_t['control']
            steady_state_cost = electricity_price * self.cstr_parameters['x_SS']['Fc'] * self.delta_t['control']

        # scale state
        if self.normalize_state and not use_discrete_model:
            x = self.state_scaler.scale(x)

        return x, storage, cost, steady_state_cost

    def get_initial_state(self, x_initialization='steady_state'):
        assert x_initialization in ['steady_state', 'from_state_memory'],\
            "x_initialization must be 'steady_state' or 'from_state_memory'"

        x = torch_.tensor([self.cstr_parameters['x_SS']['c'], self.cstr_parameters['x_SS']['T']], device=self.device)
        if self.normalize_state:
            x = self.state_scaler.scale(x)

        if x_initialization == 'from_state_memory':
            if self.initial_state_memory is None:
                warnings.warn("initial_state_memory is None, using steady-state instead to initialize dream-train environment. This is normal at the beginning of training but should not happen later on.")
            else:
                x, storage = self.initial_state_memory.get_random_initial_state(as_tensor=False)
                return x, storage
        return x

    def get_initial_storage(self, storage_initialization='empty',
                            random_limits=[1.0, 2.0]):
        if storage_initialization == 'empty':
            storage = 0.0
        elif storage_initialization == 'random':
            storage = np.random.uniform(low=random_limits[0], high=random_limits[1])
        elif type(storage_initialization) == float:
            storage = storage_initialization
        else:
            raise ValueError('storage_initialization must be "empty", "random" or a float')
        return storage


class CSTR1Env(gym.Env):
    """Gymnasium environment wrapper for :class:`CSTR1`.

    The environment exposes the two normalized states (concentration ``c`` and
    temperature ``T``) as observations and expects two normalized control
    inputs.  Rewards are set to zero by default – the class only models the
    system dynamics.  Episodes terminate after ``episode_length`` steps.
    """

    metadata = {"render_modes": []}

    def __init__(self, delta_t, normalize_state=True, normalize_action=True,
                 episode_length=200, device=None, param_overrides=None,
                 c_target: float = 0.6):
        super().__init__()
        self.model = CSTR1(delta_t, normalize_state=normalize_state,
                           normalize_action=normalize_action, device=device,
                           param_overrides=param_overrides)
        self.action_space = self.model.action_space
        if normalize_state:
            low = -np.ones(self.model.n_states, dtype=np.float32)
            high = np.ones(self.model.n_states, dtype=np.float32)
        else:
            low = self.model.x_bounds['lower'].cpu().numpy()
            high = self.model.x_bounds['upper'].cpu().numpy()
        self.observation_space = gym.spaces.Box(low=low, high=high,
                                                dtype=np.float32)
        self.episode_length = episode_length
        self.device = self.model.device
        self.state = None
        self.steps = 0
        self.c_target = c_target

    def step(self, action):
        if isinstance(action, np.ndarray):
            action = torch_.tensor(action, dtype=torch_.float32,
                                   device=self.device)
        x, _, _, _ = self.model.forward(self.state, action)
        self.state = x[-1]
        self.steps += 1
        obs = (self.state.cpu().numpy() if isinstance(self.state, torch_.Tensor)
               else self.state)
        # compute reward based on concentration error
        if self.model.normalize_state:
            c_current = (
                self.model.state_scaler.unscale(self.state)[0].item()
                if isinstance(self.state, torch_.Tensor)
                else self.model.state_scaler.unscale(torch_.tensor(self.state))[0].item()
            )
        else:
            c_current = (
                self.state[0].item() if isinstance(self.state, torch_.Tensor)
                else self.state[0]
            )
        error = c_current - self.c_target
        reward = -(error ** 2)
        terminated = False
        truncated = self.steps >= self.episode_length
        info = {"c": c_current}
        return obs, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self.model.get_initial_state()
        self.steps = 0
        obs = (self.state.cpu().numpy() if isinstance(self.state, torch_.Tensor)
               else self.state)
        info = {}
        return obs, info

### Testing
if __name__ == '__main__':
    cstr = CSTR1(delta_t = {'timestep':15*60.0, 'control':60*60.0},
                 normalize_action=True,
                 normalize_state=True)

    x = cstr.get_initial_state(x_initialization='steady_state')
    storage = cstr.get_initial_storage(storage_initialization='random')

    #action = np.array([1.0/(60*60), 390.0/(60*60)])
    action = torch_.tensor([0.0, 2*390.0/700-1], device=cstr.device)
    x, storage, cost, steady_state_cost = cstr.forward(x, action, storage)
    x = x[-1,:]
    storage = storage[-1]
    for _ in range(1000):
        x, storage, cost, steady_state_cost = cstr.forward(x, action, storage)
        x = x[-1,:]
        storage = storage[-1]

    print('x =', x)
    print('storage =', storage)
    print('cost =', cost)
    print('steady_state_cost =', steady_state_cost)

    # timing
    x = cstr.get_initial_state(x_initialization='steady_state')
    storage = cstr.get_initial_storage(storage_initialization='empty')
    import time
    start = time.time()
    for _ in range(1000):
        action = torch_.rand(2, device=cstr.device)
        _ = cstr.forward(x, action, storage)
    print('time =', time.time()-start)