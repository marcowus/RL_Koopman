import numpy as np
import torch as T
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy
from gymnasium import spaces
from typing import Callable, Tuple
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
from timeit import default_timer as timer
import warnings

# local imports
from common import weight_reset_torch_module, reset_torch_parameter

# NOTE: If you load, deepcopy, or assign a model/ensemble, always call move_model_to_device(model, device) on both the ensemble and each submodel.
# Add runtime assertions in forward/forward_actor to catch device mismatches.
def customize_agent(model, env, settings):
    # Set constant log_std exploration
    if settings['MFRL_algorithm'] == 'PPO' and settings['action_std'] is not None:
        model.policy.log_std = T.nn.Parameter(
            T.ones(env.action_space.shape[0], device=settings['device']) * np.log(settings['action_std']),
            requires_grad=False)

    # Disable the action net output layer that SB3 adds if MPC policy is used
    if settings['policy_type'] in ['LearnableKoopmanMPC', 'LearnableBoundsMPC']:
        action_net = T.nn.Linear(
            env.action_space.shape[0],
            env.action_space.shape[0])
        action_net.weight.data = T.eye(env.action_space.shape[0])
        action_net.weight.requires_grad = False
        action_net.bias.data = T.zeros(env.action_space.shape[0])
        action_net.bias.requires_grad = False
        action_net = action_net.to(next(model.policy.parameters()).device)
        model.policy.action_net = action_net

    # Reset the weights of the dynamic Koopman model
    if settings['policy_type'] in ['LearnableKoopmanMPC', 'LearnableBoundsMPC']\
    and settings['init_Koopman_model'] == 'random':
        model.policy.mlp_extractor.reset_koopman_model()

    return model

### CSTR1 MLP
def get_actor_critic_layers_CSTR1(feature_dim, hidden_layer_dim):
    state_layer_dim = int(0.375 * hidden_layer_dim)
    storage_layer_dim = int(0.125 * hidden_layer_dim)
    initial_price_and_range_layer_dim = int(0.125 * hidden_layer_dim)
    prices_layer_dim = hidden_layer_dim - state_layer_dim - storage_layer_dim - initial_price_and_range_layer_dim
    
    layers = nn.ModuleDict({
        'state_input': nn.Linear(2, state_layer_dim),
        'storage_input': nn.Linear(1, storage_layer_dim),
        'initial_price_and_range_input': nn.Linear(2, initial_price_and_range_layer_dim),
        'prices_input': nn.Linear(int(feature_dim)-3, prices_layer_dim),
        'state_fc1': nn.Linear(state_layer_dim, state_layer_dim),
        'storage_fc1': nn.Linear(storage_layer_dim, storage_layer_dim),
        'initial_price_and_range_fc1': nn.Linear(initial_price_and_range_layer_dim, initial_price_and_range_layer_dim),
        'prices_fc1': nn.Linear(prices_layer_dim, prices_layer_dim),

        'fc1': nn.Linear(hidden_layer_dim, hidden_layer_dim),
        'fc2': nn.Linear(hidden_layer_dim, hidden_layer_dim),
    })
    return layers

class PPO_ac_CSTR1(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the features extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
        self,
        feature_dim: int,
        last_layer_dim_pi: int = 64,
        last_layer_dim_vf: int = 64,
    ):
        super().__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        self.pi_layers = get_actor_critic_layers_CSTR1(feature_dim, last_layer_dim_pi)
        self.vf_layers = get_actor_critic_layers_CSTR1(feature_dim, last_layer_dim_vf)

    def forward(self, features: T.Tensor) -> Tuple[T.Tensor, T.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        features = features.to(next(self.parameters()).device)
        # Runtime assertion for device consistency
        assert features.device == next(self.parameters()).device, "Input and model are on different devices!"
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: T.Tensor) -> T.Tensor:
        features = features.to(next(self.parameters()).device)
        # Runtime assertion for device consistency
        assert features.device == next(self.parameters()).device, "Input and model are on different devices!"
        x = self.forward_through_actor_critic_layers(features, self.pi_layers)
        return x

    def forward_critic(self, features: T.Tensor) -> T.Tensor:
        features = features.to(next(self.parameters()).device)
        x = self.forward_through_actor_critic_layers(features, self.vf_layers)
        return x

    def forward_through_actor_critic_layers(self, features, layers):
            # extract different parts of the current environment state
            storage = features[:,2].unsqueeze(1)
            initial_price = features[:,3].unsqueeze(1)
            prices = features[:,3:]
            state = features[:,:2]
            price_range = prices.max(dim=1, keepdim=True)[0] - prices.min(dim=1, keepdim=True)[0]
            assert (price_range > 0.0).all() > 0.0

            # normalize prices
            initial_price_and_range = T.cat((initial_price, price_range), dim=1)
            prices = (prices - initial_price) / price_range

            # pass through network
            state = T.tanh(layers['state_input'](state))
            storage = T.tanh(layers['storage_input'](storage))
            initial_price_and_range = T.tanh(layers['initial_price_and_range_input'](initial_price_and_range))
            prices = T.tanh(layers['prices_input'](prices))

            state = T.tanh(layers['state_fc1'](state))
            storage = T.tanh(layers['storage_fc1'](storage))
            initial_price_and_range = T.tanh(layers['initial_price_and_range_fc1'](initial_price_and_range))
            prices = T.tanh(layers['prices_fc1'](prices))

            x = T.concat((state,storage,initial_price_and_range,prices), dim=1)
            x = T.tanh(layers['fc1'](x))
            x = T.tanh(layers['fc2'](x))
            return x

class PPO_CustomMLP_CSTR1(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        *args,
        **kwargs,
    ):
        # Disable orthogonal initialization
        kwargs["ortho_init"] = False
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = PPO_ac_CSTR1(self.features_dim)

### Distilation MLP
def get_actor_critic_layers_Distilation(feature_dim, hidden_layer_dim):
    state_layer_dim = int(0.375 * hidden_layer_dim)
    storage_layer_dim = int(0.125 * hidden_layer_dim)
    initial_price_and_range_layer_dim = int(0.125 * hidden_layer_dim)
    prices_layer_dim = hidden_layer_dim - state_layer_dim - storage_layer_dim - initial_price_and_range_layer_dim
    
    layers = nn.ModuleDict({
        'state_input': nn.Linear(32, state_layer_dim),
        'storage_input': nn.Linear(1, storage_layer_dim),
        'initial_price_and_range_input': nn.Linear(2, initial_price_and_range_layer_dim),
        'prices_input': nn.Linear(int(feature_dim)-(32+1), prices_layer_dim),
        'state_fc1': nn.Linear(state_layer_dim, state_layer_dim),
        'storage_fc1': nn.Linear(storage_layer_dim, storage_layer_dim),
        'initial_price_and_range_fc1': nn.Linear(initial_price_and_range_layer_dim, initial_price_and_range_layer_dim),
        'prices_fc1': nn.Linear(prices_layer_dim, prices_layer_dim),

        'fc1': nn.Linear(hidden_layer_dim, hidden_layer_dim),
        'fc2': nn.Linear(hidden_layer_dim, hidden_layer_dim),
    })
    return layers


### CSTR1 LearnableKoopmanMPC
def get_CSTR1_LearnableKoopmanMPC_optlayer(Az):
    starttime = timer()

    # settings
    nominal_production = 0.0
    num_timesteps = int( 9*4 )
    n_const_cntrl = int( 4 )
    num_x, num_z, num_u = 2, Az.shape[0], 2
    
    # get cvxpy parameters to set up optimization problem
    Az_cp = cp.Parameter(shape=(num_z,num_z))
    Au_cp = cp.Parameter(shape=(num_z,num_u))
    ZtoX_cp = cp.Parameter(shape=(num_x,num_z))
    z_init_cp = cp.Parameter(shape=(num_z,))
    storage_init_cp = cp.Parameter(shape=(1,))
    prices = cp.Parameter(shape=(num_timesteps,))
    
    # variables
    Z = dict()                              # latent space variables
    U = dict()                              # control variables
    storage = dict()                        # quantity of stored product
    for t in range(num_timesteps):
        Z[t] = cp.Variable(shape=num_z)
    for t in range(num_timesteps-1):
        U[t] = cp.Variable(shape=num_u)
    for t in range(num_timesteps):
        storage[t] = cp.Variable(shape=1)
    M_slack = np.array([10_000.0, 10_000.0]) # M_soft_constraints
    X_slack = dict()
    for t in range(num_timesteps):
        X_slack[t] = cp.Variable(shape=num_x, nonneg=True)
    M_storage_slack = 10_000.0               # M_storage
    storage_slack = dict()
    for t in range(num_timesteps):
        storage_slack[t] = cp.Variable(shape=1, nonneg=True)

    # constraints
    constraints = []
    # initial state
    constraints.append(Z[0] == z_init_cp)
    constraints.append(storage[0] == storage_init_cp)

    # upper and lower bounds of X
    for t in range(num_timesteps):
        constraints.append( ZtoX_cp @ Z[t] + X_slack[t] >= np.array([-1., -1.]) )
        constraints.append( ZtoX_cp @ Z[t] - X_slack[t] <= np.array([1., 1.]) )

    # upper and lower bounds of U
    for t in range(num_timesteps-1):
        constraints.append( U[t] >= np.array([-1., -1.]) )
        constraints.append( U[t] <= np.array([1., 1.]) )

    # constraints to enforce that U only changes every n timesteps
    for t in range(1,num_timesteps-1):
        if t % n_const_cntrl != 0:
            constraints.append( U[t] == U[t-1] )

    # system evolution constraints
    for t in range(1,num_timesteps):
        constraints.append( Az_cp @ Z[t-1] + Au_cp @ U[t-1] == Z[t] )

    # storage evolution constraints
    for t in range(1,num_timesteps):
        constraints.append( storage[t] == storage[t-1] +\
                            (U[t-1][0] - nominal_production) / n_const_cntrl * 0.2 )

    # upper and lower bounds of storage
    for t in range(1,num_timesteps):
        constraints.append( storage[t] + storage_slack[t] >= 0.0 )
        constraints.append( storage[t] - storage_slack[t] <= 6.0 )
    constraints.append( storage[num_timesteps-1] + storage_slack[num_timesteps-1] >= 1.0 ) # target_storage == 1.0

    # set objective
    objective = sum( (U[t][1]+1.0) * prices[t] for t in range(num_timesteps-1) )
    # add quadratic slack penalty
    objective += sum( X_slack[t]**2 @ M_slack for t in range(num_timesteps) )
    objective += sum( storage_slack[t]**2 * M_storage_slack for t in range(num_timesteps) )
    # minimize objective
    objective = cp.Minimize(objective)

    # formulate the problem
    prob = cp.Problem(objective, constraints)

    # make list of parameters w.r.t. which the solution of the optimization will be differentiated
    parameters = [Az_cp, Au_cp, ZtoX_cp, z_init_cp, storage_init_cp, prices]

    # create the PyTorch interface
    optlayer = CvxpyLayer(prob,
                          parameters=parameters,
                          variables=[U[0]])

    optlayer.num_timesteps = num_timesteps
    optlayer.num_x = num_x
    optlayer.num_z = num_z
    optlayer.num_u = num_u
    optlayer.n_const_cntrl = n_const_cntrl

    time = round((timer()-starttime), 2)
    print('\nTime taken to set up economic OptLayer for MPC Policy: ' + str(time) + ' seconds\n')
    return optlayer

class PPO_ac_CSTR1_LearnableKoopmanMPC(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the features extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
        self,
        feature_dim: int,
        last_layer_dim_pi: int = 2,
        last_layer_dim_vf: int = 64,
    ):
        super().__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        self.load_CSTR1_koopman_model(
            path='./pretrained_dynamic_models/CSTR1/Koopman_8_model')
        self.optlayer = get_CSTR1_LearnableKoopmanMPC_optlayer(self.Az)
        self.vf_layers = get_actor_critic_layers_CSTR1(feature_dim, last_layer_dim_vf)

    def load_CSTR1_koopman_model(self, path):
        self.koopman_model = T.load(path)
        # Move loaded model to the same device as this module
        device = self.device if hasattr(self, 'device') else next(self.parameters()).device
        self.koopman_model = self.koopman_model.to(device)
        self.set_parameters_to_current_koopman_model()
        return None

    def set_koopman_model(self, koopman_model):
        self.koopman_model = koopman_model
        self.set_parameters_to_current_koopman_model()
        return None

    def reset_koopman_model(self):
        self.koopman_model.apply(weight_reset_torch_module)
        self.set_parameters_to_current_koopman_model()
        return None

    def set_parameters_to_current_koopman_model(self):
        device = self.device if hasattr(self, 'device') else next(self.parameters()).device
        self.Az = self.koopman_model.Az.weight.to(device)
        self.Au = self.koopman_model.Au.weight.to(device)
        self.ZtoX = self.koopman_model.decoder.weight.to(device)
        self.XtoZ = self.koopman_model.encoder.to(device)
        return None

    def forward(self, features: T.Tensor) -> Tuple[T.Tensor, T.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        features = features.to(next(self.parameters()).device)
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: T.Tensor) -> T.Tensor:
        features = features.to(next(self.parameters()).device)
        prices = features[:,3:]
        storage = features[:,2].unsqueeze(1)
        state = features[:,:2]

        # get initial latent space state
        z_init = self.XtoZ(state)

        # solve mpc step
        try:
            mu = self.optlayer(self.Az, self.Au, self.ZtoX,
                               z_init, storage, 
                               T.repeat_interleave(prices, repeats=self.optlayer.n_const_cntrl, dim=1))[0]
        except:
            warnings.warn("Fallback to SCS solver in optlayer because specified solver errored.")
            mu = self.optlayer(self.Az, self.Au, self.ZtoX,
                               z_init, storage, 
                               T.repeat_interleave(prices, repeats=self.optlayer.n_const_cntrl, dim=1),
                               solver_args={'solve_method':'SCS'})[0]
        return mu

    def forward_critic(self, features: T.Tensor) -> T.Tensor:
        features = features.to(next(self.parameters()).device)
        x = self.forward_through_critic_layers(features, self.vf_layers)
        return x

    def forward_through_critic_layers(self, features, layers):
            # extract different parts of the current environment state
            storage = features[:,2].unsqueeze(1)
            initial_price = features[:,3].unsqueeze(1)
            prices = features[:,3:]
            state = features[:,:2]
            price_range = prices.max(dim=1, keepdim=True)[0] - prices.min(dim=1, keepdim=True)[0]
            assert (price_range > 0.0).all() > 0.0

            # normalize prices
            initial_price_and_range = T.cat((initial_price, price_range), dim=1)
            prices = (prices - initial_price) / price_range

            # pass through network
            state = T.tanh(layers['state_input'](state))
            storage = T.tanh(layers['storage_input'](storage))
            initial_price_and_range = T.tanh(layers['initial_price_and_range_input'](initial_price_and_range))
            prices = T.tanh(layers['prices_input'](prices))

            state = T.tanh(layers['state_fc1'](state))
            storage = T.tanh(layers['storage_fc1'](storage))
            initial_price_and_range = T.tanh(layers['initial_price_and_range_fc1'](initial_price_and_range))
            prices = T.tanh(layers['prices_fc1'](prices))

            x = T.concat((state,storage,initial_price_and_range,prices), dim=1)
            x = T.tanh(layers['fc1'](x))
            x = T.tanh(layers['fc2'](x))
            return x

class PPO_LearnableKoopmanMPC_CSTR1(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        *args,
        **kwargs,
    ):
        # Disable orthogonal initialization
        kwargs["ortho_init"] = False
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = PPO_ac_CSTR1_LearnableKoopmanMPC(self.features_dim)

### CSTR1 LearnableBoundsMPC
def get_CSTR1_LearnableBoundsMPC_optlayer(Az, Au, ZtoX):
    # settings
    nominal_production = 0.0
    num_timesteps = int( 9*4 )
    n_const_cntrl = int( 4 )
    num_x, num_z, num_u = 2, Az.shape[0], 2

    # get cvxpy parameters to set up optimization problem
    z_init_cp = cp.Parameter(shape=(num_z,))
    storage_init_cp = cp.Parameter(shape=(1,))
    prices = cp.Parameter(shape=(num_timesteps,))
    # learnable parameters
    X_lb_delta = cp.Parameter(shape=(num_x,))
    X_ub_delta = cp.Parameter(shape=(num_x,))
    storage_lb_delta = cp.Parameter(shape=(1,))
    storage_ub_delta = cp.Parameter(shape=(1,))
    target_storage_delta = cp.Parameter(shape=(1,))

    # variables
    Z = dict()                              # latent space variables
    U = dict()                              # control variables
    storage = dict()                        # quantity of stored product
    for t in range(num_timesteps):
        Z[t] = cp.Variable(shape=num_z)
    for t in range(num_timesteps-1):
        U[t] = cp.Variable(shape=num_u)
    for t in range(num_timesteps):
        storage[t] = cp.Variable(shape=1)
    M_slack = np.array([10_000.0, 10_000.0]) # M_soft_constraints
    X_slack = dict()
    for t in range(num_timesteps):
        X_slack[t] = cp.Variable(shape=num_x, nonneg=True)
    M_storage_slack = 10_000.0               # M_storage
    storage_slack = dict()
    for t in range(num_timesteps):
        storage_slack[t] = cp.Variable(shape=1, nonneg=True)

    # constraints
    constraints = []
    # initial state
    constraints.append(Z[0] == z_init_cp)
    constraints.append(storage[0] == storage_init_cp)

    # upper and lower bounds of X
    for t in range(num_timesteps):
        constraints.append( ZtoX @ Z[t] + X_slack[t] >= np.array([-1., -1.]) + X_lb_delta )
        constraints.append( ZtoX @ Z[t] - X_slack[t] <= np.array([1., 1.]) + X_ub_delta )

    # upper and lower bounds of U
    for t in range(num_timesteps-1):
        constraints.append( U[t] >= np.array([-1., -1.]) )
        constraints.append( U[t] <= np.array([1., 1.]) )

    # constraints to enforce that U only changes every n timesteps
    for t in range(1,num_timesteps-1):
        if t % n_const_cntrl != 0:
            constraints.append( U[t] == U[t-1] )

    # system evolution constraints
    for t in range(1,num_timesteps):
        constraints.append( Az @ Z[t-1] + Au @ U[t-1] == Z[t] )

    # storage evolution constraints
    for t in range(1,num_timesteps):
        constraints.append( storage[t] == storage[t-1] +\
                            (U[t-1][0] - nominal_production) / n_const_cntrl * 0.2 )

    # upper and lower bounds of storage
    for t in range(1,num_timesteps):
        constraints.append( storage[t] + storage_slack[t] >= 0.0 + storage_lb_delta )
        constraints.append( storage[t] - storage_slack[t] <= 6.0 + storage_ub_delta )
    constraints.append( storage[num_timesteps-1] + storage_slack[num_timesteps-1] >= 1.0 + target_storage_delta )

    # set objective
    objective = sum( (U[t][1]+1.0) * prices[t] for t in range(num_timesteps-1) )
    # add quadratic slack penalty
    objective += sum( X_slack[t]**2 @ M_slack for t in range(num_timesteps) )
    objective += sum( storage_slack[t]**2 * M_storage_slack for t in range(num_timesteps) )
    # minimize objective
    objective = cp.Minimize(objective)

    # formulate the problem
    prob = cp.Problem(objective, constraints)

    # make list of parameters w.r.t. which the solution of the optimization will be differentiated
    parameters = [z_init_cp, storage_init_cp, prices,
                  X_lb_delta, X_ub_delta, storage_lb_delta,
                  storage_ub_delta, target_storage_delta]

    # create the PyTorch interface
    optlayer = CvxpyLayer(prob,
                          parameters=parameters,
                          variables=[U[0]])

    optlayer.num_timesteps = num_timesteps
    optlayer.num_x = num_x
    optlayer.num_z = num_z
    optlayer.num_u = num_u
    optlayer.n_const_cntrl = n_const_cntrl

    return optlayer

class PPO_ac_CSTR1_LearnableBoundsMPC(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the features extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
        self,
        feature_dim: int,
        last_layer_dim_pi: int = 2,
        last_layer_dim_vf: int = 64,
    ):
        super().__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        self.load_CSTR1_koopman_model(
            path='./pretrained_dynamic_models/CSTR1/Koopman_8_model')
        self.vf_layers = get_actor_critic_layers_CSTR1(feature_dim, last_layer_dim_vf)

        # learnable parameters
        self.X_lb_delta = nn.Parameter(T.zeros(2))
        self.X_ub_delta = nn.Parameter(T.zeros(2))
        self.storage_lb_delta = nn.Parameter(T.zeros(1))
        self.storage_ub_delta = nn.Parameter(T.zeros(1))
        self.target_storage_delta = nn.Parameter(T.zeros(1))

    def get_bounds_delta_from_parameters(self):
        tanh = nn.Tanh()
        X_lb_delta = tanh(self.X_lb_delta) * 0.5
        X_ub_delta = tanh(self.X_ub_delta) * 0.5
        storage_lb_delta = tanh(self.storage_lb_delta) * 0.5
        storage_ub_delta = tanh(self.storage_ub_delta) * 0.5
        target_storage_delta = tanh(self.target_storage_delta)
        return X_lb_delta, X_ub_delta, storage_lb_delta, storage_ub_delta, target_storage_delta

    def log_learnable_parameters(self, logger, x_val):
        X_lb_delta, X_ub_delta, storage_lb_delta, storage_ub_delta, target_storage_delta =\
            self.get_bounds_delta_from_parameters()

        X_lb_delta = X_lb_delta.detach().cpu().numpy()
        X_ub_delta = X_ub_delta.detach().cpu().numpy()
        storage_lb_delta = storage_lb_delta.detach().cpu().numpy()
        storage_ub_delta = storage_ub_delta.detach().cpu().numpy()
        target_storage_delta = target_storage_delta.detach().cpu().numpy()

        logger.add_scalar('MPC Learned Bounds Delta/X_lb_delta_c', X_lb_delta[0], x_val)
        logger.add_scalar('MPC Learned Bounds Delta/X_lb_delta_T', X_lb_delta[1], x_val)
        logger.add_scalar('MPC Learned Bounds Delta/X_ub_delta_c', X_ub_delta[0], x_val)
        logger.add_scalar('MPC Learned Bounds Delta/X_ub_delta_T', X_ub_delta[1], x_val)
        logger.add_scalar('MPC Learned Bounds Delta/storage_lb_delta', storage_lb_delta[0], x_val)
        logger.add_scalar('MPC Learned Bounds Delta/storage_ub_delta', storage_ub_delta[0], x_val)
        logger.add_scalar('MPC Learned Bounds Delta/target_storage_delta', target_storage_delta[0], x_val)

        return None

    def load_CSTR1_koopman_model(self, path):
        self.koopman_model = T.load(path)
        device = self.device if hasattr(self, 'device') else next(self.parameters()).device
        self.koopman_model = self.koopman_model.to(device)
        self.set_parameters_to_current_koopman_model()
        return None

    def set_koopman_model(self, koopman_model):
        self.koopman_model = koopman_model
        self.set_parameters_to_current_koopman_model()
        return None

    def reset_koopman_model(self):
        self.koopman_model.apply(weight_reset_torch_module)
        self.set_parameters_to_current_koopman_model()
        return None

    def set_parameters_to_current_koopman_model(self):
        device = self.device if hasattr(self, 'device') else next(self.parameters()).device
        self.Az = self.koopman_model.Az.weight.to(device)
        self.Au = self.koopman_model.Au.weight.to(device)
        self.ZtoX = self.koopman_model.decoder.weight.to(device)
        self.XtoZ = self.koopman_model.encoder.to(device)
        for param in self.XtoZ.parameters(): # freeze encoder
            param.requires_grad = False
        self.optlayer = get_CSTR1_LearnableBoundsMPC_optlayer(
            self.Az.detach().cpu().numpy(),
            self.Au.detach().cpu().numpy(),
            self.ZtoX.detach().cpu().numpy())
        return None

    def forward(self, features: T.Tensor) -> Tuple[T.Tensor, T.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        features = features.to(next(self.parameters()).device)
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: T.Tensor) -> T.Tensor:
        features = features.to(next(self.parameters()).device)
        prices = features[:,3:]
        storage = features[:,2].unsqueeze(1)
        state = features[:,:2]

        # get initial latent space state
        z_init = self.XtoZ(state)

        # get bounds delta
        X_lb_delta, X_ub_delta, storage_lb_delta, storage_ub_delta, target_storage_delta =\
            self.get_bounds_delta_from_parameters()

        # solve mpc step
        try:
            mu = self.optlayer(
                z_init, storage,
                T.repeat_interleave(prices, repeats=self.optlayer.n_const_cntrl, dim=1),
                X_lb_delta, X_ub_delta, storage_lb_delta,
                storage_ub_delta, target_storage_delta)[0]
        except:
            warnings.warn("Fallback to SCS solver in optlayer because specified solver errored.")
            mu = self.optlayer(
                z_init, storage,
                T.repeat_interleave(prices, repeats=self.optlayer.n_const_cntrl, dim=1),
                X_lb_delta, X_ub_delta, storage_lb_delta,
                storage_ub_delta, target_storage_delta,
                solver_args={'solve_method':'SCS'})[0]
        return mu

    def forward_critic(self, features: T.Tensor) -> T.Tensor:
        features = features.to(next(self.parameters()).device)
        x = self.forward_through_critic_layers(features, self.vf_layers)
        return x

    def forward_through_critic_layers(self, features, layers):
            # extract different parts of the current environment state
            storage = features[:,2].unsqueeze(1)
            initial_price = features[:,3].unsqueeze(1)
            prices = features[:,3:]
            state = features[:,:2]
            price_range = prices.max(dim=1, keepdim=True)[0] - prices.min(dim=1, keepdim=True)[0]
            assert (price_range > 0.0).all() > 0.0

            # normalize prices
            initial_price_and_range = T.cat((initial_price, price_range), dim=1)
            prices = (prices - initial_price) / price_range

            # pass through network
            state = T.tanh(layers['state_input'](state))
            storage = T.tanh(layers['storage_input'](storage))
            initial_price_and_range = T.tanh(layers['initial_price_and_range_input'](initial_price_and_range))
            prices = T.tanh(layers['prices_input'](prices))

            state = T.tanh(layers['state_fc1'](state))
            storage = T.tanh(layers['storage_fc1'](storage))
            initial_price_and_range = T.tanh(layers['initial_price_and_range_fc1'](initial_price_and_range))
            prices = T.tanh(layers['prices_fc1'](prices))

            x = T.concat((state,storage,initial_price_and_range,prices), dim=1)
            x = T.tanh(layers['fc1'](x))
            x = T.tanh(layers['fc2'](x))
            return x

class PPO_LearnableBoundsMPC_CSTR1(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        *args,
        **kwargs,
    ):
        # Disable orthogonal initialization
        kwargs["ortho_init"] = False
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = PPO_ac_CSTR1_LearnableBoundsMPC(self.features_dim)