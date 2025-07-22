# third party imports
import stable_baselines3 as sb3
import numpy as np
import random
import torch as T
import pickle
from copy import deepcopy

# Local imports
import custom_ac_policies
import dream_models
import memory
from pse_environments.environments import get_environment
from common import timer_decorator
import common
import math
from common import get_default_device

# Utility function to recursively move a model and all submodules to a device

def move_model_to_device(model, device):
    model = model.to(device)
    for name, module in model.named_children():
        module.to(device)
    return model

def get_mbrl_agent(settings):
    # Create the environments
    env_kwargs = deepcopy(settings[f"{settings['environment']['process']}_{settings['environment']['objective_type']}"]['env_kwargs'])
    env = {}
    # Env for collecting experience
    env_kwargs['default_price_reset_type'] = 'train'
    env_kwargs['default_state_reset_type'] = 'steady_state'
    env_kwargs['episode_length'] = settings[f"{settings['environment']['process']}_{settings['environment']['objective_type']}"]['env_kwargs']['episode_length']
    env_kwargs['default_use_discrete_model'] = False
    env['train_real'] = get_environment(
                            process=settings['environment']['process'],
                            objective_type=settings['environment']['objective_type'],
                            process_kwargs=settings[f"{settings['environment']['process']}_{settings['environment']['objective_type']}"]['process_kwargs'],
                            env_kwargs=env_kwargs, device=settings['device'])
    # Env for evaluating policy
    env_kwargs['default_price_reset_type'] = 'test'
    env_kwargs['default_state_reset_type'] = 'steady_state'
    env_kwargs['episode_length'] = settings[f"{settings['environment']['process']}_{settings['environment']['objective_type']}"]['env_kwargs']['episode_length']
    env_kwargs['default_use_discrete_model'] = False
    env['eval_real'] =  get_environment(
                            process=settings['environment']['process'],
                            objective_type=settings['environment']['objective_type'],
                            process_kwargs=settings[f"{settings['environment']['process']}_{settings['environment']['objective_type']}"]['process_kwargs'],
                            env_kwargs=env_kwargs, device=settings['device'])
    # Env for training policy in dream mode
    env_kwargs['default_price_reset_type'] = 'train'
    env_kwargs['default_state_reset_type'] = 'from_state_memory'
    env_kwargs['episode_length'] = settings['policy_training']['episode_length']
    env_kwargs['default_use_discrete_model'] = True
    env['train_dream'] =  get_environment(
                            process=settings['environment']['process'],
                            objective_type=settings['environment']['objective_type'],
                            process_kwargs=settings[f"{settings['environment']['process']}_{settings['environment']['objective_type']}"]['process_kwargs'],
                            env_kwargs=env_kwargs, device=settings['device'])
    # Env for evaluating policy in dream mode
    env_kwargs['default_price_reset_type'] = 'test'
    env_kwargs['default_state_reset_type'] = 'steady_state'
    env_kwargs['episode_length'] = settings[f"{settings['environment']['process']}_{settings['environment']['objective_type']}"]['env_kwargs']['episode_length']
    env_kwargs['default_use_discrete_model'] = True
    env['eval_dream'] =  get_environment(
                            process=settings['environment']['process'],
                            objective_type=settings['environment']['objective_type'],
                            process_kwargs=settings[f"{settings['environment']['process']}_{settings['environment']['objective_type']}"]['process_kwargs'],
                            env_kwargs=env_kwargs, device=settings['device'])
    # reset environments
    for k in env.keys():
        env[k].reset()

    # Get policy
    if settings['policy_type'] in ['CustomMLP', 'LearnableKoopmanMPC', 'LearnableBoundsMPC']:
        policy = getattr(custom_ac_policies,
                        f"{settings['MFRL_algorithm']}_{settings['policy_type']}_{settings['environment']['process']}")
    else:
        policy = settings['policy_type']

    # Create the agent
    sb3_agent = getattr(sb3, settings['MFRL_algorithm'])(
                    policy, env['train_real'],
                    verbose=0, tensorboard_log=settings['logdir']+'SB3/',
                    device=settings['device'],
                    **settings['policy_training'][f"{settings['MFRL_algorithm']}_kwargs"])
    sb3_agent = custom_ac_policies.customize_agent(sb3_agent, env['train_real'], settings)

    # Get dream model ensemble
    match settings['dream_model_type']:
        case 'MLP':
            ensemble = T.nn.ModuleList([
                dream_models.MLP(
                    in_features = env['train_real'].process_model.n_states + env['train_real'].process_model.n_actions,
                    out_features = env['train_real'].process_model.n_states,
                    **settings['MLP_dream_model_kwargs'], device=settings['device'])
                for _ in range(settings['n_dream_models'])])
        case 'PINN':
            ensemble = T.nn.ModuleList([
                dream_models.PINN(
                    **settings['PINN_dream_model_kwargs'], device=settings['device'])
                for _ in range(settings['n_dream_models'])])
            
    ensemble = dream_models.Ensemble(ensemble, device=settings['device'])
    if settings['use_pretrained_dream_models']:
        for model_idx in range(settings['n_dream_models']):
            state_dict = T.load(f"{settings['saved_dream_models_dir']}{model_idx+1}.pt", map_location=settings['device'])
            ensemble.models[model_idx].load_state_dict(state_dict)
    ensemble.ensure_all_on_device(settings['device'])
    for idx, model in enumerate(ensemble.models):
        ensemble.models[idx] = move_model_to_device(model, settings['device'])
    
    # Save cpu-copies of model ensemble in dream environments
    cpu_ensemble = deepcopy(ensemble)
    cpu_ensemble.ensure_all_on_device(settings['device'])
    for idx, model in enumerate(cpu_ensemble.models):
        cpu_ensemble.models[idx] = move_model_to_device(model, settings['device'])
    # cpu_ensemble.device = T.device('cpu')  # Replaced for GPU compatibility
    # cpu_ensemble.device = T.device('cpu')  # Replaced for GPU compatibility
    # cpu_ensemble = cpu_ensemble.to(T.device('cpu'))
    # cpu_ensemble = cpu_ensemble.to(T.device('cpu'))  # Replaced for GPU compatibility
    # cpu_ensemble = cpu_ensemble.to(T.device('cuda' if T.cuda.is_available() else 'cpu'))  # Replaced for GPU compatibility
    env['train_dream'].process_model.discrete_model = deepcopy(ensemble)
    env['train_dream'].process_model.discrete_model.ensure_all_on_device(settings['device'])
    for idx, model in enumerate(env['train_dream'].process_model.discrete_model.models):
        env['train_dream'].process_model.discrete_model.models[idx] = move_model_to_device(model, settings['device'])
    env['eval_dream'].process_model.discrete_model  = deepcopy(ensemble)
    env['eval_dream'].process_model.discrete_model.ensure_all_on_device(settings['device'])
    for idx, model in enumerate(env['eval_dream'].process_model.discrete_model.models):
        env['eval_dream'].process_model.discrete_model.models[idx] = move_model_to_device(model, settings['device'])

    # Get test data for dream models
    if not settings['skip_step']['test_dream_models']:
        with open(settings['test_data_path'], 'rb') as handle:
            dream_model_test_data = pickle.load(handle)
    else:
        dream_model_test_data = None

    # Get memory
    train_memory_size = int(settings['data_collection']['memory_size']\
                            *settings['data_collection']['train_val_ratio'])
    val_memory_size = settings['data_collection']['memory_size'] - train_memory_size
    agent_memory = {
        'train': memory.MultiStepMemory( # Memory
                    max_size=train_memory_size,
                    input_shape=env['train_real'].process_model.n_states,
                    n_actions=env['train_real'].process_model.n_actions,
                    n_step_predictions=settings['dream_model_training']['n_step_prediction_loss'],
                    device=settings['device'], save_dir=settings['memory_dir'], save_name='train',
                    settings=settings),
        'val':   memory.MultiStepMemory(
                    max_size=val_memory_size,
                    input_shape=env['train_real'].process_model.n_states,
                    n_actions=env['train_real'].process_model.n_actions,
                    n_step_predictions=settings['dream_model_training']['n_step_prediction_loss'],
                    device=settings['device'], save_dir=settings['memory_dir'], save_name='val',
                    settings=settings),
    }

    # Load precollected data into memory
    if settings['data_collection']['warm_start_memory']:
        agent_memory['train'].load_memory()
        agent_memory['val'].load_memory()
    if settings['use_precollected_data']: # TODO: deprecate this in favor of warm_start_memory
        agent_memory, env = save_precollected_data_to_memory(
            agent_memory, env, settings)

    # Get summary writer
    logger = T.utils.tensorboard.SummaryWriter(log_dir=settings['logdir'])

    # Create mbrl_agent
    mbrl_agent = MBPO(
        agent=sb3_agent,
        model_ensemble=ensemble,
        dream_model_test_data=dream_model_test_data,
        environments=env,
        memory=agent_memory,
        logger=logger,
        settings=settings,
    )
    return mbrl_agent

def save_precollected_data_to_memory(memory, env, settings):
    # get necessary variables
    state_names = env['train_real'].process_model.state_names
    control_names = env['train_real'].process_model.control_names

    # load data
    with open(settings['precollected_train_data_path'], 'rb') as handle:
        train_data = pickle.load(handle)

    # extract states (X) and actions (U)
    X = {k: train_data[k][state_names].to_numpy() for k in train_data.keys()}
    U = {k: train_data[k][control_names].to_numpy() for k in train_data.keys()}

    # normalize data
    X_normalizer = env['train_real'].process_model.state_scaler
    U_normalizer = env['train_real'].process_model.action_scaler
    X = {k: X_normalizer.scale(X[k]) for k in X.keys()}
    U = {k: U_normalizer.scale(U[k]) for k in U.keys()}
    # ensure numpy arrays
    X = {k: X[k].cpu().numpy() if isinstance(X[k], T.Tensor) else X[k] for k in X.keys()}
    U = {k: U[k].cpu().numpy() if isinstance(U[k], T.Tensor) else U[k] for k in U.keys()}

    # split data into train and val
    shuffled_indices = list(range(len(X)))
    np.random.shuffle(shuffled_indices)
    train_indices = shuffled_indices[:int(len(X)*settings['data_collection']['train_val_ratio'])]
    train_or_val = ['train' if i in train_indices else 'val' for i in range(len(X))]

    # store data in memory
    for k in X.keys():
        memory[train_or_val[k]].store_timeseries(X[k], U[k])

    # pass memory to dream training environment for state initialization in dream training
    env['train_dream'].process_model.initial_state_memory = memory['train']

    return memory, env

class MBPO():
    def __init__(self, agent, model_ensemble, dream_model_test_data,
                 environments, memory, logger, settings):
        self.agent = agent
        self.model_ensemble = model_ensemble
        self.dream_model_test_data = dream_model_test_data
        self.environments = environments
        self.memory = memory
        self.logger = logger
        self.settings = settings
        self.dream_train_log_name = f"{self.settings['MBRL_algorithm']}_{self.settings['environment']['process']}_{self.settings['environment']['objective_type']}_dream"
        self.n_real_steps = 0
        self.n_real_constr_viols = 0

    @timer_decorator
    def collect_experience(self, collect_memory_timesteps,
                           random_actions=False):
        done = True
        for _ in range(collect_memory_timesteps):
            if done:
                # randomly set action std at beginning of each episode
                self.set_action_std(
                    low=self.settings['data_collection']['action_std_range'][0],
                    high=self.settings['data_collection']['action_std_range'][1])

                # choose which memory to store in (train or val)
                train_or_val = np.random.choice(
                    ['train', 'val'],
                    p=[self.settings['data_collection']['train_val_ratio'],
                       1-self.settings['data_collection']['train_val_ratio']])
                # override random decision if any memory is empty
                if self.memory['val'].is_empty()\
                and self.settings['data_collection']['train_val_ratio'] < 1.0:
                    train_or_val = 'val'
                if self.memory['train'].is_empty():
                    train_or_val = 'train'

                # reset environment
                obs, _ = self.environments['train_real'].reset()

            # get action
            if random_actions:
                act_space = self.environments['train_real'].action_space
                # act_space = gym.spaces.Box(
                #     low=-0.2, high=0.2,
                #     shape=self.environments['train_real'].action_space)
                action = act_space.sample()
            else:
                # obs = obs.to(self.settings['device'])  # Replaced for GPU compatibility
                if hasattr(obs, 'to'):
                    obs = obs.to(self.settings['device'])
                action, _ = self.agent.predict(obs, deterministic=False)

            # step environment
            try:
                obs, _, terminated, truncated, info = self.environments['train_real'].step(action)
            except:
                self.memory[train_or_val].make_last_stored_transition_terminal()
                done = True
                continue
            done = terminated or truncated
            self.n_real_constr_viols += info['constraint_violation']

            # get states from info
            states  = info['X'][:-1,:]
            states_ = info['X'][1:,:]
            actions = np.repeat(action.reshape(1,-1),
                                repeats=states.shape[0],
                                axis=0)
            dones = np.zeros(states.shape[0], dtype=bool)
            dones[-1] = done
            storage = info['storage'][:-1] # .reshape(-1,1)

            # store transitions in memory
            for i in range(states.shape[0]):
                self.memory[train_or_val].store_transition(
                    states[i,:], actions[i,:], states_[i,:], dones[i], storage[i])

            # increase step counter
            self.n_real_steps += 1

        # pass memory to dream training environment for state initialization in dream training
        self.environments['train_dream'].process_model.initial_state_memory = self.memory['train']

        # reset action std
        self.set_action_std(value=self.settings['action_std'])

        # save memory
        self.memory['train'].save_memory()
        self.memory['val'].save_memory()

        # log to tensorboard
        self.logger.add_scalar(
            'Real Steps/n_real_constr_viols',
            self.n_real_constr_viols,
            self.n_real_steps)
        return None

    @timer_decorator
    def fit_dream_model_ensemble(self):
        # fits dream models to data in memory
        # updates self.environments['dream'].process_model.discrete_model
        # returns avg validation loss and avg number of epochs trained

        val_losses = np.zeros(self.settings['n_dream_models'])
        phy_losses = np.zeros(self.settings['n_dream_models'])
        n_epochs = np.zeros(self.settings['n_dream_models'])
        
        for i_model in range(self.settings['n_dream_models']): # TODO parallelize
            if self.settings['dream_model_type'] == 'PINN':
                val_losses[i_model], phy_losses[i_model], n_epochs[i_model] =\
                    self._fit_single_PINN_model(model_type='dream', model_idx=i_model)               
            else:    
                val_losses[i_model], n_epochs[i_model] =\
                    self._fit_single_SI_model(model_type='dream', model_idx=i_model)

        # update discrete models of dream environments
        cpu_ensemble = deepcopy(self.model_ensemble)
        cpu_ensemble.ensure_all_on_device(self.settings['device'])
        self.environments['train_dream'].process_model.discrete_model = cpu_ensemble
        self.environments['eval_dream'].process_model.discrete_model  = cpu_ensemble

        # log to tensorboard
        self.logger.add_scalar(
            'System Identification/mean_data_val_loss', 
            np.mean(val_losses),
            self.n_real_steps)
        self.logger.add_scalar(
            'System Identification/mean_physics_val_loss', 
            np.mean(phy_losses),
            self.n_real_steps)
        self.logger.add_scalar(
            'System Identification/mean_n_epochs', 
            np.mean(n_epochs),
            self.n_real_steps)
        self.logger.flush()

        return np.mean(val_losses), np.mean(n_epochs)

    def _fit_single_SI_model(self, model_type, model_idx):
        def _get_predictions_from_dream_model(model, states, actions, states_, dones):
            predictions = T.zeros_like(states_)
            pred_state = states[:,0,:]
            for i_step in range(states_.shape[1]):
                # get model prediction
                pred_state = model.forward(pred_state, actions[:,i_step,:])

                # store prediction
                predictions[:,i_step,:] = pred_state
            return predictions

        # get model to fit
        if model_type == 'dream':
            model = self.model_ensemble.models[model_idx]

        # possibly reset model parameters
        if random.random() < self.settings['dream_model_training']['reset_probability']:
            model.apply(common.weight_reset_torch_module)

        i_epoch, best_epoch = 0,0
        best_val_loss = np.inf
        while i_epoch < self.settings['dream_model_training']['max_epochs']:
            i_epoch += 1
            # training
            memory_depleted = False
            while not memory_depleted:
                # get batch
                states, actions, states_, dones, memory_depleted = \
                    self.memory['train'].get_next_random_batch(
                        self.settings['dream_model_training']['batch_size'])

                # get predictions from dream model
                predictions = _get_predictions_from_dream_model(
                    model, states, actions, states_, dones)

                # update dream model
                loss = model.loss_function(predictions, states_)
                model.optimizer.zero_grad()
                loss.backward()
                model.optimizer.step()

            # validation
            # check whether to validate
            if i_epoch % self.settings['dream_model_training']['validate_every_n_epochs'] != 0:
                continue

            # get validation training data
            states, actions, states_, dones = self.memory['val'].get_entire_memory()

            # get predictions from dream model
            predictions = _get_predictions_from_dream_model(
                model, states, actions, states_, dones)

            # calculate loss
            loss = model.loss_function(predictions, states_)

            # check if validation loss is best
            if loss < best_val_loss:
                best_val_loss = loss
                best_epoch = i_epoch
                best_model_state_dict = deepcopy(model.state_dict())

            # check if training should stop early
            if i_epoch - best_epoch >= self.settings['dream_model_training']['early_stopping_patience']:
                break

        # load best model
        model.load_state_dict(best_model_state_dict)

        # save model in self
        if model_type == 'dream':
            self.model_ensemble.models[model_idx] = model

        return best_val_loss.item(), i_epoch

    def fit_koopman_model(self):
        # fits Koopman model to data in memory

        def _get_koopman_batch_loss(model, states, actions, states_):
            device = model.settings['device'] if hasattr(model, 'settings') and 'device' in model.settings else states.device
            model = move_model_to_device(model, device)
            states = states.to(dtype=T.float32, device=device)
            actions = actions.to(dtype=T.float32, device=device)
            states_ = states_.to(dtype=T.float32, device=device)
            loss_fcn = T.nn.MSELoss()

            pred_z = model.encode(states[:,0,:])
            Z_pred = T.zeros([states_.shape[0], states_.shape[1], model.latent_dim], device=device)
            for i_step in range(states_.shape[1]):
                pred_z = model.predict(pred_z, actions[:,i_step,:])
                Z_pred[:,i_step,:] = pred_z

            ae_loss = loss_fcn(
                model.decode(model.encode(states)),
                states)
            pred_loss = loss_fcn(Z_pred, model.encode(states_))
            comb_loss = loss_fcn(model.decode(Z_pred), states_)

            total_loss = 1/3 * (ae_loss + pred_loss + comb_loss)
            return total_loss, (ae_loss, pred_loss, comb_loss)

        assert self.settings['policy_type'] in ['LearnableKoopmanMPC', 'LearnableBoundsMPC'],\
            "Koopman model can only be fit when policy type is LearnableKoopmanMPC or LearnableBoundsMPC"

        model = self.agent.policy.mlp_extractor.koopman_model
        model = move_model_to_device(model, self.settings['device'])
        for param in model.parameters(): # make all parameters trainable
            param.requires_grad = True
        optimizer = T.optim.Adam(
            model.parameters(),
            lr=self.settings['koopman_SI_training']['learning_rate'])

        i_epoch, best_epoch = 0,0
        best_val_loss = np.inf
        while i_epoch < self.settings['koopman_SI_training']['max_epochs']:
            i_epoch += 1
            # training
            memory_depleted = False
            while not memory_depleted:
                try:
                    # get experience batch
                    states, actions, states_, dones, memory_depleted = \
                        self.memory['train'].get_next_random_batch(
                            self.settings['koopman_SI_training']['batch_size'])
                except:
                    memory_depleted = True
                    self.memory['train']._reset_remember_order()
                    break

                loss, _ = _get_koopman_batch_loss(model, states, actions, states_)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
            # validation
            # check whether to validate
            if i_epoch % self.settings['koopman_SI_training']['validate_every_n_epochs'] != 0:
                continue

            # get validation training data
            states, actions, states_, dones = self.memory['val'].get_entire_memory()

            # calculate loss
            loss, (ae_loss, pred_loss, comb_loss) = _get_koopman_batch_loss(model, states, actions, states_)

            # check if validation loss is best
            if loss < best_val_loss:
                best_val_loss = loss
                best_epoch = i_epoch
                ae_loss, pred_loss, comb_loss = ae_loss.item(), pred_loss.item(), comb_loss.item()
                best_model_state_dict = deepcopy(model.state_dict())

            # check if training should stop early
            if i_epoch - best_epoch >= self.settings['koopman_SI_training']['early_stopping_patience']:
                break

        # load and save best model
        model.load_state_dict(best_model_state_dict)
        model = move_model_to_device(model, self.settings['device'])
        self.agent.policy.mlp_extractor.set_koopman_model(model)

        # log to tensorboard
        self.logger.add_scalar(
            'Koopman System Identification/total_loss', 
            best_val_loss,
            self.n_real_steps)
        self.logger.add_scalar(
            'Koopman System Identification/autoencoder_loss', 
            ae_loss,
            self.n_real_steps)
        self.logger.add_scalar(
            'Koopman System Identification/prediction_loss', 
            pred_loss,
            self.n_real_steps)
        self.logger.add_scalar(
            'Koopman System Identification/combined_loss', 
            comb_loss,
            self.n_real_steps)
        self.logger.add_scalar(
            'Koopman System Identification/n_epochs', 
            i_epoch,
            self.n_real_steps)
        self.logger.flush()
        return best_val_loss.item(), i_epoch

    def set_action_std(self, value=None, low=None, high=None):
        assert value is not None or (low is not None and high is not None), \
            "Either value or low and high must be specified"
        assert value is None or (low is None and high is None), \
            "Specify either value or low and high, not both"

        if value is not None:
            action_std = value
        else:
            action_std = np.random.default_rng().uniform(low=low, high=high)

        self.agent.policy.log_std = T.nn.Parameter(
            T.ones(self.environments['train_real'].action_space.shape[0], device=self.agent.device) * np.log(action_std),
            requires_grad=False)
        return None

    @timer_decorator
    def dream_train_policy(self):
        # set agent environment to dream environment
        self.agent.set_env(self.environments['train_dream'], force_reset=True)
        assert self.environments['train_dream'].default_idx_discrete_model is None, \
            "default_idx_discrete_model must be None for dream training to ensure random choice of model"

        # evaluate policy on dream evaluation environment
        mean_rewards, _ = self._evaluate_policy_dream()
        best_mean_rewards = mean_rewards

        # evaluate policy on real environment
        mean_reward_real, _ = self.evaluate_policy()

        # log to tensorboard
        self.logger.add_scalar(
            'Dream Training/mean_dream_reward', 
            np.mean(mean_rewards),
            self.agent._n_updates+1)
        self.logger.add_scalar(
            'Dream Training/max_dream_reward', 
            np.max(mean_rewards),
            self.agent._n_updates+1)
        self.logger.add_scalar(
            'Dream Training/min_dream_reward', 
            np.min(mean_rewards),
            self.agent._n_updates+1)
        self.logger.add_scalar(
            'Dream Training/mean_real_reward', 
            mean_reward_real,
            self.agent._n_updates+1)
        self.logger.flush()

        # training loop
        steps_between_evals = self.settings['policy_training']['PPO_kwargs']['n_steps']\
            *self.settings['policy_training']['validate_every_n_updates']
        n_updates_best_reward = np.zeros(self.settings['n_dream_models']).astype(int)
        n_updates = 0
        while True:
            # train agent on dream environment
            self.agent.learn(
                total_timesteps=steps_between_evals,
                reset_num_timesteps=False,
                tb_log_name=self.dream_train_log_name,
                callback=None,
                progress_bar=False)
            n_updates += self.settings['policy_training']['validate_every_n_updates']

            # evaluate policy on dream evaluation environment
            mean_rewards, _ = self._evaluate_policy_dream()

            # evaluate policy on real environment
            mean_reward_real, _ = self.evaluate_policy()

            # log to tensorboard
            self.logger.add_scalar(
                'Dream Training/mean_dream_reward', 
                np.mean(mean_rewards),
                self.agent._n_updates)
            self.logger.add_scalar(
                'Dream Training/max_dream_reward', 
                np.max(mean_rewards),
                self.agent._n_updates)
            self.logger.add_scalar(
                'Dream Training/min_dream_reward', 
                np.min(mean_rewards),
                self.agent._n_updates)
            self.logger.add_scalar(
                'Dream Training/mean_real_reward', 
                mean_reward_real,
                self.agent._n_updates)
            self.logger.flush()

            # check if mean rewards are best
            new_best_bools = mean_rewards > best_mean_rewards
            best_mean_rewards[new_best_bools] = mean_rewards[new_best_bools]
            n_updates_best_reward[new_best_bools] = n_updates

            # check if training should stop
            still_improving = n_updates - n_updates_best_reward < self.settings['policy_training']['early_stopping_patience']
            if not np.count_nonzero(still_improving) / self.settings['n_dream_models'] >= self.settings['policy_training']['improvement_ratio_threshold']:
                break

        # log to tensorboard
        self.logger.add_scalar(
            'Dream Training/n_policy_updates', self.agent._n_updates, self.n_real_steps)
        self.logger.flush()
        if self.settings['policy_type'] == 'LearnableBoundsMPC':
            self.agent.policy.mlp_extractor.log_learnable_parameters(
                self.logger, self.n_real_steps)
        return np.mean(mean_rewards)

    def _evaluate_policy_dream(self):
        env_name = 'eval_dream'

        mean_rewards = np.zeros(self.settings['n_dream_models'])
        std_rewards  = np.zeros(self.settings['n_dream_models'])

        for i_model in range(self.settings['n_dream_models']):
            self.environments[env_name].default_idx_discrete_model = i_model

            mean_reward, std_reward = sb3.common.evaluation.evaluate_policy(
                model = self.agent,
                env = self.environments[env_name],
                n_eval_episodes=self.settings['policy_validation']['n_episodes'],
                deterministic=self.settings['policy_validation']['deterministic'],
                warn=False
            )
            mean_rewards[i_model] = mean_reward
            std_rewards[i_model] = std_reward

        return mean_rewards, std_rewards

    def evaluate_policy(self, log_x_axis=None, n_episodes=None):
        if n_episodes is None:
            n_episodes = self.settings['policy_validation']['n_episodes']
        # test policy on real environment
        env_name = 'eval_real'
        mean_reward, std_reward = sb3.common.evaluation.evaluate_policy(
            model = self.agent,
            env = self.environments[env_name],
            n_eval_episodes=self.settings['policy_validation']['n_episodes'],
            deterministic=self.settings['policy_validation']['deterministic'],
            warn=False
        )

        # log to tensorboard
        if log_x_axis == 'n_real_steps':
            self.logger.add_scalar(
                'Real Steps/mean_reward',
                mean_reward,
                self.n_real_steps)
            self.logger.flush()
        return mean_reward, std_reward

    def test_dream_model_ensemble(self):
        # get necessary variables
        state_names = self.environments['train_real'].process_model.state_names

        # set up MSEs
        MSEs = {k: [] for k in state_names}
        MSEs['total'] = []

        # loop through dream models
        for i_model in range(self.settings['n_dream_models']):
            MSEs_model = self._test_single_SI_model(
                model_type='dream', model_idx=i_model)
            for k, v in MSEs_model.items():
                MSEs[k].append(v)

        # log to tensorboard
        for k, v in MSEs.items():
            self.logger.add_scalar(
                f'System Identification/test_mse_model_{k}', 
                np.mean(v),
                self.n_real_steps)
        self.logger.flush()

        return np.mean(MSEs['total'])

    def _test_single_SI_model(self, model_type, model_idx):
        # get model to test
        if model_type == 'dream':
            model = self.model_ensemble.models[model_idx]

        # get necessary variables
        mse_fcn = T.nn.MSELoss()
        device = self.settings['device']
        state_names = self.environments['train_real'].process_model.state_names
        control_names = self.environments['train_real'].process_model.control_names

        # get normalizers
        X_normalizer = self.environments['train_real'].process_model.state_scaler
        U_normalizer = self.environments['train_real'].process_model.action_scaler

        # set up MSEs
        MSEs = {k: [] for k in state_names}
        MSEs['total'] = []

        # loop through test datasets
        for i_tds, tds in self.dream_model_test_data.items():
            # get X and U data from dataset
            X = tds.loc[:, state_names].to_numpy()
            U = tds.loc[:, control_names].to_numpy()

            # normalize data
            X = X_normalizer.scale(X)
            U = U_normalizer.scale(U)

            # ensure numpy arrays for any NumPy ops
            if isinstance(X, T.Tensor):
                X = X.cpu().numpy()
            if isinstance(U, T.Tensor):
                U = U.cpu().numpy()

            # convert to tensors
            X = T.tensor(X, dtype=T.float32).to(device)
            U = T.tensor(U, dtype=T.float32).to(device)

            # simulate entire timeseries using model
            x = X[0, :]
            X_pred = T.zeros_like(X)
            X_pred[0, :] = x
            for t in range(X.shape[0]-1):
                x = model.forward(x, U[t, :])
                X_pred[t+1, :] = x

            # calculate MSEs
            for i, name in enumerate(state_names):
                MSEs[name].append(mse_fcn(X_pred[:, i], X[:, i]).item())
            MSEs['total'].append(mse_fcn(X_pred, X).item())

        # average over datasets
        MSEs = {k: np.mean(v) for k, v in MSEs.items()}
        return MSEs

    def save_self(self, directory=None, name_suffix=""):
        self.save_models(directory, name_suffix)
        self.save_class_attributes(directory, name_suffix)
        self.save_memory(directory, name_suffix)
        return None

    def save_class_attributes(self, directory=None, name_suffix=""):
        if directory is None:
            directory = self.settings['models_dir']
        attributes = self.get_attributes()
        with open(f"{directory}agent_attributes{name_suffix}.pkl", 'wb') as handle:
            pickle.dump(attributes, handle)
        return None

    def save_memory(self, directory=None, name_suffix=""):
        self.memory['train'].save_memory(
            path=f"{directory}train_memory{name_suffix}.pkl")
        self.memory['val'].save_memory(
            path=f"{directory}val_memory{name_suffix}.pkl")
        return None

    def get_attributes(self):
        attribute_names = [
            'dream_model_test_data', 'n_real_steps', 'n_real_constr_viols']
        attributes = {att: self.__dict__[att] for att in attribute_names}
        return attributes

    def save_models(self, directory=None, name_suffix=""):
        if directory is None:
            directory = self.settings['models_dir']
        self.agent.save(f"{directory}agent{name_suffix}.pt")
        T.save(self.model_ensemble.state_dict(), f"{directory}model_ensemble{name_suffix}.pt")
        return None

    def load_self(self, directory=None, name_suffix=""):
        self.load_models(directory, name_suffix)
        self.load_class_attributes(directory, name_suffix)
        self.load_memory(directory, name_suffix)
        return None

    def load_memory(self, directory=None, name_suffix=""):
        self.memory['train'].load_memory(
            path=f"{directory}train_memory{name_suffix}.pkl")
        self.memory['val'].load_memory(
            path=f"{directory}val_memory{name_suffix}.pkl")
        return None

    def load_class_attributes(self, directory=None, name_suffix=""):
        if directory is None:
            directory = self.settings['models_dir']
        with open(f"{directory}agent_attributes{name_suffix}.pkl", 'rb') as handle:
            attributes = pickle.load(handle)
        self.set_attributes(attributes)
        return None

    def set_attributes(self, attributes):
        for k, v in attributes.items():
            self.__dict__[k] = v
        return None

    def load_models(self, directory=None, name_suffix=""):
        self.load_agent(directory, name_suffix)
        self.load_dream_model_ensemble(directory, name_suffix)
        return None

    def load_agent(self, directory=None, name_suffix=""):
        # Get policy
        policy = self.settings['policy_type']

        if directory is None:
            directory = self.settings['models_dir']
        self.agent = self.agent.load(f"{directory}agent{name_suffix}.pt", policy = policy)  # Use .pt for agent loading
        # self.agent = self.agent.load(f"{directory}agent{name_suffix}.zip", policy = policy)  # Old line for .zip files
        if str(type(self.agent.policy.mlp_extractor)) ==\
            "<class 'custom_ac_policies.PPO_ac_CSTR1_LearnableBoundsMPC'>":
            self.agent.policy.mlp_extractor.set_parameters_to_current_koopman_model()
        return None

    def load_dream_model_ensemble(self, directory=None, name_suffix=""):
        if directory is None:
            directory = self.settings['models_dir']
        state_dict = T.load(f"{directory}model_ensemble{name_suffix}.pt", map_location=self.settings['device'])
        self.model_ensemble.load_state_dict(state_dict)
        self.model_ensemble.ensure_all_on_device(self.settings['device'])
        for idx, model in enumerate(self.model_ensemble.models):
            self.model_ensemble.models[idx] = move_model_to_device(model, self.settings['device'])
        # update discrete models of dream environments
        self.environments['train_dream'].process_model.discrete_model = deepcopy(self.model_ensemble)
        self.environments['train_dream'].process_model.discrete_model.ensure_all_on_device(self.settings['device'])
        self.environments['eval_dream'].process_model.discrete_model  = deepcopy(self.model_ensemble)
        self.environments['eval_dream'].process_model.discrete_model.ensure_all_on_device(self.settings['device'])
        return None

    def load_single_dream_model(self, model_idx, directory=None):
        if directory is None:
            directory = self.settings['models_dir']
        state_dict = T.load(f"{directory}single_model_{model_idx}.pt", map_location=self.settings['device'])
        self.model_ensemble.models[model_idx].load_state_dict(state_dict)
        self.model_ensemble.models[model_idx] = move_model_to_device(self.model_ensemble.models[model_idx], self.settings['device'])
        # update discrete models of dream environments
        self.environments['train_dream'].process_model.discrete_model = deepcopy(self.model_ensemble)
        self.environments['train_dream'].process_model.discrete_model.ensure_all_on_device(self.settings['device'])
        self.environments['eval_dream'].process_model.discrete_model  = deepcopy(self.model_ensemble)
        self.environments['eval_dream'].process_model.discrete_model.ensure_all_on_device(self.settings['device'])
        return None

    def plot_policy_behavior(self):
        # get settings
        n_episodes = self.settings['test']['plot_n_episodes']

        # loop through episodes
        for i_ep in range(1, n_episodes+1):
            self._plot_single_episode(i_ep)
        return None

    def _plot_single_episode(self, i_ep):
        # get settings
        env = self.environments['eval_real']
        env_dream = self.environments['eval_dream']
        deterministic = self.settings['policy_validation']['deterministic']

        rewards = []
        observations = []
        actions = []

        dreamed_rewards = []
        dreamed_observations = []
        dreamed_actions = []
        if self.settings['environment']['objective_type'] == 'DemandResponse':
            costs = {'actual': [],
                     'steady_state': []}
            n_constr_viols = 0

        obs, info = env.reset()
        observations.append(obs)
        terminated, truncated = False, False

        env_dream.reset()
        env_dream.x = env.x
        env_dream.electricity_price = env.electricity_price
        env_dream.storage = env.storage
        dreamed_obs = obs
        dreamed_observations.append(dreamed_obs)

        while not (terminated or truncated):
            action = self.agent.predict(obs, deterministic=deterministic)[0]
            actions.append(action)
            dreamed_action = self.agent.predict(dreamed_obs, deterministic=deterministic)[0]
            dreamed_actions.append(dreamed_action)

            obs, reward, terminated, truncated, info = env.step(action)
            observations.append(obs)
            rewards.append(reward)
            dreamed_obs, dreamed_reward, _, _, _ = env_dream.step(dreamed_action)
            dreamed_observations.append(dreamed_obs)
            dreamed_rewards.append(dreamed_reward)
            if self.settings['environment']['objective_type'] == 'DemandResponse':
                costs['actual'].append(info['true_cost'])
                costs['steady_state'].append(info['steady_state_cost'])
                n_constr_viols += int(info['constraint_violation'])

            if terminated or truncated:
                if self.settings['environment']['objective_type'] == 'DemandResponse':
                    relative_cost = sum(costs['actual'])/sum(costs['steady_state'])
                    print(f"Episode: {i_ep}, Return: {sum(rewards)}, Relative cost: {relative_cost}, n_constr_viols: {n_constr_viols}")
                else:
                    print(f"Episode: {i_ep}, Return: {sum(rewards)}")

        # for debugging purposes
        if False:
            trajectory = {
                'observations': observations,
                'actions': actions,
                'rewards': rewards,
                'dreamed_observations': dreamed_observations,
                'dreamed_actions': dreamed_actions,
                'dreamed_rewards': dreamed_rewards,
                'relative_cost': relative_cost,
                'n_constr_viols': n_constr_viols,
            }
            with open(f"{self.settings['plots_dir']}/pol_behavior_{i_ep}.pkl", 'wb') as handle:
                pickle.dump(trajectory, handle, protocol=pickle.HIGHEST_PROTOCOL)

        if self.settings['environment']['objective_type'] == 'DemandResponse'\
        and self.settings['environment']['process'] == 'CSTR1':
            common.plot_cstr_demand_response_trajectory(
                observations, actions, rewards,
                dreamed_observations, dreamed_actions, dreamed_rewards,
                relative_cost, n_constr_viols,
                save_figure=True, show_figure=True,
                figure_path=f"{self.settings['plots_dir']}/pol_behavior_{i_ep}.pdf")
        elif self.settings['environment']['objective_type'] == 'StabilizeState'\
        and self.settings['environment']['process'] == 'CSTR1':
            common.plot_cstr_stabilize_trajectory(observations, actions, rewards,
                                save_figure=True, show_figure=True,
                                figure_path=f"{self.settings['plots_dir']}/pol_behavior_{i_ep}.pdf")
        else:
            raise NotImplementedError
        return None

    def _plot_single_episode_old(self, i_ep):
        # get settings
        env = self.environments['eval_real']
        env_dream = self.environments['eval_dream']
        deterministic = self.settings['policy_validation']['deterministic']

        rewards = []
        observations = []
        dreamed_obs = []
        actions = []
        if self.settings['environment']['objective_type'] == 'DemandResponse':
            costs = {'actual': [],
                     'steady_state': []}
            n_constr_viols = 0

        obs, info = env.reset()
        env_dream.reset()
        env_dream.x = env.x
        observations.append(obs)
        terminated, truncated = False, False

        while not (terminated or truncated):
            action = self.agent.predict(obs, deterministic=deterministic)[0]
            actions.append(action)

            env_dream.x = env.x
            obs, reward, terminated, truncated, info = env.step(action)
            observations.append(obs)
            rewards.append(reward)
            _, _, _, _, info_dream = env_dream.step(action)
            dreamed_obs.append(info_dream['X'][[0,-1],:])
            if self.settings['environment']['objective_type'] == 'DemandResponse':
                costs['actual'].append(info['true_cost'])
                costs['steady_state'].append(info['steady_state_cost'])
                n_constr_viols += int(info['constraint_violation'])

            if terminated or truncated:
                if self.settings['environment']['objective_type'] == 'DemandResponse':
                    relative_cost = sum(costs['actual'])/sum(costs['steady_state'])
                    print(f"Episode: {i_ep}, Return: {sum(rewards)}, Relative cost: {relative_cost}, n_constr_viols: {n_constr_viols}")
                else:
                    print(f"Episode: {i_ep}, Return: {sum(rewards)}")

        if self.settings['environment']['objective_type'] == 'DemandResponse'\
        and self.settings['environment']['process'] == 'CSTR1':
            common.plot_cstr_demand_response_trajectory(
                observations, actions, rewards, dreamed_obs,
                relative_cost, n_constr_viols,
                save_figure=True, show_figure=True,
                figure_path=f"{self.settings['plots_dir']}/pol_behavior_{i_ep}.pdf")
        elif self.settings['environment']['objective_type'] == 'StabilizeState'\
        and self.settings['environment']['process'] == 'CSTR1':
            common.plot_cstr_stabilize_trajectory(observations, actions, rewards,
                                save_figure=True, show_figure=True,
                                figure_path=f"{self.settings['plots_dir']}/pol_behavior_{i_ep}.pdf")
        else:
            raise NotImplementedError
        return None
    
    def _fit_single_PINN_model(self, model_type, model_idx):

        delta_t = self.settings['CSTR1_DemandResponse']['process_kwargs']['delta_t']['timestep']/3600
        n_step  = self.settings['dream_model_training']['n_step_prediction_loss']
        # function to reshape tensors 
        def reshape_traj_batch(tensor):            
            return tensor.reshape(-1,tensor.shape[-1])
        
        # function to adjust data for PINN
        def prepare_pinn_data(states, actions, states_):
            times = T.linspace(delta_t, delta_t*(n_step), n_step, dtype=T.float32)     
            out_states = T.empty_like(states_)
            controls = T.empty_like(actions)
            out_states = states_                       
            controls = actions
                                    
            #reshape data tensors for PINN training
            times = reshape_traj_batch(times.unsqueeze(0).repeat(states.shape[0],1).unsqueeze(2))
            init_states = reshape_traj_batch(states[:,0,:].unsqueeze(1).repeat(1,out_states.shape[1],1))
            controls = reshape_traj_batch(controls)
            out_states = reshape_traj_batch(out_states)
            
            #unscale training data (scaled again in forward pass of PINN)
            init_states = model.state_scaler.unscale(init_states)
            out_states = model.state_scaler.unscale(out_states)
            controls = model.action_scaler.unscale(controls)

            return times, init_states, controls, out_states
        
        # IDW utilities
        # tuning parameter for weight updates
        alpha = 0.5
        wi = 1
        wd = 1
        
        # from Maddu et al. 2022 GitHub project
        def loss_grad_std_wn(loss, net):
            grad_ = T.zeros((0), dtype=T.float32, device=next(net.parameters()).device)  # Ensure grad_ is on the model's device
            for elem in T.autograd.grad(loss, net.parameters(), retain_graph=True):
                # grad_ = T.cat((grad_, elem.view(-1)))  # Replaced for GPU compatibility
                grad_ = T.cat((grad_, elem.view(-1).to(grad_.device)))
            return T.std(grad_)
                            
        # get model to fit
        if model_type == 'dream':
            model = self.model_ensemble.models[model_idx]
        
        # possibly reset model parameters
        if random.random() < self.settings['dream_model_training']['reset_probability']:
            model.apply(common.weight_reset_torch_module)
               
        # Stage 1 - Training with SGD (ADAM)
        i_epoch = 0
        while i_epoch < model.iter_adam:
            i_epoch += 1

            batch_counter = 0
            batch_cycle = math.floor(self.memory['train'].mem_cntr/self.settings['dream_model_training']['batch_size'])
            memory_depleted = False
            while not memory_depleted:
                try:
                    # get experience batch
                    states, actions, states_, dones, memory_depleted = \
                        self.memory['train'].get_next_random_batch(
                            self.settings['dream_model_training']['batch_size'])
                except:
                    memory_depleted = True
                    self.memory['train']._reset_remember_order()
                    break

                # adjust experience data for PINN
                times, init_states, controls, out_states = prepare_pinn_data(states, actions, states_)
                # Move all tensors to model.device
                device = model.device
                times = times.to(device)
                init_states = init_states.to(device)
                controls = controls.to(device)
                out_states = out_states.to(device)

                # batch PINN data
                t_pinn, x_pinn, u_pinn = model.batch_pinn_data(batch_cycle, batch_counter, full_batch = False)
                t_pinn = t_pinn.to(device)
                x_pinn = x_pinn.to(device)
                u_pinn = u_pinn.to(device)

                # initial condition loss (not batched since way smaller number of data points than other sets)
                loss_i = model.loss_init()

                # physics loss
                loss_p = model.loss_physics(t_pinn, x_pinn, u_pinn)

                # data loss
                loss_d = model.loss_data(times, init_states, controls, out_states)

                # Inverse Dirichlet weighting                
                # check & apply inverse dirichlet weighting
                with T.no_grad():
                    if i_epoch % model.idw_cycle == 0 and batch_counter == 0:                     
                        std_p = loss_grad_std_wn(loss_p, model)
                        std_i = loss_grad_std_wn(loss_i, model)
                        std_d = loss_grad_std_wn(loss_d, model)

                        wi = (1-alpha)*wi + alpha*(std_p/std_i)
                        wd = (1-alpha)*wd + alpha*(std_p/std_d)

                loss = loss_p + wi*loss_i + wd*loss_d 
                model.adam.zero_grad()                  
                loss.backward()
                model.adam.step()

                batch_counter += 1
        # Stage 2 - Training with LBFGS

        #closure for LBFGS
        def closure():
            model.lbfgs.zero_grad()
            # Move all tensors to model.device
            device = model.device
            t_pinn_ = t_pinn.to(device)
            x_pinn_ = x_pinn.to(device)
            u_pinn_ = u_pinn.to(device)
            times_ = times.to(device)
            init_states_ = init_states.to(device)
            controls_ = controls.to(device)
            out_states_ = out_states.to(device)
            loss = model.loss_physics(t_pinn_, x_pinn_, u_pinn_) + wd*model.loss_data(times_, init_states_, controls_, out_states_) + wi*model.loss_init()
            loss.backward()   
            return loss    

        # take full batch PINN data
        t_pinn, x_pinn, u_pinn = model.batch_pinn_data(batch_cycle, batch_counter, full_batch = True)
        t_pinn = t_pinn.to(model.device)
        x_pinn = x_pinn.to(model.device)
        u_pinn = u_pinn.to(model.device)
        
        # get entire train memory for LBFGS (no batching)
        states, actions, states_, dones = self.memory['train'].get_entire_memory()

        times, init_states, controls, out_states = prepare_pinn_data(states, actions, states_)

        # get validation training data
        val_states, val_actions, val_states_, val_dones = self.memory['val'].get_entire_memory()
        val_times, val_init_states, val_controls, val_out_states = prepare_pinn_data(val_states, val_actions, val_states_)
        # Move all validation tensors to model.device
        val_times = val_times.to(model.device)
        val_init_states = val_init_states.to(model.device)
        val_controls = val_controls.to(model.device)
        val_out_states = val_out_states.to(model.device)

        best_epoch = 0
        best_val_loss = np.inf
        i_epoch = 0
        while i_epoch < model.iter_lbfgs:
            i_epoch += 1    
            model.lbfgs.step(closure)  
            
            if i_epoch % self.settings['dream_model_training']['validate_every_n_epochs'] != 0:
               continue

            val_loss_d = model.loss_data(val_times, val_init_states, val_controls, val_out_states)
            val_loss_p = model.loss_physics(val_times, val_init_states, val_controls)
            val_loss_i = model.loss_init()
            val_loss =   val_loss_p + wd*val_loss_d + wi*val_loss_i

            # check if validation loss is best
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_data = val_loss_d
                best_val_physics = val_loss_p
                best_epoch = i_epoch
                best_model_state_dict = deepcopy(model.state_dict())

            # check if training should stop early
            if i_epoch - best_epoch >= self.settings['dream_model_training']['early_stopping_patience']:
                break

        # load best model
        model.load_state_dict(best_model_state_dict)

        # save model in self
        if model_type == 'dream':
            self.model_ensemble.models[model_idx] = move_model_to_device(model, self.settings['device'])

        return best_val_data.item(), best_val_physics.item(), i_epoch

