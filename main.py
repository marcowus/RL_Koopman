# 3rd party imports
import os
import torch as T
import numpy as np
import time
import pickle
import warnings

# Local imports
import mbpo
from common import get_default_device

seed = None
if seed is not None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    T.manual_seed(seed)

def get_settings():
    # Settings
    settings = {
        ### General
        'train_or_test':        'test',                # options: 'train', 'test'
        'environment': {
            'process':          'CSTR1',
            'objective_type':   'DemandResponse',
        },
        'policy_type': 	        'LearnableBoundsMPC',   # options: 'MlpPolicy', 'CustomMLP', 'LearnableKoopmanMPC', 'LearnableBoundsMPC'
        'dream_model_type':     'PINN',                 # options: 'MLP', 'PINN'
        'MBRL_algorithm':       'MBPO',
        'MFRL_algorithm':       'PPO',
        'device': get_default_device(),  # This is the global device for all tensors/models
        'profile':              False,
        'debug':                True,

        ### Test settings
        'test': {
            'agent_name_suffix':            '_BestTestMSE',
            'n_episodes':                   5,
            'plot':                         True,
            'plot_n_episodes':              5,
        },

        ### Environment
        # CSTR1_DemandResponse environment
        'CSTR1_DemandResponse': {
            'process_kwargs': {
                'delta_t': {
                    'timestep': 15*60.0,
                    'control':  60*60.0
                },
            },
            'env_kwargs': {
                'episode_length': 24*7,
                'price_prediction_horizon': 9,
                'storage_size': 6.0,
                'terminate_episode_at_relative_constr_viol': 1.0,
                'constant_constr_viol_penalty': 0.1,
                'step_reward': 1.0,
                'cost_reward_scaling_factor': 5.0e-6,
            }
        },

        ### Policy
        'action_std': 	                    0.05,
        'init_Koopman_model':               'random',

        ### MBPO
        'max_real_timesteps': 		        2_500,
        # Data collection
        'data_collection': {
            'warm_start_memory':            False,
            'memory_size':                  100_000,
            'action_std_range':             [0.0, 0.1],
            'train_val_ratio':              0.75,
        },

        # Dream model
        'n_dream_models':                   10,
        'MLP_dream_model_kwargs': {
            'hidden_layer_sizes':           [32, 32],
            'activation':                   T.nn.Tanh,
            'output_activation':            T.nn.Identity,
            'predict_delta':                True,
            'learning_rate':                0.0001,
            'use_weight_normalization':     False,
        },
        'PINN_dream_model_kwargs': {
            'hidden_layer_sizes':           [32, 32],    
            'data_idx':                     [0, 1],            
            'activation':                   T.nn.Tanh,      
            'output_activation':            T.nn.Identity,
            'learning_rate':                0.001,
            'use_weight_normalization':     False,
            'idw_cycle':                    10,
            'iter_adam':                    1000,
            'iter_lbfgs':                   300,
            'Np':                           2_000,
            'Ni':                           100,
        },

        # Dream model training
        'use_pretrained_dream_models':      False,
        'use_precollected_data':            False,
        'precollected_data_name':           '5d_29July_train',
        'dream_model_training':{
            'reset_probability':            0.34,
            'max_epochs':                   5_000,
            'batch_size':                   64,
            'validate_every_n_epochs':      5,
            'early_stopping_patience':      100,
        },

        # Dream model testing
        'dream_model_testing':{
            'test_data_name': {
                'CSTR1':                    '5d_29July_test',
            },
        },

        # Koopman model SI training
        'koopman_SI_training':{
            'max_epochs':                   5_000,
            'batch_size':                   64,
            'learning_rate':                0.0001,
            'validate_every_n_epochs':      5,
            'early_stopping_patience':      25,
            'n_step_prediction_loss':       10,
        },

        # Policy training
        'policy_training': {
            'episode_length':               8,
            'improvement_ratio_threshold':  0.7,
            'PPO_kwargs': {
                'n_steps': 			        2048,
                'batch_size': 			    256,
                'gamma': 				    0.95,
                'gae_lambda': 			    0.95,
                'clip_range': 			    0.2,
                'clip_range_vf': 		    None,
                'ent_coef': 			    0.001,
                'vf_coef': 				    0.5,
                'max_grad_norm': 		    0.5,
                'policy_kwargs': {          # only relevant for generic 'MlpPolicy'
                    'activation_fn': 	    T.nn.Tanh,
                    'net_arch':             [64, 64],
                    'squash_output':        False,
                },
            },
        },

        # Policy evaluation
        'policy_validation': {
            'n_episodes':                   5,
            'deterministic':                True,
        },

        # Printing
        'print_every_n_episodes':           1,

        # settings to skip steps of the main algorithm
        'skip_step': {
            'collect_experience':           False,
            'fit_dream_models':             False,
            'test_dream_models':            False,
            'SI_fit_koopman_model':         False,
            'train_policy':                 False,
            'evaluate_policy':              False,
        },
    }

    # Conditional settings
    if settings['policy_type'] in ['CustomMLP', 'MlpPolicy']:
        settings['policy_training']['PPO_kwargs']['learning_rate'] = 1e-4
        settings['policy_training']['PPO_kwargs']['n_epochs'] = 10
        settings['policy_training']['validate_every_n_updates'] = 5
        settings['policy_training']['early_stopping_patience'] = 25
    elif settings['policy_type'] in ['LearnableKoopmanMPC', 'LearnableBoundsMPC']:
        settings['policy_training']['PPO_kwargs']['n_epochs'] = 1
        settings['policy_training']['validate_every_n_updates'] = 1
        settings['policy_training']['early_stopping_patience'] = 10
        match settings['policy_type']:
            case 'LearnableKoopmanMPC':
                settings['policy_training']['PPO_kwargs']['learning_rate'] = 1e-5
            case 'LearnableBoundsMPC':
                settings['policy_training']['PPO_kwargs']['learning_rate'] = 1e-3

    match settings['environment']['process']:
        case 'CSTR1':
            settings['data_collection']['initial_random_actions'] = 20
            settings['data_collection']['collect_memory_timesteps'] = \
                ((200, 20),         # (up_to_n_steps, collect_n_steps)
                 (500, 50),
                 (5_000, 250),
                 (20_000, 1_000),
                 (100_000, 5_000))
            settings['dream_model_training']['n_step_prediction_loss'] = 4
        case _:
            raise ValueError(f"Settings for process '{settings['environment']['process']}' not implemented.")

    settings['delta_t'] = settings[settings['environment']['process']+'_'+settings['environment']['objective_type']]['process_kwargs']['delta_t']

    # Change settings if debug-mode is activated
    if settings['debug']:
        settings['data_collection']['initial_random_actions'] = 50
        settings['data_collection']['collect_memory_timesteps'] = ((500, 50),         # (up_to_n_steps, collect_n_steps)
                                                                   (10_000, 250),
                                                                   (100_000, 1_000))
        settings['n_dream_models'] = 3
        settings['dream_model_training']['max_epochs'] = 50
        settings['koopman_SI_training']['max_epochs'] = 50
        settings['policy_training']['validate_every_n_updates'] = 1
        settings['policy_training']['early_stopping_patience'] = 2
        settings['policy_training']['PPO_kwargs']['n_steps'] = 128
        settings['policy_training']['PPO_kwargs']['n_epochs'] = 2

    # Directories
    settings['models_dir'] = f"./models/{settings['environment']['process']}_{settings['environment']['objective_type']}_{settings['MBRL_algorithm']}/{time.strftime('%Y%m%d-%H%M%S')}/"
    settings['logdir'] = f"./logs/{settings['environment']['process']}_{settings['environment']['objective_type']}_{settings['MBRL_algorithm']}/"
    settings['precollected_train_data_path'] = f"./data/{settings['environment']['process']}/{settings['precollected_data_name']}.pickle"
    settings['test_data_path'] = f"./data/{settings['environment']['process']}/{settings['dream_model_testing']['test_data_name'][settings['environment']['process']]}.pickle"
    settings['saved_dream_models_dir'] = f"./saved_models/{settings['environment']['process']}/dream_models/single/{settings['dream_model_type']}/"
    settings['test']['agent_dir'] = f"./saved_models/{settings['environment']['process']}/{settings['environment']['objective_type']}/"
    settings['plots_dir'] = f"./plots/"
    settings['memory_dir'] = f"./data/memory/{settings['environment']['process']}/{settings['environment']['objective_type']}/"

    os.makedirs(settings['models_dir'], exist_ok=True)
    os.makedirs(settings['logdir'], exist_ok=True)
    os.makedirs(settings['logdir']+'SB3/', exist_ok=True)
    os.makedirs(settings['plots_dir'], exist_ok=True)
    os.makedirs(settings['memory_dir'], exist_ok=True)
    os.makedirs('./warm_start_checkpoint/', exist_ok=True)

    # settings assertions

    return settings

def print_training_information(info, i_outer_loop, settings):
    if i_outer_loop%(settings['print_every_n_episodes']*10) == 0:
        print("")
        print(f"{'Process':<8} {'Objective type':<15} | \
{'i_outer_loop':<13} {'n_real_steps':<13} | \
{'mean_reward_real':<17} {'mean_mean_reward_dream':<23} | \
{'SI_mean_val_loss':<17} {'SI_mean_test_mse':<17} {'SI_mean_epochs':<15} | \
{'time_collect_exp':<17} {'time_fit_dream_model':<21} {'time_train_policy':<18}")

    if i_outer_loop%settings['print_every_n_episodes'] == 0:
        print(f"{settings['environment']['process']:<8} {settings['environment']['objective_type']:<15} | \
{i_outer_loop+1:<13} {info['n_real_steps']:<13} | \
{round(info['mean_reward_real'],2):<17} {round(info['mean_mean_reward_dream'],2):<23} | \
{round(info['mean_val_loss_dream_model'],6):<17} {round(info['mean_test_mse'],6):<17} {round(info['mean_epochs_dream_model'],2):<15} | \
{round(info['time_collect_exp'],1):<17} {round(info['time_fit_dream_model'],1):<21} {round(info['time_train_policy'],1):<18}")
    return None

def write_progress_checkpoint(dir, variables, mbrl_agent):
    with open(dir+'checkpoint.pickle', 'wb') as handle:
        pickle.dump(variables, handle, protocol=pickle.HIGHEST_PROTOCOL)
    mbrl_agent.save_self(directory=dir, name_suffix='_checkpoint')
    return None

def load_progress_checkpoint(dir, mbrl_agent):
    warnings.warn(
        "Loading progress checkpoint. The agent will not be learned from scratch.",
        UserWarning)
    with open(dir+'checkpoint.pickle', 'rb') as handle:
        variables = pickle.load(handle)
    mbrl_agent.load_self(directory=dir, name_suffix='_checkpoint')
    return variables

def delete_warm_start_checkpoint():
    folder = './warm_start_checkpoint/'
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        os.unlink(file_path)
    return None

def train(mbrl_agent, settings):
    if settings['profile']:
        import cProfile
        profiler = cProfile.Profile()
        profiler.enable()

    # Make empty arrays to save training information
    training_info = {
        'n_real_steps':             0,
        # System Identification
        'mean_val_loss_dream_model':0.0,
        'mean_epochs_dream_model':  0,
        'mean_test_mse':            0.0,

        # Policy training
        'mean_mean_reward_dream':   0.0,
        'mean_reward_real':         0.0,

        # Timing
        'time_collect_exp':         0.0,
        'time_fit_dream_model':     0.0,
        'time_train_policy':        0.0,
    }

    # Evaluate the untrained policy
    if not settings['skip_step']['evaluate_policy']\
    and not os.path.isfile('./warm_start_checkpoint/checkpoint.pickle'):
        _, _ = mbrl_agent.evaluate_policy(log_x_axis='n_real_steps')

    # Train the agent
    i_outer_loop = 0
    best_mean_reward_real  = -np.inf
    best_mean_reward_dream = -np.inf
    best_mean_test_mse     = np.inf

    # check if training should be warm-started from checkpoint
    if os.path.isfile('./warm_start_checkpoint/checkpoint.pickle'):
        [settings, training_info, i_outer_loop,
         best_mean_reward_real, best_mean_reward_dream, best_mean_test_mse] = \
            load_progress_checkpoint('./warm_start_checkpoint/', mbrl_agent)

    while mbrl_agent.n_real_steps < settings['max_real_timesteps']:
        # Collect data from training environment
        if not settings['skip_step']['collect_experience']\
        and not (i_outer_loop==0 and settings['use_precollected_data']):
            random_actions = True \
                if mbrl_agent.n_real_steps < settings['data_collection']['initial_random_actions'] \
                else False
            for step_limit, collect_steps in settings['data_collection']['collect_memory_timesteps']:
                if mbrl_agent.n_real_steps < step_limit:
                    break
            if random_actions:
                collect_steps = settings['data_collection']['initial_random_actions']
            _, training_info['time_collect_exp'] = \
                mbrl_agent.collect_experience(
                    collect_steps,
                    random_actions=random_actions)
            training_info['n_real_steps'] = mbrl_agent.n_real_steps

        # Train the dream models
        if not settings['skip_step']['fit_dream_models']\
        and not (i_outer_loop==0 and settings['use_precollected_data']\
                 and settings['use_pretrained_dream_models']):
            (training_info['mean_val_loss_dream_model'],
            training_info['mean_epochs_dream_model']),\
            training_info['time_fit_dream_model'] = \
                mbrl_agent.fit_dream_model_ensemble()

        # Test the dream models
        if not settings['skip_step']['test_dream_models']:
            training_info['mean_test_mse'] = mbrl_agent.test_dream_model_ensemble()

        # SI-train the Koopman model
        if settings['policy_type'] in ['LearnableKoopmanMPC', 'LearnableBoundsMPC']\
        and settings['init_Koopman_model'] == 'random'\
        and not settings['skip_step']['SI_fit_koopman_model']:
            mbrl_agent.fit_koopman_model()

        # Train the policy on the dream environment
        if not settings['skip_step']['train_policy']:
            training_info['mean_mean_reward_dream'], training_info['time_train_policy'] = \
                mbrl_agent.dream_train_policy()

        # Evaluate the policy
        if not settings['skip_step']['evaluate_policy']:
            training_info['mean_reward_real'], _ = mbrl_agent.evaluate_policy(log_x_axis='n_real_steps')

        # Save model
        mbrl_agent.save_models(name_suffix=f'_{mbrl_agent.n_real_steps}RealSteps')
        if training_info['mean_reward_real'] > best_mean_reward_real:
            best_mean_reward_real = training_info['mean_reward_real']
            mbrl_agent.save_models(name_suffix='_BestRealReward')
        if training_info['mean_mean_reward_dream'] > best_mean_reward_dream:
            best_mean_reward_dream = training_info['mean_mean_reward_dream']
            mbrl_agent.save_models(name_suffix='_BestDreamReward')
        if training_info['mean_test_mse'] < best_mean_test_mse:
            best_mean_test_mse = training_info['mean_test_mse']
            mbrl_agent.save_models(name_suffix='_BestTestMSE')

        # Print training information
        print_training_information(training_info, i_outer_loop, settings)
        i_outer_loop += 1

        # Save progress
        write_progress_checkpoint(
            dir='./warm_start_checkpoint/',
            variables=[settings, training_info, i_outer_loop,
                       best_mean_reward_real, best_mean_reward_dream, best_mean_test_mse],
            mbrl_agent=mbrl_agent)

    if settings['profile']:
        profiler.disable()
        profiler.dump_stats(f"{settings['logdir']}profile.prof")

    delete_warm_start_checkpoint()
    return None

def test(mbrl_agent, settings):
    # Test the agent
    mbrl_agent.evaluate_policy(log_x_axis='n_real_steps')

    # Load agent
    mbrl_agent.load_agent(
        directory=settings['test']['agent_dir'],
        name_suffix=settings['test']['agent_name_suffix'])

    # Test the agent
    mean_reward, std_reward = mbrl_agent.evaluate_policy(
        n_episodes=settings['test']['n_episodes'])
    print(f"Mean reward over {settings['test']['n_episodes']} episodes: {mean_reward} +- {std_reward}")

    # Plot behavior
    if settings['test']['plot']:
        mbrl_agent.plot_policy_behavior()

    return None

if __name__ == '__main__':

    settings = get_settings()

    # Get MBRL agent
    mbrl_agent = mbpo.get_mbrl_agent(settings)

    # Execute main function
    if settings['train_or_test'] == 'train':
        train(mbrl_agent, settings)
    elif settings['train_or_test'] == 'test':
        test(mbrl_agent, settings)

    print('Done!')