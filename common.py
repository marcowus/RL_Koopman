from time import time
import numpy as np
import matplotlib.pyplot as plt
import torch as T


# useful across projects
def get_default_device():
    return T.device('cuda' if T.cuda.is_available() else 'cpu')

class ElementwiseScaler():
    # scales and unscales a vector elementwise
    # min and max should be np.arrays of the same shape as the vector x
    def __init__(self,
                 min_unscaled, max_unscaled,
                 min_scaled, max_scaled,
                 device=None):
        self.device = device if device is not None else get_default_device()
        self.min_unscaled = min_unscaled
        self.max_unscaled = max_unscaled
        self.range_unscaled = max_unscaled - min_unscaled
        self.min_scaled = min_scaled
        self.max_scaled = max_scaled
        self.range_scaled = max_scaled - min_scaled

    def scale(self, x):
        if isinstance(x, np.ndarray):
            x = T.tensor(x, dtype=T.float32, device=self.device)
        if isinstance(x, T.Tensor):
            min_unscaled = T.tensor(self.min_unscaled, dtype=x.dtype, device=self.device) if not isinstance(self.min_unscaled, T.Tensor) else self.min_unscaled.to(self.device)
            range_unscaled = T.tensor(self.range_unscaled, dtype=x.dtype, device=self.device) if not isinstance(self.range_unscaled, T.Tensor) else self.range_unscaled.to(self.device)
            min_scaled = T.tensor(self.min_scaled, dtype=x.dtype, device=self.device) if not isinstance(self.min_scaled, T.Tensor) else self.min_scaled.to(self.device)
            range_scaled = T.tensor(self.range_scaled, dtype=x.dtype, device=self.device) if not isinstance(self.range_scaled, T.Tensor) else self.range_scaled.to(self.device)
            x = (x - min_unscaled) / range_unscaled
            x = x * range_scaled + min_scaled
            return x
        else:
            x = (x - self.min_unscaled) / self.range_unscaled
            x = x * self.range_scaled + self.min_scaled
            return x

    def unscale(self, x):
        if isinstance(x, np.ndarray):
            x = T.tensor(x, dtype=T.float32, device=self.device)
        if isinstance(x, T.Tensor):
            min_unscaled = T.tensor(self.min_unscaled, dtype=x.dtype, device=self.device) if not isinstance(self.min_unscaled, T.Tensor) else self.min_unscaled.to(self.device)
            range_unscaled = T.tensor(self.range_unscaled, dtype=x.dtype, device=self.device) if not isinstance(self.range_unscaled, T.Tensor) else self.range_unscaled.to(self.device)
            min_scaled = T.tensor(self.min_scaled, dtype=x.dtype, device=self.device) if not isinstance(self.min_scaled, T.Tensor) else self.min_scaled.to(self.device)
            range_scaled = T.tensor(self.range_scaled, dtype=x.dtype, device=self.device) if not isinstance(self.range_scaled, T.Tensor) else self.range_scaled.to(self.device)
            x = (x - min_scaled) / range_scaled
            x = x * range_unscaled + min_unscaled
            return x
        else:
            x = (x - self.min_scaled) / self.range_scaled
            x = x * self.range_unscaled + self.min_unscaled
            return x

    def _self_to_numpy(self):
        self.min_unscaled = np.array(self.min_unscaled)
        self.max_unscaled = np.array(self.max_unscaled)
        self.range_unscaled = np.array(self.range_unscaled)
        self.min_scaled = np.array(self.min_scaled)
        self.max_scaled = np.array(self.max_scaled)
        self.range_scaled = np.array(self.range_scaled)
        return None

    def _self_to_torch(self, dtype=T.float32):
        self.min_unscaled = T.tensor(self.min_unscaled, dtype=dtype)
        self.max_unscaled = T.tensor(self.max_unscaled, dtype=dtype)
        self.range_unscaled = T.tensor(self.range_unscaled, dtype=dtype)
        self.min_scaled = T.tensor(self.min_scaled, dtype=dtype)
        self.max_scaled = T.tensor(self.max_scaled, dtype=dtype)
        self.range_scaled = T.tensor(self.range_scaled, dtype=dtype)
        return None

def timer_decorator(some_function):
    # https://stackoverflow.com/questions/70642928/python-measure-function-execution-time-with-decorator
    def wrapper(*args, **kwargs):
        t1 = time()
        result = some_function(*args, **kwargs)
        end = time()-t1
        return result, end
    return wrapper

def log_dict_with_tensorboard(writer, dict, step):
    for key in dict:
        writer.add_scalar(key, dict[key], step)
    writer.flush()
    return None

def weight_reset_torch_module(model):
    # https://discuss.pytorch.org/t/how-to-re-set-alll-parameters-in-a-network/20819/14
    # use via: model.apply(weight_reset)
    if isinstance(model, T.nn.Conv2d) or isinstance(model, T.nn.Linear):
        model.reset_parameters()

def reset_torch_parameter(param):
    assert isinstance(param, T.nn.parameter.Parameter), \
        "Input must be a torch.nn.parameter.Parameter"
    assert param.dim() == 2, "Currently only implemented for 2D parameters"

    with T.no_grad():
        new_param = T.nn.Linear(param.shape[1], param.shape[0])
        new_param = new_param.weight
    return new_param

def plot_cstr_demand_response_trajectory(
        observations, actions, rewards,
        dreamed_observations, dreamed_actions, dreamed_rewards,
        relative_cost, n_constr_viols,
        show_dream=False, save_figure=True, show_figure = False,
        figure_path=f"./figures/trajectory.pdf"
        ):
    observations = np.stack(observations, axis=0)
    actions = np.stack(actions, axis=0)
    rewards = np.stack(rewards, axis=0)
    dreamed_observations = np.stack(dreamed_observations, axis=0)
    dreamed_actions = np.stack(dreamed_actions, axis=0)
    dreamed_rewards = np.stack(dreamed_rewards, axis=0)
    
    c = observations[:-1,0]
    T = observations[:-1,1]
    storage = observations[:-1,2]
    price = observations[:-1,3]
    rho = actions[:,0]
    Fc = actions[:,1]
    t = list(range(len(c)))

    dreamed_c = dreamed_observations[:-1,0]
    dreamed_T = dreamed_observations[:-1,1]
    dreamed_storage = dreamed_observations[:-1,2]
    dreamed_rho = dreamed_actions[:,0]
    dreamed_Fc = dreamed_actions[:,1]

    fig, ax = plt.subplots(6, 1, sharex=True, sharey=False, figsize=(10,10))
    linewidth = 0.8

    fig.suptitle(f'Relative cost: {round(float(relative_cost)*100,1)}% - n_constr_viols: {n_constr_viols}', fontsize=16)
    
    ax[0].plot(t,c, linewidth=linewidth)
    ax[0].plot(t, [-1.0]*len(t), linewidth=linewidth, color='black', linestyle='--')
    ax[0].plot(t, [1.0]*len(t), linewidth=linewidth, color='black', linestyle='--')
    ax[0].set_title(label='Product concentration')
    
    ax[1].plot(t,T, linewidth=linewidth)
    ax[1].plot(t, [-1.0]*len(t), linewidth=linewidth, color='black', linestyle='--')
    ax[1].plot(t, [1.0]*len(t), linewidth=linewidth, color='black', linestyle='--')
    ax[1].set_title(label='Temperature')
    
    ax[2].plot(t,rho, linewidth=linewidth)
    ax[2].plot(t, [-1.0]*len(t), linewidth=linewidth, color='black', linestyle='--')
    ax[2].plot(t, [1.0]*len(t), linewidth=linewidth, color='black', linestyle='--')
    ax[2].set_title(label='Production Volume')
    
    ax[3].plot(t,Fc, linewidth=linewidth)
    ax[3].plot(t, [-1.0]*len(t), linewidth=linewidth, color='black', linestyle='--')
    ax[3].plot(t, [1.0]*len(t), linewidth=linewidth, color='black', linestyle='--')
    ax31 = ax[3].twinx()
    ax31.plot(t,price, linewidth=linewidth, color='red', linestyle='--')
    ax31.tick_params(axis='y', colors='red')
    ax[3].set_title(label='Coolant Flowrate | Elec. price')

    ax[4].plot(t,storage, linewidth=linewidth)
    ax[4].set_title(label='Stored product (in hours of nominal production)')

    ax[5].plot(t,rewards, linewidth=linewidth)
    ax[5].set_title(label='Reward')

    if show_dream:
        ax[0].plot(t,dreamed_c, linewidth=linewidth, color='green', linestyle='--')
        ax[1].plot(t,dreamed_T, linewidth=linewidth, color='green', linestyle='--')
        ax[2].plot(t,dreamed_rho, linewidth=linewidth, color='green', linestyle='--')
        ax[3].plot(t,dreamed_Fc, linewidth=linewidth, color='green', linestyle='--')
        ax[4].plot(t,dreamed_storage, linewidth=linewidth, color='green', linestyle='--')
        ax[5].plot(t,dreamed_rewards, linewidth=linewidth, color='green', linestyle='--')

    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.grid(False)
    plt.xlabel('Time steps', fontsize=13)
    
    plt.tight_layout()
    if save_figure:
        plt.savefig(figure_path)
    if show_figure:
        plt.show()
    plt.close(fig)

def plot_cstr_stabilize_trajectory(
        observations, actions, rewards,
        save_figure=True, show_figure = False,
        figure_path=f"./figures/trajectory.pdf"
        ):
    observations = np.stack(observations, axis=0)
    actions = np.stack(actions, axis=0)
    rewards = np.stack(rewards, axis=0)
    
    c = observations[:-1,0]
    T = observations[:-1,1]
    rho = observations[:-1,2]
    Fc = actions[:,0]
    t = list(range(len(c)))
    
    fig, ax = plt.subplots(5, 1, sharex=True, sharey=False, figsize=(10,10))
    linewidth = 0.8
    
    ax[0].plot(t,c, linewidth=linewidth)
    ax[0].plot(t, [-1.0]*len(t), linewidth=linewidth, color='black', linestyle='--')
    ax[0].plot(t, [1.0]*len(t), linewidth=linewidth, color='black', linestyle='--')
    ax[0].set_title(label='Product concentration')
    
    ax[1].plot(t,T, linewidth=linewidth)
    ax[1].plot(t, [-1.0]*len(t), linewidth=linewidth, color='black', linestyle='--')
    ax[1].plot(t, [1.0]*len(t), linewidth=linewidth, color='black', linestyle='--')
    ax[1].set_title(label='Temperature')
    
    ax[2].plot(t,rho, linewidth=linewidth)
    ax[2].plot(t, [-1.0]*len(t), linewidth=linewidth, color='black', linestyle='--')
    ax[2].plot(t, [1.0]*len(t), linewidth=linewidth, color='black', linestyle='--')
    ax[2].set_title(label='Production Volume')
    
    ax[3].plot(t,Fc, linewidth=linewidth)
    ax[3].plot(t, [-1.0]*len(t), linewidth=linewidth, color='black', linestyle='--')
    ax[3].plot(t, [1.0]*len(t), linewidth=linewidth, color='black', linestyle='--')
    ax[3].set_title(label='Coolant Flowrate')

    ax[4].plot(t,rewards, linewidth=linewidth)
    ax[4].set_title(label='Reward')

    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.grid(False)
    plt.xlabel('Time steps', fontsize=13)
    
    plt.tight_layout()
    if save_figure:
        plt.savefig(figure_path)
    if show_figure:
        plt.show()
    plt.close(fig)