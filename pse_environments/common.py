# NOTE: If you load, deepcopy, or assign a model/ensemble, always call move_model_to_device(model, device) on both the ensemble and each submodel.
import numpy as np
import random
import os
import pandas as pd
import matplotlib.pyplot as plt
import torch as T
from common import get_default_device

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

# Electricity Prices
class Prices_generator():
    def __init__(self):
        train_prices_path = os.path.join(os.path.dirname(__file__), 'resources/train_prices.xlsx')
        train_prices = pd.read_excel(train_prices_path)
        train_prices = train_prices.loc[:,'AT_price_day_ahead'].to_numpy()

        test_prices_path = os.path.join(os.path.dirname(__file__), 'resources/test_prices.xlsx')
        test_prices = pd.read_excel(test_prices_path)
        test_prices = test_prices.loc[:,'AT_price_day_ahead'].to_numpy()

        self.prices = {'train': train_prices,
                       'test':  test_prices}

    def generate_timeseries(self, length, train_or_test = 'train'):
        price_trajectory = self.prices[train_or_test]
        startindex = np.random.randint(0,len(price_trajectory)-1-length)
        return price_trajectory[startindex:startindex+length]

# Episode plotting
def plot_episode(observations, actions, rewards,infos,model_name,
                 objective_type, process_name,
                 save_path, show):
    
    
    if objective_type == "DemandResponse":

        costs = []
        steady_state_costs = []
        n_constr_viols = 0
        penalties = []
        
        for i in range(len(infos)):
            costs.append(infos[i]["true_cost"])
            steady_state_costs.append(infos[i]["steady_state_cost"])
            n_constr_viols += infos[i]["constraint_violation"]
            penalties.append(infos[i]["penalty"])
        relative_cost = sum(costs)/sum(steady_state_costs)
        
        if process_name == "CSTR1":
            observations = np.stack(observations, axis=0)
            actions = np.stack(actions, axis=0)
            rewards = np.stack(rewards, axis=0)

            c = observations[:-1,0]
            T = observations[:-1,1]
            storage = observations[:-1,2]
            price = observations[:-1,3]
            rho = actions[:,0]
            Fc = actions[:,1]
            t = list(range(len(c)))

            violation_rate = n_constr_viols/len(rho)
            mean_violation = np.mean(penalties)

            fig, ax = plt.subplots(6, 1, sharex=True, sharey=False, figsize=(10,10))
            linewidth = 0.8
            if model_name is not None:
                fig.suptitle(f'{process_name}, {model_name}, Relative cost: {round(relative_cost*100,1)}%, Mean rel. violation: {round(mean_violation,3)}, Violation rate: {round(violation_rate*100,1)}%', fontsize=14)
            else:
                fig.suptitle(f'{process_name}, Relative cost: {round(relative_cost*100,1)}%, Mean rel. violation: {round(mean_violation,3)}, Violation rate: {round(violation_rate*100,1)}%', fontsize=14)
            
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
            ax[4].plot(t, [0.0]*len(t), linewidth=linewidth, color='black', linestyle='--')
            ax[4].set_title(label='Stored product (in hours of nominal production)')

            ax[5].plot(t,rewards, linewidth=linewidth)
            ax[5].set_title(label='Reward')

            fig.add_subplot(111, frameon=False)
            plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
            plt.grid(False)
            plt.xlabel('Time steps', fontsize=13)
            
            plt.tight_layout()
            if save_path is not None:
                plt.savefig(save_path)
            if show:
                plt.show()
            plt.close(fig)