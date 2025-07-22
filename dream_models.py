import torch as T
from torch.nn.utils.parametrizations import weight_norm
import random
from common import ElementwiseScaler, get_default_device
from scipy.stats import qmc
import constants as cs
import numpy as np
import torch.nn as nn
import math

# Utility function to recursively move a model and all submodules to a device

def move_model_to_device(model, device):
    model = model.to(device)
    for name, module in model.named_children():
        module.to(device)
    return model

# NOTE: If you load, deepcopy, or assign a model/ensemble, always call move_model_to_device(model, device) on both the ensemble and each submodel.
# This is now enforced in code via ensure_all_on_device methods and runtime assertions.
class Ensemble(T.nn.Module):
    def __init__(self, models, device=None):
        super().__init__()
        self.n_models = len(models)
        self.models = models
        self.device = device if device is not None else get_default_device()
        self.ensure_all_on_device(self.device)

    def ensure_all_on_device(self, device):
        self.to(device)
        for idx, model in enumerate(self.models):
            self.models[idx] = move_model_to_device(model, device)
        self.device = device

    def forward(self, x, u, model_idx=None):
        if model_idx is None:
            model_idx = random.randint(0, self.n_models-1)
        return self.models[model_idx].forward(x, u)

    def env_step(self, delta_t, x, u, model_idx=None):
        if model_idx is None:
            model_idx = random.randint(0, self.n_models-1)
        # Runtime assertion for device consistency
        assert x.device == self.models[model_idx].input_layer.weight.device, "Input and model are on different devices!"
        return self.models[model_idx].env_step(delta_t, x, u)

class MLP(T.nn.Module):
    def __init__(self, in_features, out_features, hidden_layer_sizes,
                 activation=T.nn.Tanh, output_activation=T.nn.Identity,
                 predict_delta=True, learning_rate=0.001,
                 use_weight_normalization=True, device=None):
        super().__init__()
        self.predict_delta = predict_delta
        self.num_hidden_layers = len(hidden_layer_sizes)
        self.use_weight_normalization = use_weight_normalization
        self.device = device if device is not None else get_default_device()
        # layers
        self.input_layer = T.nn.Linear(in_features=in_features, out_features=hidden_layer_sizes[0]).to(self.device)
        self.hidden_layers = T.nn.ModuleList(
            [T.nn.Linear(in_features=hidden_layer_sizes[i], out_features=hidden_layer_sizes[i+1]).to(self.device)
            for i in range(self.num_hidden_layers-1)]
        )
        self.output_layer = T.nn.Linear(in_features=hidden_layer_sizes[-1], out_features=out_features).to(self.device)
        # weight normalization
        if self.use_weight_normalization:
            self.input_layer = weight_norm(self.input_layer, dim=None)
            for i_layer in range(self.num_hidden_layers-1):
                self.hidden_layers[i_layer] = weight_norm(self.hidden_layers[i_layer], dim=None)
            self.output_layer = weight_norm(self.output_layer, dim=None)
        # activation functions
        self.activation = activation()
        self.output_activation = output_activation()
        # optimizer and loss function
        self.optimizer = T.optim.Adam(self.parameters(), lr=learning_rate)
        self.loss_function = T.nn.MSELoss()
        self.ensure_all_on_device(self.device)

    def ensure_all_on_device(self, device):
        self.to(device)
        self.device = device

    def forward(self, x, u):
        '''
        x: state
        u: action
        z: next state
        '''
        # concatenate state and action
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if u.dim() == 1:
            u = u.unsqueeze(0)
        assert x.shape[0] == u.shape[0], "x and u must have same batch size"
        z = T.cat([x, u], dim=1)
        # Runtime assertion for device consistency
        assert z.device == self.input_layer.weight.device, "Input and model are on different devices!"
        # pass through network
        z = self.activation(self.input_layer(z))
        for i_layer in range(self.num_hidden_layers-1):
            z = self.activation(self.hidden_layers[i_layer](z))
        z = self.output_activation(self.output_layer(z))
        if self.predict_delta:
            z = x + z
        return z

    def env_step(self, delta_t, x, u):
        # get n_steps to predict
        n_steps = int(delta_t['control']/delta_t['timestep'])

        # arrays to tensor
        x = T.tensor(x, dtype=T.float32).to(self.device)
        u = T.tensor(u, dtype=T.float32).to(self.device)

        # solution tensor
        X = T.zeros((n_steps+1, x.shape[0]), dtype=T.float32).to(self.device)
        X[0,:] = x

        # predict next states
        for i_step in range(n_steps):
            x = self.forward(x, u)
            X[i_step+1,:] = x

        # return in array form
        X_np = X.detach().cpu().numpy() if isinstance(X, T.Tensor) else X
        return X_np
    
# PINN Class
class PINN(T.nn.Module):
    def __init__(self, hidden_layer_sizes, data_idx, activation=T.nn.Tanh, output_activation=T.nn.Identity,
                 learning_rate=0.001, use_weight_normalization=True, idw_cycle=10, iter_adam=1000, iter_lbfgs=300, Np = 10000, Ni = 100, device=None):
        super().__init__()
        self.num_hidden_layers = len(hidden_layer_sizes)
        self.use_weight_normalization = use_weight_normalization
        self.device = device if device is not None else get_default_device()
        # layers
        self.input_layer = T.nn.Linear(in_features=5, out_features=hidden_layer_sizes[0], dtype=T.float32).to(self.device)
        self.hidden_layers = T.nn.ModuleList(
            [T.nn.Linear(in_features=hidden_layer_sizes[i], out_features=hidden_layer_sizes[i+1], dtype=T.float32).to(self.device)
            for i in range(self.num_hidden_layers-1)]
        )
        self.output_layer = T.nn.Linear(in_features=hidden_layer_sizes[-1], out_features=3, dtype=T.float32).to(self.device)
        #Initialization        
        T.nn.init.xavier_normal_(self.input_layer.weight.data, gain=nn.init.calculate_gain('tanh'))
        nn.init.zeros_(self.input_layer.bias.data)
        for i in range(self.num_hidden_layers-1):
            T.nn.init.xavier_normal_(self.hidden_layers[i].weight.data, gain=nn.init.calculate_gain('tanh'))
            nn.init.zeros_(self.hidden_layers[i].bias.data)        
        T.nn.init.xavier_normal_(self.output_layer.weight.data, gain=nn.init.calculate_gain('tanh'))
        nn.init.zeros_(self.output_layer.bias.data)
        # weight normalization
        if self.use_weight_normalization:
            self.input_layer = weight_norm(self.input_layer, dim=None)
            for i_layer in range(self.num_hidden_layers-1):
                self.hidden_layers[i_layer] = weight_norm(self.hidden_layers[i_layer], dim=None)
            self.output_layer = weight_norm(self.output_layer, dim=None)
        # scalers
        self.action_scaler = ElementwiseScaler(
                min_unscaled = T.tensor([0.8, 0.0], dtype=T.float32).to(self.device),
                max_unscaled = T.tensor([1.2, 700.0], dtype=T.float32).to(self.device),
                min_scaled = T.tensor([-1.0, -1.0], dtype=T.float32).to(self.device),
                max_scaled = T.tensor([1.0, 1.0], dtype=T.float32).to(self.device))
        self.state_scaler = ElementwiseScaler(
                min_unscaled = T.tensor([0.9*0.1367, 0.8*0.7293], dtype=T.float32).to(self.device),  # c_lower, T_lower
                max_unscaled = T.tensor([1.1*0.1367, 1.2*0.7293], dtype=T.float32).to(self.device),  # c_upper, T_upper
                min_scaled = T.tensor([-1.0, -1.0], dtype=T.float32).to(self.device),
                max_scaled = T.tensor([1.0, 1.0], dtype=T.float32).to(self.device))
        self.time_scaler = ElementwiseScaler(
                            min_unscaled = T.tensor([0.0], dtype=T.float32).to(self.device),
                            max_unscaled = T.tensor([1.0], dtype=T.float32).to(self.device),
                            min_scaled = T.tensor([-1.0], dtype=T.float32).to(self.device),
                            max_scaled = T.tensor([1.0], dtype=T.float32).to(self.device))      
        # activation functions
        self.activation = activation()
        self.output_activation = output_activation()    
        # optimizers and loss function
        self.adam = T.optim.Adam(self.parameters(), lr=learning_rate)
        self.lbfgs = T.optim.LBFGS(self.parameters(), line_search_fn='strong_wolfe')
        self.loss_function = T.nn.MSELoss()           
        self.ensure_all_on_device(self.device)
        # Add all attributes used elsewhere
        self.iter_adam = iter_adam
        self.iter_lbfgs = iter_lbfgs
        self.idw_cycle = idw_cycle
        self.Np = Np
        self.Ni = Ni
        self.idx = data_idx
        self.weights = T.ones(3)
        self.t_init, self.x_init, self.u_init, self.y_init = self.generate_init_data()
        self.X_pinn = self.generate_pinn_data()
        self.t_init = self.t_init.to(self.device)
        self.x_init = self.x_init.to(self.device)
        self.u_init = self.u_init.to(self.device)
        self.y_init = self.y_init.to(self.device)
        self.X_pinn = T.from_numpy(self.X_pinn).float().to(self.device)

    def ensure_all_on_device(self, device):
        self.to(device)
        self.device = device

    def forward(self, t, x, u):
        # Runtime assertion for device consistency
        assert t.device == self.input_layer.weight.device, "Input and model are on different devices!"
        device = self.input_layer.weight.device
        t = t.to(self.device)
        x = x.to(self.device)
        u = u.to(self.device)
        # concatenate time, state and action
        assert t.shape[0] == x.shape[0] == u.shape[0], "t, x and u must have same batch size"
        # scale inputs  
        t = self.time_scaler.scale(t)
        x = self.state_scaler.scale(x)
        u = self.action_scaler.scale(u)
        # concantenate inputs
        z = T.cat([t, x, u], dim=1)      
        # pass through network
        z = self.activation(self.input_layer(z))
        for i_layer in range(self.num_hidden_layers-1):
            z = self.activation(self.hidden_layers[i_layer](z))
        z = self.output_layer(z)
        return z 
    
    def generate_init_data(self):        
           
        #generate collocation points for initial condition training   
        sampler = qmc.LatinHypercube(d=cs.lb_pinn.size-1)
        sample = sampler.random(self.Ni)  
        sample = qmc.scale(sample,cs.lb_pinn[1:],cs.ub_pinn[1:])      
        X_xu_init = T.from_numpy(sample).float().to(self.device)
        t_init = T.zeros(self.Ni, dtype=T.float32).to(self.device)       
        t_init = t_init.unsqueeze(1)
        
        u_init = X_xu_init[:,0:2]
        x_init = X_xu_init[:,2:4]
        y_init = x_init
        
        return t_init, x_init, u_init, y_init

    def generate_pinn_data(self):

        #generate collocation points for physics loss training        
        sampler = qmc.LatinHypercube(d=cs.lb_pinn.size)   
        sample = sampler.random(self.Np) 
        sample = qmc.scale(sample,cs.lb_pinn,cs.ub_pinn)    

        return sample

    def batch_pinn_data(self, batch_cycle, batch_counter, full_batch = True):
        if full_batch or batch_cycle == 0:    
            # t_pinn = T.from_numpy(self.X_pinn[:,0]).float().unsqueeze(1)  # Replaced for GPU compatibility
            t_pinn = self.X_pinn[:,0].unsqueeze(1)
            # u_pinn = T.from_numpy(self.X_pinn[:,1:3]).float()  # Replaced for GPU compatibility
            u_pinn = self.X_pinn[:,1:3]
            # x_pinn = T.from_numpy(self.X_pinn[:,3:5]).float()  # Replaced for GPU compatibility
            x_pinn = self.X_pinn[:,3:5]
        else:
            batch_size = math.floor(self.X_pinn.shape[0]/batch_cycle)
            if batch_counter == 0:
                # np.random.shuffle(self.X_pinn)  # Replaced for GPU compatibility
                idx = T.randperm(self.X_pinn.shape[0], device=self.X_pinn.device)
                self.X_pinn = self.X_pinn[idx]
            start = batch_counter*batch_size
            if batch_counter == batch_cycle:
                end = self.X_pinn.shape[0]
            else:
                end = (batch_counter+1)*batch_size
            # t_pinn = T.from_numpy(self.X_pinn[start:end,0]).float().unsqueeze(1)  # Replaced for GPU compatibility
            t_pinn = self.X_pinn[start:end,0].unsqueeze(1)
            # u_pinn = T.from_numpy(self.X_pinn[start:end,1:3]).float()  # Replaced for GPU compatibility
            u_pinn = self.X_pinn[start:end,1:3]
            # x_pinn = T.from_numpy(self.X_pinn[start:end,3:5]).float()  # Replaced for GPU compatibility
            x_pinn = self.X_pinn[start:end,3:5]
        return t_pinn, x_pinn, u_pinn

    # Physics loss term
    def loss_physics(self, times, init_states, controls):        
        t = times.clone()
        t.requires_grad = True

        # forward pass
        c, Temp, R = T.t(self.forward(t, init_states, controls))
                 
        # take derivative
        c_t = T.autograd.grad(c, t, T.ones_like(c), create_graph=True)[0]
        Temp_t =  T.autograd.grad(Temp, t, T.ones_like(Temp), create_graph=True)[0]
        
        # ensure all constants are tensors on the correct device
        V = T.tensor(cs.V, dtype=controls.dtype, device=controls.device) if not isinstance(cs.V, T.Tensor) else cs.V.to(controls.device)
        Tf = T.tensor(cs.Tf, dtype=Temp.dtype, device=Temp.device) if not isinstance(cs.Tf, T.Tensor) else cs.Tf.to(Temp.device)
        alpha = T.tensor(cs.alpha, dtype=Temp.dtype, device=Temp.device) if not isinstance(cs.alpha, T.Tensor) else cs.alpha.to(Temp.device)
        Tc = T.tensor(cs.Tc, dtype=Temp.dtype, device=Temp.device) if not isinstance(cs.Tc, T.Tensor) else cs.Tc.to(Temp.device)

        # calculate RHS
        f1 = (controls[:,0]/V)*(1 - c) - R
        f2 = (controls[:,0]/V)*(Tf - Temp) + R - controls[:,1]*alpha*(Temp-Tc)          

        # ensure all tensors are on the same device for loss_function
        c_t = c_t.to(self.device)
        f1 = f1.unsqueeze(1).to(self.device)
        Temp_t = Temp_t.to(self.device)
        f2 = f2.unsqueeze(1).to(self.device)

        return 1/2*(self.loss_function(c_t, f1) + self.loss_function(Temp_t, f2))    
        
    # Initial condition loss term
    def loss_init(self):
        #calculate initial condition loss
        return self.loss_function(self.forward(self.t_init, self.x_init, self.u_init)[:,self.idx], self.y_init)
    
    # Data loss term
    def loss_data(self, times, init_states, controls, out_states):        
        #calculate data loss 
        return self.loss_function(self.forward(times, init_states, controls)[:,self.idx], out_states)        
           
    def env_step(self, delta_t, x, u):
        # get n_steps to predict
        n_steps = int(delta_t['control']/delta_t['timestep'])
        times = (T.linspace(0, delta_t['control']/3600, n_steps+1, dtype=T.float32).unsqueeze(1)).to(self.device)
        # arrays to tensor
        x = T.tensor(x, dtype=T.float32).unsqueeze(0).repeat(times.shape[0],1).to(self.device)
        u = T.tensor(u, dtype=T.float32).unsqueeze(0).repeat(times.shape[0],1).to(self.device)
        # Runtime assertion for device consistency
        assert x.device == self.input_layer.weight.device, "Input and model are on different devices!"
        x = self.state_scaler.unscale(x)
        u = self.action_scaler.unscale(u)
        # solution tensor
        X = T.zeros((n_steps+1, x.shape[0]), dtype=T.float32).to(self.device)
        # predict next states       
        X = self.forward(times, x, u)[:,:2]
        X[0,:] = x[0,:]
        X = self.state_scaler.scale(X)
        # return in array form
        X_np = X.detach().cpu().numpy() if isinstance(X, T.Tensor) else X
        return X_np
    
class CSTRParameters:
    # default: 'dopri5', options: 'dopri5', 'rk4' and more
    # see: https://github.com/rtqichen/torchdiffeq
    def __init__(self, delta_t, integration_method):
        self.V       = 20.0
        self.k       = 300.0 / (60*60.0)
        self.N       = 5.0
        self.T_f     = 0.3947
        self.alpha_c = 1.95e-04
        self.T_c     = 0.3816
        self.tau_1   = 4.84
        self.tau_2   = 14.66

        self.integration_method = integration_method
        if self.integration_method == 'rk4':
            self.odeint_options = dict(step_size = delta_t)
        elif self.integration_method == 'dopri5':
            self.odeint_options = None
        else:
            raise ValueError("Invalid integration method!")