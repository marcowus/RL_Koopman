# NOTE: If you load, deepcopy, or assign a model/ensemble, always call move_model_to_device(model, device) on both the ensemble and each submodel.
import numpy as np
import random
import torch as T
import pickle
from copy import deepcopy
from common import get_default_device

class MultiStepMemory():
    def __init__(self, max_size, input_shape, n_actions,
                 n_step_predictions, device=None,
                 save_dir=None, save_name=None, settings=None):
        self.settings = settings
        self.save_path = save_dir + save_name + '.pickle'
        self.mem_size = max_size
        self.device = device if device is not None else get_default_device()
        self.mem_cntr = 0
        self.indexlist = None
        self.return_counter = 0

        self.state_memory = np.zeros((self.mem_size, input_shape))
        self.new_state_memory = np.zeros((self.mem_size, input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)
        self.storage_memory = np.zeros(self.mem_size)

        self.n_step_predictions = n_step_predictions

    def store_transition(self, state, action, state_, done, storage=None):
        # Ensure all inputs are numpy arrays (on CPU)
        if isinstance(state, T.Tensor):
            state = state.detach().cpu().numpy()
        if isinstance(action, T.Tensor):
            action = action.detach().cpu().numpy()
        if isinstance(state_, T.Tensor):
            state_ = state_.detach().cpu().numpy()
        if isinstance(storage, T.Tensor):
            storage = storage.detach().cpu().numpy()
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.new_state_memory[index] = state_
        self.terminal_memory[index] = done
        self.storage_memory[index] = storage
        self.mem_cntr += 1
        self.indexlist = None

    def make_last_stored_transition_terminal(self):
        index = (self.mem_cntr-1) % self.mem_size
        self.terminal_memory[index] = True

    def store_timeseries(self, states, actions): # TODO: does not work with storages yet
        # Ensure states and actions are numpy arrays (on CPU)
        if isinstance(states, T.Tensor):
            states = states.detach().cpu().numpy()
        if isinstance(actions, T.Tensor):
            actions = actions.detach().cpu().numpy()
        for i in range(len(states)-2):
            self.store_transition(states[i], actions[i], states[i+1], False)
        self.store_transition(states[-2], actions[-2], states[-1], True)
        return None

    def _reset_remember_order(self, n_step_predictions=None):
        if n_step_predictions is None:
            n_step_predictions = self.n_step_predictions

        # find indices that are not acceptable as start-indices for trajectory prediction
        # (too close to the end of a trajectory)
        if self.mem_cntr < self.mem_size:
            terminal_indices = np.nonzero(self.terminal_memory[:self.mem_cntr-1])[0]
            terminal_indices = np.append(terminal_indices, self.mem_cntr-1)
        else:
            terminal_indices = np.nonzero(self.terminal_memory)[0]
            terminal_indices = np.append(terminal_indices, self.mem_size-1)
        almost_done = [np.array(range(i-n_step_predictions+1, i+1)) for i in terminal_indices]
        almost_done = np.unique(np.concatenate(almost_done))
        almost_done = almost_done[almost_done >= 0]

        # get random start indices
        self.return_counter = 0
        index_array = np.array(range(self.mem_cntr))\
            if self.mem_cntr < self.mem_size\
            else np.array(range(self.mem_size))
        self.indexlist = np.delete(index_array, almost_done)

        # get constant-control trajectories
        if self.settings['dream_model_type'] == 'PINN':
            new_control_every_n_steps = \
                int(self.settings['delta_t']['control'] / self.settings['delta_t']['timestep'])
            self.indexlist = self.indexlist[\
                self.indexlist % new_control_every_n_steps == 0]

        random.shuffle(self.indexlist)
        
        return None

    def _reset_remember_order_alternative(self):
        # get random start indices
        self.return_counter = 0
        self.indexlist = np.array(range(self.mem_cntr-self.n_step_predictions))\
            if self.mem_cntr < self.mem_size\
            else np.array(range(self.mem_size-self.n_step_predictions))
        random.shuffle(self.indexlist)
        return None

    def get_next_random_batch(self, batch_size,
                              as_tensor=True):
        if self.indexlist is None:
            self._reset_remember_order()

        if self.return_counter+batch_size > len(self.indexlist)-1:
            assert self.return_counter < len(self.indexlist)-1, "Memory is depleted"
            last_index = len(self.indexlist)-1
            memory_depleted = True
        else:
            last_index = self.return_counter+batch_size
            memory_depleted = False

        batch_start_indices = self.indexlist[self.return_counter:last_index]
        batch_indices = np.array([np.array(range(i, i+self.n_step_predictions)) for i in batch_start_indices])
        self.return_counter += batch_size

        states = self.state_memory[batch_indices]
        states_ = self.new_state_memory[batch_indices]
        actions = self.action_memory[batch_indices]
        dones = self.terminal_memory[batch_indices]

        if memory_depleted:
            self._reset_remember_order()

        if as_tensor:
            states = T.tensor(states, dtype=T.float32).to(self.device)
            actions = T.tensor(actions, dtype=T.float32).to(self.device)
            states_ = T.tensor(states_, dtype=T.float32).to(self.device)
            dones = T.tensor(dones, dtype=T.bool).to(self.device)

        return states, actions, states_, dones, memory_depleted

    def get_entire_memory(self, as_tensor=True):
        self._reset_remember_order()

        batch_start_indices = self.indexlist
        batch_indices = np.array([np.array(range(i, i+self.n_step_predictions)) for i in batch_start_indices])

        states = self.state_memory[batch_indices]
        states_ = self.new_state_memory[batch_indices]
        actions = self.action_memory[batch_indices]
        dones = self.terminal_memory[batch_indices]

        if as_tensor:
            states = T.tensor(states, dtype=T.float32).to(self.device)
            actions = T.tensor(actions, dtype=T.float32).to(self.device)
            states_ = T.tensor(states_, dtype=T.float32).to(self.device)
            dones = T.tensor(dones, dtype=T.bool).to(self.device)

        return states, actions, states_, dones

    def get_random_initial_state(self, as_tensor=True):
        if self.mem_cntr < self.mem_size:
            index = random.randint(0, self.mem_cntr-1)
        else:
            index = random.randint(0, self.mem_size-1)

        state = self.state_memory[index]
        storage = self.storage_memory[index]

        if as_tensor:
            state = T.tensor(state, dtype=T.float32).to(self.device)
            storage = T.tensor(storage, dtype=T.float32).to(self.device)

        return state, storage

    def get_single_random_batch(self, batch_size,
                                as_tensor=True):
        # not used in training
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        states_ = self.new_state_memory[batch]
        dones = self.terminal_memory[batch]

        if as_tensor:
            states = T.tensor(states, dtype=T.float32).to(self.device)
            actions = T.tensor(actions, dtype=T.float32).to(self.device)
            states_ = T.tensor(states_, dtype=T.float32).to(self.device)
            dones = T.tensor(dones, dtype=T.bool).to(self.device)

        return states, actions, states_, dones

    def save_memory(self, path=None):
        if path is None:
            path = self.save_path
        data = (self.state_memory[:self.mem_cntr,:],
                self.action_memory[:self.mem_cntr,:],
                self.new_state_memory[:self.mem_cntr,:],
                self.terminal_memory[:self.mem_cntr],
                self.mem_cntr)
        with open(path, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return None

    def load_memory(self, path=None):
        if path is None:
            path = self.save_path
        with open(path, 'rb') as handle:
            data = pickle.load(handle)

        self.mem_cntr = data[4]
        self.state_memory[:self.mem_cntr,:] = data[0]
        self.action_memory[:self.mem_cntr,:] = data[1]
        self.new_state_memory[:self.mem_cntr,:] = data[2]
        self.terminal_memory[:self.mem_cntr] = data[3]

        self._reset_remember_order()
        return None

    def is_empty(self):
        return self.mem_cntr == 0

def combine_memories(memory1, memory2):
    memory = deepcopy(memory1)
    memory.mem_cntr = memory1.mem_cntr + memory2.mem_cntr

    states = np.concatenate(
        (memory1.state_memory[:memory1.mem_cntr,:],
         memory2.state_memory[:memory2.mem_cntr,:]), axis=0)
    new_states = np.concatenate(
        (memory1.new_state_memory[:memory1.mem_cntr,:],
         memory2.new_state_memory[:memory2.mem_cntr,:]), axis=0)
    actions = np.concatenate(
        (memory1.action_memory[:memory1.mem_cntr,:],
         memory2.action_memory[:memory2.mem_cntr,:]), axis=0)
    terminals = np.concatenate(
        (memory1.terminal_memory[:memory1.mem_cntr],
         memory2.terminal_memory[:memory2.mem_cntr]), axis=0)
    storage = np.concatenate(
        (memory1.storage_memory[:memory1.mem_cntr],
         memory2.storage_memory[:memory2.mem_cntr]), axis=0)

    memory.state_memory[:memory.mem_cntr,:] = states
    memory.new_state_memory[:memory.mem_cntr,:] = new_states
    memory.action_memory[:memory.mem_cntr,:] = actions
    memory.terminal_memory[:memory.mem_cntr] = terminals
    memory.storage_memory[:memory.mem_cntr] = storage

    memory._reset_remember_order()
    return memory