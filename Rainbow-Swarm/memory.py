import random
from collections import namedtuple
import torch
import numpy as np

Transition = namedtuple(
    'Transition', ('timestep', 'state', 'action', 'reward', 'nonterminal'))
blank_trans = Transition(0, {'cnn':torch.zeros(
    9,3,128,128, dtype=torch.float32),'oth':torch.zeros(9,11,dtype=torch.float32)}, torch.zeros(9,dtype=torch.int64), torch.zeros(9,dtype=torch.float32), False)


# Segment tree data structure where parent node values are sum/max of children node values
class SegmentTree():
    def __init__(self, size):
        self.index = 0
        self.size = size
        self.full = False  # Used to track actual capacity
        # Initialise fixed size tree with all (priority) zeros
        self.sum_tree = [0] * (2 * size - 1)
        self.data = [None] * size  # Wrap-around cyclic buffer
        self.max = 1  # Initial max value to return (1 = 1^ω)

    # Propagates value up tree given a tree index
    def _propagate(self, index, value):
        parent = (index - 1) // 2
        left, right = 2 * parent + 1, 2 * parent + 2
        self.sum_tree[parent] = self.sum_tree[left] + self.sum_tree[right]
        if parent != 0:
            self._propagate(parent, value)

    # Updates value given a tree index
    def update(self, index, value):
        self.sum_tree[index] = value  # Set new value
        self._propagate(index, value)  # Propagate value
        self.max = max(value, self.max)

    def append(self, data, value):
        self.data[self.index] = data  # Store data in underlying data structure
        self.update(self.index + self.size - 1, value)  # Update tree
        self.index = (self.index + 1) % self.size  # Update index
        self.full = self.full or self.index == 0  # Save when capacity reached
        self.max = max(value, self.max)

    # Searches for the location of a value in sum tree
    def _retrieve(self, index, value):
        left, right = 2 * index + 1, 2 * index + 2
        if left >= len(self.sum_tree):
            return index
        elif value <= self.sum_tree[left]:
            return self._retrieve(left, value)
        else:
            return self._retrieve(right, value - self.sum_tree[left])

    # Searches for a value in sum tree and returns value, data index and tree index
    def find(self, value):
        index = self._retrieve(0, value)  # Search for index of item from root
        data_index = index - self.size + 1
        # Return value, data index, tree index
        return (self.sum_tree[index], data_index, index)

    # Returns data given a data index
    def get(self, data_index):
        return self.data[data_index % self.size]

    def total(self):
        return self.sum_tree[0]


class ReplayMemory():
    def __init__(self, args, capacity):
        self.device = args.device
        self.capacity = capacity
        self.history = args.history_length
        self.discount = args.discount
        self.n = args.multi_step
        # Initial importance sampling weight β, annealed to 1 over course of training
        self.priority_weight = args.priority_weight
        self.priority_exponent = args.priority_exponent
        self.t = 0  # Internal episode timestep counter
        # Store transitions in a wrap-around cyclic buffer within a sum tree for querying priorities
        self.transitions = SegmentTree(capacity)

    # Adds state and action at time t, reward and terminal at time t + 1
    def append(self, state, action, reward, terminal):
        # Only store last frame and discretise to save memory
        # Store new transition with maximum priority
        self.transitions.append(Transition(
            self.t, state, action, reward, not terminal), self.transitions.max)
        self.t = 0 if terminal else self.t + 1  # Start new episodes with t = 0

    # Returns a transition with blank states where appropriate
    def _get_transition(self, idx):
        transition = [None] * (self.history + self.n)
        transition[self.history - 1] = self.transitions.get(idx)
        for t in range(self.history - 2, -1, -1):  # e.g. 2 1 0
            if transition[t + 1].timestep == 0:
                transition[t] = blank_trans  # If future frame has timestep 0
            else:
                transition[t] = self.transitions.get(
                    idx - self.history + 1 + t)
        for t in range(self.history, self.history + self.n):  # e.g. 4 5 6
            if transition[t - 1].nonterminal:
                transition[t] = self.transitions.get(
                    idx - self.history + 1 + t)
            else:
                transition[t] = blank_trans  # If prev (next) frame is terminal
        return transition

    # Returns a valid sample from a segment
    def _get_sample_from_segment(self, segment, i):
        valid = False
        while not valid:
            # Uniformly sample an element from within a segment
            sample = random.uniform(i * segment, (i + 1) * segment)
            # Retrieve sample from tree with un-normalised probability
            prob, idx, tree_idx = self.transitions.find(sample)
            # Resample if transition straddled current index or probablity 0
            if (self.transitions.index - idx) % self.capacity > self.n and (idx - self.transitions.index) % self.capacity >= self.history and prob != 0:
                valid = True  # Note that conditions are valid but extra conservative around buffer index 0

        # Retrieve all required transition data (from t - h to t + n)
        transition = self._get_transition(idx)
        # Create un-discretised state and nth next state
        # state = torch.stack([trans.state for trans in transition[:self.history]]).to(
        #     dtype=torch.float32, device=self.device).div_(255)
        # next_state = torch.stack([trans.state for trans in transition[self.n:self.n +
        #                                                               self.history]]).to(dtype=torch.float32, device=self.device).div_(255)
        
        state = [trans.state for trans in transition[:self.history]]
        next_state = [trans.state for trans in transition[self.n:self.n + self.history]]
        # Discrete action to be used as index
        action = torch.tensor(
            transition[self.history - 1].action, dtype=torch.int64, device=self.device)
        # Calculate truncated n-step discounted return R^n = Σ_k=0->n-1 (γ^k)R_t+k+1 (note that invalid nth next states have reward 0)
        a = torch.stack([self.discount ** n * transition[self.history + n - 1].reward for n in range(self.n)])
        R = torch.tensor(a.sum(dim=0), dtype=torch.float32, device=self.device)
        # Mask for non-terminal nth next states
        nonterminal = torch.tensor(
            [transition[self.history + self.n - 1].nonterminal], dtype=torch.float32, device=self.device)
        # print("return batch",i)
        return prob, idx, tree_idx, state, action, R, next_state, nonterminal

    def sample(self, batch_size):
        # Retrieve sum of all priorities (used to create a normalised probability distribution)
        p_total = self.transitions.total()
        # Batch size number of segments, based on sum over all probabilities
        segment = p_total / batch_size
        print("get sample:")
        batch = [self._get_sample_from_segment(segment, i) for i in range(
            batch_size)]  # Get batch of valid samples
        print("sample batch:")
        # print(batch)
        probs, idxs, tree_idxs, states, actions, returns, next_states, nonterminals = zip(
            *batch)
        agent_count = len(states[0][0]['cnn'])
        replayRobotIndex = np.random.randint(0, agent_count, batch_size)
        states_cnn = torch.zeros(batch_size,4*3,128,128)
        states_oth = torch.zeros(batch_size,44)
        next_states_cnn = torch.zeros(batch_size,4*3,128,128)
        next_states_oth = torch.zeros(batch_size,44)
        actions_ = torch.zeros(batch_size,4)
        returns_ = torch.zeros(batch_size,4)
        for i in range(batch_size):
            states_cnn[i] = torch.cat([states[i][n]['cnn'][replayRobotIndex[i]] for n in range(4)])
            states_oth[i] = torch.cat([states[i][n]['oth'][replayRobotIndex[i]] for n in range(4)])
            next_states_cnn[i] = torch.cat([next_states[i][n]['cnn'][replayRobotIndex[i]] for n in range(4)])
            next_states_oth[i] = torch.cat([next_states[i][n]['oth'][replayRobotIndex[i]] for n in range(4)])
            actions_[i] = torch.tensor([actions[i][replayRobotIndex[n]] for n in range(4)])
            returns_[i] = torch.tensor([returns[i][replayRobotIndex[n]] for n in range(4)])
        states = {'cnn':states_cnn,'oth':states_oth}
        next_states = {'cnn':next_states_cnn,'oth':next_states_oth}
        actions, returns, nonterminals = actions_[:,3], returns_[:,3], torch.stack(nonterminals)
        # Calculate normalised probabilities
        probs = torch.tensor(probs, dtype=torch.float32,
                             device=self.device) / p_total
        capacity = self.capacity if self.transitions.full else self.transitions.index
        # Compute importance-sampling weights w
        weights = (capacity * probs) ** -self.priority_weight
        # Normalise by max importance-sampling weight from batch
        weights = weights / weights.max()
        return tree_idxs, states, actions, returns, next_states, nonterminals, weights

    def update_priorities(self, idxs, priorities):
        priorities.pow_(self.priority_exponent)
        [self.transitions.update(idx, priority)
         for idx, priority in zip(idxs, priorities)]

    # Set up internal state for iterator
    def __iter__(self):
        self.current_idx = 0
        return self

    # Return valid states for validation
    def __next__(self):
        if self.current_idx == self.capacity:
            raise StopIteration
        # Create stack of states
        state_stack = [None] * self.history
        state_stack[-1] = self.transitions.data[self.current_idx].state
        prev_timestep = self.transitions.data[self.current_idx].timestep
        for t in reversed(range(self.history - 1)):
            if prev_timestep == 0:
                # If future frame has timestep 0
                state_stack[t] = blank_trans.state
            else:
                state_stack[t] = self.transitions.data[self.current_idx +
                                                       t - self.history + 1].state
                prev_timestep -= 1
        state = torch.stack(state_stack, 0).to(
            dtype=torch.float32, device=self.device).div_(255)  # Agent will turn into batch
        self.current_idx += 1
        return state
