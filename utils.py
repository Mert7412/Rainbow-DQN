import numpy as np

import torch

class ReplayMemory:
    def __init__(self,capacity,frame_size=(84,84)):
        self.frames = np.zeros((capacity,*frame_size),dtype=np.uint8)
        self.actions = np.zeros((capacity),dtype=np.uint8)
        self.rewards = np.zeros((capacity))
        self.dones = np.zeros((capacity),dtype=np.uint8)

        self.priorities = SumTree(capacity)

        self.capacity = capacity
        self.idx = 0
        self.size = 0

        self.e = 1e-5
        self.a = 0.6
        self.b = 0.4
        self.b_inc = 6e-7

    def calculate_priority(self,td_error):
        return (np.abs(td_error) + self.e) ** self.a

    def push(self,td_error,state,action,reward,done):
        self.frames[self.idx] = state
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.dones[self.idx] = done
        self.priorities.add(self.idx,self.calculate_priority(td_error))

        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size+1,self.capacity)

    def update_priorities(self,data_indices,td_error):
        new_p = self.calculate_priority(td_error)
        
        for i in range(len(data_indices)):
            old_p = self.priorities.get_by_idx(data_indices[i])
            change = new_p[i] - old_p
            tree_idx = data_indices[i] + self.priorities.cap -1
            self.priorities.update_tree(tree_idx,change)


    def sample(self,batch_size,stack_size = 4):
        if self.size < stack_size:
            return
        
        segment = self.priorities.total() / batch_size
        indices = []
        priorities = []

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)

            p = np.random.uniform(a,b)
            data_idx, priority = self.priorities.get(p)
            indices.append(data_idx)
            priorities.append(priority)
        
        priorities = np.array(priorities)

        probs = priorities / self.priorities.total()
      

        wi = (self.size * probs) ** (-self.b) #importance-sampling (IS) weights
        wi = wi/np.max(wi)

        self.b = min(1.,self.b + self.b_inc)
   
        indices = np.array(indices).reshape((batch_size,1))
    
        cols = np.arange(stack_size)  
        state_indices = (np.ones((batch_size,stack_size),dtype=int) * indices) - cols[::-1]

        done_frames = self.dones[state_indices]
        mask = np.zeros_like(done_frames)
        mask = np.cumsum(done_frames[:,::-1],axis=1)[:,::-1] > 0

        state = self.frames[state_indices].copy()
        state[mask] = 0

        dones = self.dones[indices]
        actions = self.actions[indices]
        rewards = self.rewards[indices]

        next_state_indices = state_indices + 1

        done_frames = self.dones[next_state_indices]
        mask = np.zeros_like(done_frames)
        mask = np.cumsum(done_frames[:,::-1],axis=1)[:,::-1] > 0

        next_state = self.frames[next_state_indices].copy()
        next_state[mask] = 0
       
        state_batch = torch.from_numpy(state).float().div(255)
        action_batch = torch.from_numpy(actions).long()
        reward_batch = torch.from_numpy(rewards).float()
        next_state_batch = torch.from_numpy(next_state).float().div(255)
        done_batch = torch.from_numpy(dones).int()

        return state_batch,action_batch,reward_batch,next_state_batch,done_batch,wi,indices

    def __len__(self):
        return self.size

class SumTree:
    def __init__(self,capacity):
        self.cap = capacity

        self.tree = np.zeros(2 * capacity -1)

        self.size = 0

    def total(self):
        return self.tree[0]
    
    def get_by_idx(self,data_idx):
        tree_idx = data_idx + self.cap -1
        return self.tree[tree_idx]

    def update_tree(self,node_idx,change):
        parent_idx = (node_idx -1) // 2

        self.tree[parent_idx] += change

        if parent_idx != 0:
            self.update_tree(parent_idx,change)

    def add(self,data_idx,priority):
        tree_idx = data_idx + self.cap -1

        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        
        self.update_tree(tree_idx,change)

    def get(self,priority):
        def _find_index(idx,p):
            left = 2 * idx + 1
            right = left + 1

            if left >= len(self.tree):
                return idx

            if p <= self.tree[left]:
                return _find_index(left,p)
            else:
                p -= self.tree[left]
                return _find_index(right,p)
        
        tree_idx = _find_index(0,priority)
        data_idx = tree_idx - self.cap +1

        return data_idx,self.tree[tree_idx]


