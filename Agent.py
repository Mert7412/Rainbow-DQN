import numpy as np

import torch
from torch import nn
from torch import optim

from model import DQN

class Agent:
    def __init__(self,nframes,naction):
        self.nactions = naction

        self.gamma =  0.99
        self.eps_max = 1
        self.eps_min = 0.01
        self.eps = self.eps_max
        eps_step = 1000000
        self.lr = 5e-5
        self.decay = (-1/eps_step) * np.log(self.eps_min/self.eps_max)
        

        self.policy_network = DQN(nframes,naction)
        self.target_network = DQN(nframes,naction)
        self.target_network.load_state_dict(self.policy_network.state_dict())

        self.optimizer = optim.AdamW(self.policy_network.parameters(),lr=self.lr)

    
    def select_action(self,state,current_step,evaluate = False):

        self.eps = self.eps_min + (self.eps_max - self.eps_min) * np.exp(-self.decay*current_step)

        rnd = np.random.random() 

        if rnd > self.eps or evaluate:
            with torch.no_grad():
                max_q_Values =  self.policy_network(state).max(1)
                action = max_q_Values.indices.item()
                if evaluate:
                    q_value = max_q_Values.values.item()
                    return action,q_value
                else:
                    return action
        else:
            return np.random.randint(0,self.nactions)

            
    def update_model(self,replay_memory,batch_size = 32):
        if len(replay_memory) < batch_size:
            return
        
        state_batch,action_batch,reward_batch,next_state_batch,done_batch,wi,indices= replay_memory.sample(batch_size)

        sa_values = self.policy_network(state_batch).gather(1,action_batch)
    
        with torch.no_grad():
            best_actions = self.policy_network(next_state_batch).max(1).indices.unsqueeze(1)
            ns_values = self.target_network(next_state_batch).gather(1,best_actions)

        q_target = reward_batch + (1-done_batch) * (self.gamma * ns_values)

        q_loss = nn.functional.huber_loss(sa_values,q_target,reduction="none")
        
        weights_tensor = torch.tensor(wi, dtype=torch.float32).unsqueeze(1)

        loss = (q_loss * weights_tensor).mean()

        td_error = (q_target - sa_values).detach().numpy()
        replay_memory.update_priorities(indices,td_error)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_network.parameters(), 1.0) 
        self.optimizer.step()

        return loss
            
    def update_target_netwok(self):
        self.target_network.load_state_dict(self.policy_network.state_dict())

    def save_checkpoint(self,path, episode, step):
        print("\n--- Saving checkpoint ---")
        torch.save({
            "episode":episode,
            "step":step,
            "policy_net":self.policy_network.state_dict(),
            "target_net":self.target_network.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
        },path)
        print("--- Checkpoint saved ---\n")

    def load_checkpoint(self,checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.policy_network.load_state_dict(checkpoint["policy_net"])
        self.target_network.load_state_dict(checkpoint["target_net"])
        episode = checkpoint["episode"]
        step = checkpoint["step"]
        return episode,step

    