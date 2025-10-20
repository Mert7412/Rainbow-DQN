import gymnasium as gym
import ale_py

from collections import deque
import numpy as np
import torch

class Environment():
    def __init__(self,env_name,terminal_on_life_loss = True,clip_reward = True):
        
        gym.register_envs(ale_py)

        self.env = gym.make(env_name,obs_type="grayscale",frameskip = 1) 
        self.env = gym.wrappers.AtariPreprocessing(self.env,terminal_on_life_loss=terminal_on_life_loss)
        if clip_reward:
            self.env = gym.wrappers.ClipReward(self.env,-1,1)

        self.frames = deque([],maxlen=4)

    def _stack_to_tensor(self):
        current_state = np.array(self.frames)
        state_tensor = torch.tensor(current_state,dtype=torch.float32).unsqueeze(0).float().div(255)
        return state_tensor

    def step(self,action):

        next_obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(next_obs)

        done = terminated or truncated
        next_state_tensor = self._stack_to_tensor()

        return next_state_tensor,next_obs,reward,done

    def reset(self):
        obs,info = self.env.reset()
        self.env.step(1)
        self.frames.extend([obs]*4)

        return self._stack_to_tensor()

