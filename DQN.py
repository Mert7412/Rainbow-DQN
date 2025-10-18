import numpy as np
from collections import deque

import gymnasium as gym
import ale_py

import torch
import torch.nn as nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from utils import ReplayMemory
from model import DQN

replay_memory = ReplayMemory(1000000)
writer = SummaryWriter("experiments/per_ddqn_experiment_1")

def select_action(state,episode):
    global eps
    eps = eps_min + (eps_max - eps_min) * np.exp(-decay*episode)

    rnd = np.random.random() 

    if rnd < eps:
        return env.action_space.sample()
    else:
        with torch.no_grad():
            return policy_network(state).max(1).indices.item()

def calculate_loss(batch_size = 256):
    if len(replay_memory) < batch_size:
        return
    
    state_batch,action_batch,reward_batch,next_state_batch,done_batch,wi,indices= replay_memory.sample(batch_size)

    weights_tensor = torch.tensor(wi, dtype=torch.float32).unsqueeze(1)

    sa_values = policy_network(state_batch).gather(1,action_batch)
   
    with torch.no_grad():
        best_actions = policy_network(next_state_batch).max(1).indices.unsqueeze(1)
        ns_values = target_network(next_state_batch).gather(1,best_actions)

    q_target = reward_batch + (1-done_batch) * (GAMMA * ns_values)
    q_loss = nn.functional.huber_loss(sa_values,q_target,weight = weights_tensor)
    
    td_error = (q_target - sa_values).detach().numpy()
    replay_memory.update_priorities(indices,td_error)

    return q_loss

def evaluate_model(num_ep,current_episode):
    total_reward = 0
    total_q_value = 0
    total_steps = 0 

    for i in range(num_ep):
        obs,info = env.reset()
  
        env.step(1)
        frames = deque([obs]*4,maxlen=4)
        terminate = False
        while not terminate:
            with torch.no_grad():
                current_state = np.array(frames)
                state_tensor = torch.tensor(current_state,dtype=torch.float32).unsqueeze(0).float().div(255)

                q_values = policy_network(state_tensor)
    

                max_q_result = q_values.max(1)
                action = max_q_result.indices.item()
                max_q_value_for_state = max_q_result.values.item()

                total_steps += 1
                next_state, reward, terminated, truncated, info = env.step(action)
                
                frames.append(next_state)

                done = terminated or truncated

                total_reward += reward
                total_q_value += max_q_value_for_state

                if done:
                    terminate = True


    avg_reward = total_reward / num_ep
    avg_q_value = total_q_value / total_steps

    writer.add_scalar(f"eval/average_rewards", avg_reward, current_episode)
    writer.add_scalar(f"eval/average_q_value", avg_q_value, current_episode)
    
    print(f"Evaluation at episode {current_episode}: Avg Reward: {avg_reward:.2f}, Avg Q-Value: {avg_q_value:.2f}")


NSTEPS = 10000000
GAMMA = 0.99
LR = 5e-5
TAU = 10000
eps_max = 1
eps_min = 0.01
eps = eps_max
eps_step = 1000000
decay = (-1/eps_step) * np.log(eps_min/eps_max)

gym.register_envs(ale_py)

env = gym.make('ALE/Breakout-v5',obs_type="grayscale",frameskip = 1) 
env = gym.wrappers.AtariPreprocessing(env,terminal_on_life_loss=True)
env = gym.wrappers.ClipReward(env,-1,1)
obs, info = env.reset(seed=44)

policy_network = DQN(4,4)
target_network = DQN(4,4)
target_network.load_state_dict(policy_network.state_dict())
optimizer = optim.AdamW(policy_network.parameters(),lr=LR)


def train(resume = False,chekpoint_path= None):
    global replay_memory

    nepisode = 0
    step = 0
    if resume :
        checkpoint = torch.load(chekpoint_path)
        policy_network.load_state_dict(checkpoint["policy_net"])
        target_network.load_state_dict(checkpoint["target_net"])
        nepisode = checkpoint["episode"]
        step = checkpoint["step"]
      
    losses = []
    log_interval = 1000
    evaluation_interval = 500
    while step < NSTEPS:   
        obs,info = env.reset()
        env.step(1)
        frames = deque([obs]*4,maxlen=4)
        terminate = False
        nepisode +=1
         
        while not terminate:

            current_state = np.array(frames)
            state_tensor = torch.tensor(current_state,dtype=torch.float32).unsqueeze(0).float().div(255)
            action = select_action(state_tensor,step)
        
            next_state, reward, terminated, truncated, info = env.step(action)
            frames.append(next_state)

            next_state = np.array(frames)
            next_state_tensor = torch.tensor(next_state,dtype=torch.float32).unsqueeze(0).float().div(255)
            done = terminated or truncated
            with torch.no_grad():
                
                sa_values = policy_network(state_tensor).gather(1,torch.tensor([[action]]))
                best_actions = policy_network(next_state_tensor).max(1).indices.unsqueeze(1)
                ns_values = target_network(next_state_tensor).gather(1,best_actions)

                q_target = reward + (1-done) * (GAMMA * ns_values)
                td_error = q_target - sa_values
          

            replay_memory.push(td_error.item(),frames[-1],action,reward,truncated or terminated)

            if done:
                terminate = True

            if len(replay_memory) < 1e4:
                continue

            step += 1
    
            loss = calculate_loss(32)

            if loss is not None:        
                losses.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_value_(policy_network.parameters(), 1.0) 
                optimizer.step()

            if step%TAU == 0:               
                for params,target_params in zip(policy_network.parameters(),target_network.parameters()):
                    target_params.data.copy_(params.data)

            if (len(losses) % log_interval) == 0:
                mean_loss = np.mean(losses)
                writer.add_scalar("train/loss",mean_loss,nepisode)
                print(f"episode: {nepisode} loss: {mean_loss} epsilon:{eps}  replay_size:{len(replay_memory)}")

                losses = []

        if len(replay_memory) < 1e4:
                continue
        if (nepisode % evaluation_interval) == 0:
            print(f"\n--- Running evaluation at episode {nepisode} ---")

            policy_network.eval() 
            evaluate_model(num_ep=10,current_episode=nepisode) 
            policy_network.train() 

            print("--- Evaluation complete, resuming training ---\n")
        
      
        if (nepisode % 250) == 0:
            print("\n--- Saving checkpoint ---")
            torch.save({
                "episode":nepisode,
                "step":step,
                "policy_net":policy_network.state_dict(),
                "target_net":target_network.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            },"Breakout_checkpoint.tar")
            print("--- Checkpoint saved ---\n")


train(True,"Breakout_checkpoint.tar")