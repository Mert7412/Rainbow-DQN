import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter

from Memory import ReplayMemory
from Environment import Environment
from Agent import Agent

NSTEPS = 10000000
TAU = 10000

env = Environment("ALE/Breakout-v5")
agent = Agent(4,4)
replay_memory = ReplayMemory(1000000)
writer = SummaryWriter("experiments/per_ddqn_experiment_1")

def evaluate_model(num_ep,current_step):
    total_reward = 0
    total_q_value = 0
    total_steps = 0 

    for i in range(num_ep):
        state_tensor = env.reset()

        terminate = False
        while not terminate:
            
            action,q_value = agent.select_action(state_tensor,current_step,True)

            next_state_tensor, next_obs, reward, done = env.step(action)

            state_tensor = next_state_tensor

            total_reward += reward
            total_q_value += q_value

            total_steps +=1

            terminate = done

    avg_reward = total_reward / num_ep
    avg_q_value = total_q_value / total_steps

    writer.add_scalar(f"eval/average_rewards", avg_reward,current_step)
    writer.add_scalar(f"eval/average_q_value", avg_q_value,current_step)
    
    print(f"Evaluation at step {current_step}: Avg Reward: {avg_reward:.2f}, Avg Q-Value: {avg_q_value:.2f}")

def train(resume = False,checkpoint_path= None):
    episode_counter = 0
    step_counter = 0

    if resume:
        episode_counter,step_counter = agent.load_checkpoint(checkpoint_path)

    losses = []

    log_interval = 1000
    evaluation_interval = 5000
    checkpoint_interval = 5000
    while step_counter < NSTEPS:
        state_tensor = env.reset()

        terminate = False
        episode_counter +=1

        while not terminate:
            action = agent.select_action(state_tensor,step_counter)

            next_state_tensor, next_obs, reward, done = env.step(action)

            with torch.no_grad():
                
                sa_values = agent.policy_network(state_tensor).gather(1,torch.tensor([[action]]))
                best_actions = agent.policy_network(next_state_tensor).max(1).indices.unsqueeze(1)
                ns_values = agent.target_network(next_state_tensor).gather(1,best_actions)

                q_target = reward + (1-done) * (agent.gamma * ns_values)
                td_error = q_target - sa_values
            
            replay_memory.push(td_error.item(),next_obs,action,reward,done)

            state_tensor = next_state_tensor
          
            terminate = done

            if len(replay_memory) < 1e4:
                continue

            step_counter += 1

            loss = agent.update_model(replay_memory,32)
            losses.append(loss.item())

            if (step_counter % TAU) == 0:               
                agent.update_target_netwok()

            if (len(losses) % log_interval) == 0:
                mean_loss = np.mean(losses)
                writer.add_scalar("train/loss",mean_loss,episode_counter)
                print(f"episode: {episode_counter} loss: {mean_loss} epsilon:{agent.eps}  replay_size:{len(replay_memory)}")

                losses = []

            if (step_counter % evaluation_interval) == 0:
                print(f"\n--- Running evaluation at step {step_counter} ---")

                agent.policy_network.eval() 
                evaluate_model(num_ep=10,current_step=step_counter) 
                agent.policy_network.train() 

                print("--- Evaluation complete, resuming training ---\n")

            if (step_counter % checkpoint_interval) == 0:
                agent.save_checkpoint(checkpoint_path,episode_counter,step_counter)

train(checkpoint_path="Breakout_checkpoint.tar")



