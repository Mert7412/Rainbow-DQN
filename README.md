# **Rainbow DQN Implementation 🌈**

This repository contains a PyTorch implementation of a Deep Q-Network (DQN) agent designed to master the Atari game Breakout from pixel data. The project's ultimate goal is to build the full "Rainbow" agent by progressively integrating its seven core components.

**Current Status:** This agent successfully implements and combines two critical improvements over the original DQN:

* **Double Q-Learning (DDQN)**  
* **Prioritized Experience Replay (PER)**

The agent is configured to train on ALE/Breakout-v5 and includes detailed logging, evaluation, and checkpointing.

## **About the Project**

The "Rainbow" paper demonstrated that combining several independent improvements to DQN could lead to state-of-the-art performance. This repository is a step-by-step journey to replicate that success.

### **Implemented Features**

* \[✅\] **Base DQN (2015):** The foundational algorithm using a deep neural network, experience replay, and a target network.  
* \[✅\] **Double Q-Learning (DDQN):** Reduces the overestimation bias of Q-values by decoupling action selection (using the online network) from value evaluation (using the target network). This is implemented in the calculate\_loss function.  
* \[✅\] **Prioritized Experience Replay (PER):** Instead of uniform sampling, this agent prioritizes transitions with a high TD-error, leading to significantly more efficient learning. This is implemented in utils.py using a SumTree data structure and is integrated into the training loop with importance-sampling weights.  
* \[⬜️\] **Dueling Network Architecture**  
* \[⬜️\] **Multi-step Learning**  
* \[⬜️\] **Distributional Q-Learning (C51)**  
* \[⬜️\] **Noisy Nets for Exploration**

## **Code Structure**

The project is organized into three main files:

* DQN.py: The main script for training and evaluating the agent. It contains the training loop, environment setup, hyperparameter definitions, and logging logic.  
* model.py: Defines the DQN class, which is a convolutional neural network (CNN) based on the architecture used in the original DeepMind paper.  
* utils.py: Contains the ReplayMemory and SumTree classes, which together provide a sophisticated implementation of Prioritized Experience Replay.

## Installation

To get started, clone the repository and install the required dependencies.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Mert7412/Rainbow-DQN.git
    cd Rainbow-DQN
    ```

2.  **Create a virtual environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install torch numpy gymnasium ale-py tensorboard
    ```

## **Usage**

### **Training the Agent**

All hyperparameters and settings are configured directly within DQN.py. To start training, simply run the script:

python DQN.py

* **Logs:** Training progress, including loss and evaluation metrics, will be saved in the experiments/per\_ddqn\_experiment\_1 directory.  
* **Checkpoints:** The model will be saved periodically to a file named Breakout\_checkpoint.tar.

### **Resuming Training**

The script is configured by default to resume from the last saved checkpoint. To start a fresh training run, you can modify the last line of DQN.py from:
```bash
# Resumes training from a checkpoint  
train(True, "Breakout\_checkpoint.tar")
```

to:
```bash
# Starts a new training session  
train(False)
```

### **Monitoring with TensorBoard**

You can visualize the agent's performance in real-time using TensorBoard. Run the following command in your terminal from the project's root directory:
```bash
tensorboard \--logdir=experiments
```
This will launch a web server where you can view charts for training loss, average evaluation rewards, and average Q-values.

## **Implementation Details**

### **Model Architecture (model.py)**

The network is a CNN that takes a stack of 4 grayscale frames (84x84) as input. The architecture for the Breakout environment is as follows:

| Layer Type | Filters / Output Units | Kernel Size | Stride | Activation |
| :---- | :---- | :---- | :---- | :---- |
| Conv2d | 32 | 8x8 | 4 | ReLU |
| Conv2d | 64 | 4x4 | 2 | ReLU |
| Conv2d | 64 | 3x3 | 1 | ReLU |
| Flatten | \- | \- | \- | \- |
| Linear | 256 | \- | \- | ReLU |
| Linear | 256 | \- | \- | ReLU |
| Linear | 4 (Action Space) | \- | \- | None |

### **Prioritized Experience Replay (utils.py)**

The ReplayMemory class uses a SumTree to enable efficient sampling of experiences based on their priority (TD-error). The sample method is particularly noteworthy as it:

* Performs proportional sampling based on calculated priorities.  
* Computes importance-sampling (IS) weights to correct for the bias introduced by prioritized sampling.  
* Correctly assembles stacks of 4 frames for both state and next\_state, carefully handling episode boundaries by masking frames that cross into a new episode.

### **DDQN \+ PER Loss Calculation (DQN.py)**

The calculate\_loss function is where the DDQN and PER components merge:

1. **DDQN:** It uses the policy\_network to *select* the best action for the next state but uses the target\_network to *evaluate* the Q-value of that action. This prevents the upward bias common in standard Q-learning.  
2. **PER:** The calculated importance-sampling weights (wi) from the replay buffer are applied to the Huber loss, ensuring that the gradient updates are scaled appropriately.  
3. **Priority Updates:** After the loss is calculated, the new TD-errors are used to update the priorities of the sampled experiences in the SumTree.

## Paper References

**Rainbow: Combining Improvements in Deep Reinforcement Learning**
- **Authors:** Matteo Hessel, Joseph Modayil, Hado van Hasselt, Tom Schaul, Georg Ostrovski, Will Dabney, Dan Horgan, Bilal Piot, Mohammad Azar, David Silver
- **Link:** [https://arxiv.org/abs/1710.02298](https://arxiv.org/abs/1710.02298)

---

**Prioritized Experience Replay**
- **Authors:** Tom Schaul, John Quan, Ioannis Antonoglou, David Silver
- **Link:** [https://arxiv.org/abs/1511.05952](https://arxiv.org/abs/1511.05952)
