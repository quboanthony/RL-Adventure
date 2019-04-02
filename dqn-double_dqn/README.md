[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Project 1: Navigation with Double DQN

### Introduction

For this project, you will train an agent to navigate (and collect bananas!) in a large, square world.  

![Trained Agent][image1]

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

### Getting Started

1. Follow the instructions in [this repository](https://github.com/openai/gym) to perform a minimal install of OpenAI gym.  
	- Next, install the **classic control** environment group by following the instructions [here](https://github.com/openai/gym#classic-control).
	- Then, install the **box2d** environment group by following the instructions [here](https://github.com/openai/gym#box2d).

2. Download the environment from one of the links below：
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.


3. Place the file in the DRLND GitHub repository, in the `p1_navigation/` folder, and unzip (or decompress) the file. 

4. Pytorch is installed with following command:

- conda install pytorch torchvision cuda100 -c pytorch

5. My basic working environment is as follows:

- Windows 10
- Python 3.5
- CUDA 10.0

### Instructions

In order to train the againt,execute the cells in sequences after "4.It's Your Turn!".


### Description

For Q-learning, there is a consequence of overestimation. Since every move we chose the max Q-value action only considering the next one step action, it is obviously 'optimistic'. Over-estimation in dqn was analyzed in the [article](https://arxiv.org/abs/1509.06461), it not necessarily cause problem but can sometimes lead to unstablization in some Atari game experiments. 

Double Q-learning comes to resolve this problem in dqn. The main idea is to modify the target function in dqn. The target function if dqn is orignally like this:

$$ Y^{DQN} = R_{t+1}+\gamma\max\limit_a Q(S_{t+1},a,\theta^{-})$$, where $\theta^{-}$ are the weights from the target network.

Double Q-learning tries to 'give a second thought' for this best action, and introduced another network in the best action decision:

$$ Y^{Double-DQN} = R_{t+1}+\gamma Q(S_{t+1},argmax(Q(S_{t+1},a,\theta)),\theta^{-})$$,where $\theta^{-}$ are the weights from the target network but $\theta$ is from another network, like the on-line network.



### Resources

- [Human-Level Control through Deep Reinforcement Learning](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)
- [Deep Reinforcement Learning with Double Q-Learning](https://arxiv.org/abs/1509.06461)
- [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581)
- [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952)