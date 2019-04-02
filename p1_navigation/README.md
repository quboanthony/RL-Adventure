[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Project 1: Navigation

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

A typical DQN method is used in this project, states of the environments are used as input and actions are traited as outputs. 

The network is composed of 3 fully connected layers, the dimensions of these layers are 128,64,32.

The activate functions between the layers are relu function, the output layer has no activation functions.

### Training

Here we tried to train the agent with parameters as follows:

- n_episodes=1000
- eps_start=1.0
- eps_end=0.005
- eps_decay=0.985

We found that if the eps decays a little bit faster, the agent can learn faster to achieve an average score of 13;
Also, a deeper network and larger dimensions could help the agent learn better. For example, we have observed that with less nodes, the agent may just learn to avoid the blue banana but do not know how to go around it.

Finally, the environment was solved in 274 episodes, with an average score of 13.01. And we tested the model and it seems learn something, 15 score was achieved in a test run.