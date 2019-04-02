# -*- coding: utf-8 -*-
"""
Spyder Editor

This is the Multiagent file
"""

from ddpg_agent import Agent, ReplayBuffer
import numpy as np


NUM_AGENTS=2            # number of agents
BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
UPDATE_EVERY=2          # every steps of learning
GAMMA = 0.99            # discount factor

class MultiAgents():
    
    def __init__(self, state_size, action_size, random_seed):
        """Initialize an MultiAgents object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        
        self.state_size=state_size
        self.action_size=action_size
        
        
        self.t_step=0
        
        self.ddpg_agents=[Agent(state_size=self.state_size, action_size=self.action_size, random_seed=random_seed) for _ in range(NUM_AGENTS)]
        
        # create shared replay buffer
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
        
    def act(self, states):
        actions=[agent.act(np.expand_dims(states,axis=0)) for agent, states in zip(self.ddpg_agents, states)]
        return actions
    
    def reset(self):
        for agent in self.ddpg_agents:
            agent.reset()
        
    def step(self, states, actions, rewards, next_states, dones):
        
        for state,action,reward,next_state,done in zip(states,actions,rewards,next_states,dones):
            self.memory.add(state,action,reward,next_state,done)
            
        self.t_step=self.t_step+1
        
        if self.t_step%UPDATE_EVERY==0:
            if len(self.memory)>BATCH_SIZE:
                for agent in self.ddpg_agents:
                    #shared replay buffer
                    experiences=self.memory.sample()
                    
                    agent.learn(experiences,GAMMA)
                    


