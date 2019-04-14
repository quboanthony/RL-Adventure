# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 15:55:33 2019

@author: 1
"""

from unityagents import UnityEnvironment
import numpy as np

class CollectBanana():
    def __init__(self,envname_path):
        self.base=UnityEnvironment(envname_path)
        self.brain_name=self.base.brain_names[0]
        self.brain=self.base.brains[self.brain_name]
        self.action_size=self.brain.vector_action_space_size
        self.train_mode=True
        self.last_frame=None
        self.last2_frame=None
        self.last3_frame=None
        self.reset()
        self.state_size=self.state.shape
        
    
    def get_state(self):
        frame=np.transpose(self.env_info.visual_observations[0],(0,3,1,2))[:,:,:,:]
        frame_size=frame.shape
        nframes=4
        self.state=np.zeros((1,frame_size[1],nframes,frame_size[2],frame_size[3]))
        self.state[0,:,0,:,:]=frame
        if not(self.last_frame is None):
            self.state[0,:,1,:,:]=self.last_frame
        if not(self.last2_frame is None):
            self.state[0,:,2,:,:]=self.last2_frame
        if not(self.last3_frame is None):
            self.state[0,:,3,:,:]=self.last3_frame
        self.last3_frame=self.last2_frame
        self.last2_frame=self.last_frame
        self.last_frame=frame
        
    def reset(self):
        self.env_info=self.base.reset(train_mode=self.train_mode)[self.brain_name]
        self.get_state()
        return self.state
    
    def step(self,action):
        self.env_info=self.base.step(action)[self.brain_name]
        self.get_state()
        reward=self.env_info.rewards[0]
        done=self.env_info.local_done[0]
        return self.state,reward,done, None
    
    def close(self):
        self.base.close()