import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        #self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        self.memory=self.PMemory(BUFFER_SIZE)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        transition=np.hstack((state,action,reward,next_state,done))
        self.memory.store(transition)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                b_idx,experiences,ISWeights = self.memory.sample(BATCH_SIZE)
                self.learn(b_idx,experiences, ISWeights, GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))


    def weighted_mse_loss(self,input,target,weights):
        '''

        Return the weighted mse loss to be used by Prioritized experience replay



        :param input: torch.Tensor.

        :param target: torch.Tensor.

        :param weights: torch.Tensor.

        :return loss:  torch.Tensor.

        '''

        # source: http://
    
        # forums.fast.ai/t/how-to-make-a-custom-loss-function-pytorch/9059/20
    
        out = (input-target)**2
    
        out = out * weights.expand_as(out)
    
        loss = out.mean(0)  # or sum over whatever dimensions
    
        return loss
    
    def learn(self, b_idx, experiences, ISWeights, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        ## TODO: compute and minimize the loss
        "*** YOUR CODE HERE ***"
        #Double dqn: the action is determined by q-learning rule with online qnetwork, and the Q value of this action is obtained by target qnetwork
        arg_maxs=self.qnetwork_local(next_states).detach().max(1)[1].unsqueeze(1)# for torch object, .max()[1] returns the argmax, .max()[0] returns the max value
        
        Q_targets_next=self.qnetwork_target(next_states).gather(1,arg_maxs)
        
        Q_targets=rewards+(gamma*Q_targets_next*(1-dones))
        
        Q_expected=self.qnetwork_local(states).gather(1,actions)
        
        td_errors=Q_expected-Q_targets
        
        self.memory.batch_update(b_idx, td_errors)
        
        Weights=torch.tensor(ISWeights,device=device,dtype=torch.float).reshape(-1, 1)
        
        loss = self.weighted_mse_loss(Q_expected,Q_targets,Weights)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


#class ReplayBuffer:
#    """Fixed-size buffer to store experience tuples."""
#
#    def __init__(self, action_size, buffer_size, batch_size, seed):
#        """Initialize a ReplayBuffer object.
#
#        Params
#        ======
#            action_size (int): dimension of each action
#            buffer_size (int): maximum size of buffer
#            batch_size (int): size of each training batch
#            seed (int): random seed
#        """
#        self.action_size = action_size
#        self.memory = deque(maxlen=buffer_size)  
#        self.batch_size = batch_size
#        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
#        self.seed = random.seed(seed)
#    
#    def add(self, state, action, reward, next_state, done):
#        """Add a new experience to memory."""
#        e = self.experience(state, action, reward, next_state, done)
#        self.memory.append(e)
#    
#    def sample(self):
#        """Randomly sample a batch of experiences from memory."""
#        experiences = random.sample(self.memory, k=self.batch_size)
#
#        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
#        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
#        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
#        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
#        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
#  
#        return (states, actions, rewards, next_states, dones)
#
#    def __len__(self):
#        """Return the current size of internal memory."""
#        return len(self.memory)
            
class SumTree():
    data_insert_point=0
    def __init__(self,capacity):
        # capacity: dimension of the sample space
        self.capacity=capacity
        # the bottom leaves are all the samples, which has the dimension of capacity.
        # the parent nodes has the dimension of capacity-1
        self.tree=np.zeros(2*capacity-1)
        # The replay data buffer
        self.data=np.zeros(capacity,dtype=object)

    
    def add(self,p,data):
        #locate the data position in the tree
        tree_idx=self.data_insert_point+self.capacity-1
        #store the transition into the replay buffer
        self.data[self.data_insert_point]=data
        #update the tree accoding to the new priority of data
        self.update(tree_idx,p)
        #if the data has exceeded the capacity, score it from the beginning.
        self.data_insert_point+=1
        if self.data_insert_point>self.capacity:
            self.data_insert_point=0

    def update(self,tree_idx,p):
        #when the TD error change for the sample, we would like to update the related child leaf and its parents in SumTree.
        change=p-self.tree[tree_idx]
    
        self.tree[tree_idx]=p
    
        while tree_idx!=0:
            tree_idx=(tree_idx-1)//2 #locate back to its parent leaf
            self.tree[tree_idx]+=change
    
    def get_leaf(self,v):
        # We would like to get the actual sample when we have drew a number such as v from an uniform distribution. Here we can get the sample's index in the tree, the sample's TD error, and the sample itself.
        parent_node=0
        while True:
            child_node_l=parent_node*2+1
            child_node_r=child_node_l+1
            if child_node_l>=len(self.tree):
                leaf_idx=parent_node
            else:
                if v<=self.tree[child_node_l]:
                    parent_node=child_node_l
                else:
                    v-=self.tree[child_node_l]
                    parent_node=child_node_r
    
        data_idx=leaf_idx-self.capacity+1
        return leaf_idx,self.tree[leaf_idx],self.data[data_idx]
    
    def total_p(self):
        #get the total TD-error,which is located on the root of SumTree.
        return self.tree[0]
    

class PMemory():
    """

    This Memory class is modified based on the original code from:

    https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
    https://github.com/ljpzzz/machinelearning/blob/master/reinforcement-learning/ddqn_prioritised_replay.py

    """
    epsilon=0.01
    alpha=0.6
    beta=0.4
    beta_increment_per_sampling=0.001
    abs_err_upper =1.
    
    def __init__(self,capacity):
        self.tree=SumTree(capacity)
        
    def store(self,transition):
        max_p=np.max(self.tree.tree[-self.tree.capacity:])
        if max_p==0:
            max_p=self.abs_err_upper
        self.tree.add(max_p,transition)
    
    def sample(self,n):
        b_idx,b_memory, ISWeights=np.empty((n,),dtype=np.int32),np.empty((n,self.tree.data[0].size)),np.empty((n,1))
        pri_seg=self.tree.total_p/n
        self.beta=np.min([1.,self.beta+self.beta_increment_per_sampling])
        
        min_prob=np.min(self.tree.tree[-self.tree.capacity:])/self.tree.total_p
        if min_prob==0:
            min_prob=0.00001
            
        for i in range(n):
            a,b=pri_seg*i,pri_seg*(i+1)
            v=np.random.uniform(a,b)
            idx,p,data=self.tree.get_leaf(v)
            prob=p/self.tree.total_p
            ISWeights[i,0]=np.power(prob/min_prob,-self.beta)
            b_idx[i]=idx
            b_memory[i,:]=data
            
        states = torch.from_numpy(np.vstack([d[0] for d in b_memory if d is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([d[1] for d in b_memory if d is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([d[2] for d in b_memory if d is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([d[3] for d in b_memory if d is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([d[4] for d in b_memory if d is not None]).astype(np.uint8)).float().to(device)
        return b_idx,(states,actions,rewards,next_states,dones), ISWeights
    
    def batch_update(self, tree_idx, abs_errors):
        abs_errors+=self.epsilon #avoid zero
        
        clipped_errors=np.minimum(abs_errors,self.abs_err_upper)
        ps=np.power(clipped_errors,self.alpha)
        for ti,p in zip(tree_idx,ps):
            self.tree.update(ti,p)
            
            
    
            
    