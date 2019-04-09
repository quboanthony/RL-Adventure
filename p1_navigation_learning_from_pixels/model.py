import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        "*** YOUR CODE HERE ***"
        self.cv1=nn.Conv2d(3,32,8,4)
        self.cv2=nn.Conv2d(32,64,4,2)
        self.cv3=nn.Conv2d(64,64,3,1)
        self.fc4=nn.Linear(64*7*7,512)
        self.fc5=nn.Linear(512,action_size)
    def forward(self, state):
        """Build a network that maps state -> action values."""
        x=F.relu(self.cv1(state))
        x=F.relu(self.cv2(x))
        x=F.relu(self.cv3(x))
        x=x.view(x.size(0),-1)
        x=F.relu(self.fc4(x))
        return self.fc5(x)
