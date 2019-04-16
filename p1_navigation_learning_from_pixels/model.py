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
        self.cv1=nn.Conv3d(3,32,kernel_size=(1,3,3),stride=(1,3,3))
        self.bn1 = nn.BatchNorm3d(32)
        self.cv2=nn.Conv3d(32,64,kernel_size=(1,3,3),stride=(1,3,3))
        self.bn2 = nn.BatchNorm3d(64)
        self.cv3=nn.Conv3d(64,64,kernel_size=(1,3,3),stride=(1,3,3))
        self.bn3= nn.BatchNorm3d(64)
        conv_out_size=self._get_conv_out_size(state_size)
        self.fc4=nn.Linear(conv_out_size,512)
        self.fc5=nn.Linear(512,action_size)
    def forward(self, state):
        """Build a network that maps state -> action values."""
        x=F.relu(self.bn1(self.cv1(state)))
        x=F.relu(self.bn2(self.cv2(x)))
        x=F.relu(self.bn3(self.cv3(x)))
        x=x.view(x.size(0),-1)
        x=F.relu(self.fc4(x))
        return self.fc5(x)

    
    def _get_conv_out_size(self, shape):

        x = torch.rand(shape)
        
        x=F.relu(self.bn1(self.cv1(x)))
        x=F.relu(self.bn2(self.cv2(x)))
        x=F.relu(self.bn3(self.cv3(x)))

        n_size = x.data.view(1, -1).size(1)

        print('Convolution output size:', n_size)

        return n_size