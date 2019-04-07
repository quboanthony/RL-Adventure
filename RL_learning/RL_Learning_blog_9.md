# Deep Reinforcement learning Learning blog 9 - Dueling DQN

## Intro

Double DQN improves the target computation, Prioritized Experience Replay improves the memory buffer mechanism. In this blog, I read the [Dueling DQN](https://arxiv.org/abs/1511.06581) paper, which improves the Q-netwrok architecture and changed the way of function approximation of the value function.

This blog mainly referenced the [Dueling DQN](https://arxiv.org/abs/1511.06581) paper, [Pinard's blog](https://www.cnblogs.com/pinard/p/9923859.html) and ['Kai Arulkumaran's blog](http://torch.ch/blog/2016/04/30/dueling_dqn.html).

## Dueling DQN Insight

As explained in the Dueling DQN paper, the key insight behind this architecture is that it is not necessary to estimate the value of each action choice every time regardless the states. The example showed in the paper:

![alt text](fig_blog_9/Enduro.png "Enduro")

With method proposed by [Simonyan *et al.*](https://arxiv.org/abs/1312.6034), they showed that the action taken only matters when the high possibility of collison eminents.

In order to let the agent to know the differences issued as above, the writer has proposed a new qnetwork design. The architecture of Dueling DQN has been through very thoughtful design. As for the example of NatureDQN, they modified the last second fully connected layer of the CNN into a combination of a *value function* $V$ and a *advantage function* $A$,

$$
Q(s,a;\theta,\alpha,\beta)=V(s;\theta,\beta)+A(s,a;\theta,\alpha)
$$

where $V(s;\theta,\beta)$ is the value function, $A(s,a;\theta,\alpha)$ is the advantage functions

$$
A^{\pi}(s,a)=Q^{\phi}(s,a)-V^{\pi}(s)
$$

$\alpha$ and $\beta$ means the parameters of the two streams of fully-connected layers, $\theta$ is the parameter of the convolution network as used in the NatureDQN. Note that the use of convolution network is not a prerequisite for the Dueling DQN. The Difference between the Dueling DQN and the normal DQN can be illustrated as the image from the paper:

![alt text](fig_blog_9/DuelDQN-Arch.png "DuelDQN-Arch")

However, as mentioned in the paper, a simple decomposition of the output layer like this may have the problem with 'indentifibility', which means that in many occasions, $V(s;\theta,\beta)$ and $V^{\pi}(s)$ may cancel out with eachother, and $Q(s,a;\theta,\alpha,\beta)$ just equals to $Q^{\phi}(s,a)$. In such case the decomposition is not working as expected.

Therefore, in the paper, the writer proposed a 'centralization operation' of the advantage function:

$$
Q(s,a;\theta,\alpha,\beta)=V(s;\theta,\beta)+A(s,a;\theta,\alpha)-\frac{1}{|A|}\Sigma_{a'}A(s,a';\theta,\alpha)
$$

## Model implementation in pytorch
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed,fc1_size=128,fc2_size=64,fc3_size=32):
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
        self.fc1=nn.Linear(state_size,fc1_size)
        self.fc2=nn.Linear(fc1_size,fc2_size)
        self.fc3=nn.Linear(fc2_size,fc3_size)
        self.val=nn.Linear(fc3_size,1)
        self.adv=nn.Linear(fc3_size,action_size)


    def forward(self, state):
        """Build a network that maps state -> action values."""
        x=F.relu(self.fc1(state))
        x=F.relu(self.fc2(x))
        x=F.relu(self.fc3(x))
        xval=self.val(x)
        xadv=self.adv(x)
        return xval-xadv.mean(1,keepdim=True)+xadv

```
