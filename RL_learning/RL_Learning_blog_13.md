# Deep Reinforcement learning Learning blog 13 - Proximal Policy Optimization
## Intro
In  the previous blog, we summarized the idea of REINFORCE algorthm, in which we update the policy  function  weights by gradient information of expected reward function.

However, this most basic policy gradient method has several  problems  and we will  show   what they are  and the important tweaks in addition to it.

Some of these  important tweaks lead us to the proximal policy  Optimization (PPO) and Trust  Region Policy Optimization (TRPO). These tweaks have allowed faster and more stable learning.

This blog mainly referenced to the Udacity course and the orginal paper of [PPO](https://arxiv.org/abs/1707.06347)


## Beyond Reinforcement

Key ingredients of REINFORCE algorthm:

First, initialize a random policy $\pi_{\theta}(a;s)$, and using the policy to collect a trajectory, i.e. a list of (state, actions, rewards) at each time step:

$$
s_1,a_1,r_1,s_2,a_2,r_2,\cdots
$$
Second, we compute the total reward of the trajectory $R=r_1+r_2+r_3+\cdots$, and compute an estimate the gradient of the expected rewards,$g$:
$$
g=R\Sigma_t\nabla_\theta \log\pi_\theta(a_t|s_t)
$$
Third, we update our policy using gradient  ascent with learning rate $\alpha$:
$$
\theta \leftarrow \theta+\alpha g
$$

The process then repeats.

There are some main problems of REINFORCE:
1. The update process is very **inefficient**. The policy was ran once, update once, and then it was threw away.

2.  The gradient estimate $g$ is very **noisy**. By chance  the collected trajectory may not be representative of the policy.

3. There is no clear **credit assignment**. A trajectory may contain many good/bad actions and whether these actions are reinforced depends only  on the final total output.

## Noise Reduction

In REINFORCE algorthm, we opimize the policy by maximizing the expected rewards $U(\theta)$. To achieve this, the gradient is given by an average over all the possible trajectories,

$$
\nabla_\theta U(\theta)=\overbrace{\Sigma_{\tau}P(\tau;\theta)}^{average\space over\space all\space trajectories}\underbrace{\left(R_{\tau}\Sigma_{t}\nabla_\theta\log\pi_{\theta}(a^{(\tau)}_t | s^{(\tau)}_{t})\right)}_{only\space one\space is\space  sampled}
$$

Instead of using millions or  infinite of trajectories as noted  by the mathematic equation, we  simply  take  one trajectory to compute the gradient and update our policy.

Thus, this alternative makes our update comes down to chance, sometimes the only collected trajectory simply does not contain useful information. The hope is that after traininig for a long time, the tiny signal accumulates.

The easiest option to reduce the noise in gradient is to simply sample more trajectories. Using distributed computing, we can collect multiple trajectories in parallel. Then we can  estimate the policy gradient by averaging across all the different trajectories.

$$
\left .\begin{matrix}
s^{(1)}_{t}, & a^{(1)}_{t}, & r^{(1)}_{t} \\
s^{(2)}_{t}, & a^{(2)}_{t}, & r^{(2)}_{t} \\
s^{(3)}_{t}, & a^{(3)}_{t}, & r^{(3)}_{t} \\
& \vdots &
\end{matrix} \right\}\rightarrow g=\frac{1}{N}\Sigma^{N}_{i=1}R_i\Sigma_{t}\nabla_\theta\log\pi_{\theta}(a^{(i)}_{t}|s^{(i)}_{t})
$$

## Rewars Normalization

There  is another bonus for running multiple trajectories:  we can  collect all the total rewards  and get a sense of how they are distributed.

In many cases, the distribution of rewards shifts as learning  happens.  Reward= 1  might be really  good in the beginning, but really bad after 1000 training episodes.

Learning can  be improved if we normalize the rewards,
$$
R_i\leftarrow\frac{R_i-\mu}{\sigma} \\
\mu=\frac{1}{N}\Sigma^N_i  R_i \\
\sigma=\sqrt{\frac{1}{N}\Sigma^N_i  (R_i-\mu)^2}
$$
where $\mu$ is the mean, and $\sigma$ is the standard deviation. When all the $R_i$ are the same, $\sigma=0$, we can set all the normalized rewards to 0 to avoid numerical problems.

Intuitively, normalizing the rewards  roughly  corresponds to picking half the actions to encourage/discourage, while also making sure the steps for  gradient ascents  are not too large/small.

##  Credit assignment

Going back to the gradient estimate, we can take a closer look at the total reward $R$, which is just a sum of reward at each step $R=r_1+r_2+r_3+\cdots+r_{t-1}+r_{t}+\cdots$
$$
g=\Sigma_{t}(\cdots+r_{t-1}+r_{t}+\cdots)\nabla_{\theta}\log\pi_{\theta}(a_{t}|s_{t})
$$

Let's think about what happens at time-step t. Even before anction is decided, the agent has already received all the rewards up until step $t-1$.  So we can think of that part of the total rewardas  the reward  from the past. The rest is denoted as the future reward.
$$
(\overbrace{\cdots+r_{t_1}}^{R^{past}_{t}}\overbrace{+r_{t}+\cdots}^{R^{future}_{t}})
$$
Because we have a Markov process,  the action at  time-step $t$ can  only affect  the future reward, so the past reward  shouldn't be contributing to the policy gradient. So to properly assign credit to the action  $a_t$, we should ignore the past reward. So a better  policy gradient would simply  have the future reward as the coefficient.
$$
g=\Sigma_{t}R^{future}_{t}\nabla_\theta\log\pi_{\theta}(a_{t}|s_{t})
$$

## Notes on Gradient Modification
It turns out that mathematically, ignoring past rewards might change the gradient for each  specific trajectory, but it doesn't change the averaged  gradient. So even through the gradient is different during training, on average we are still maximizing the average reward. In  fact, the resultant gradient is  less  noisy. So training using future rewards should speed things up.

## Importance Sampling
In the REINFORCE algorthm, we start with a policy, $\pi_\theta$, then using this policy to generate one or multiple trajectories $(s_t,a_t,r_t)$ to reduce noise. Afterwards, we compute a policy gradient, $g$, and update $\theta'\leftarrow\theta+\alpha g$.

At this point, the trajectories we've just generated are simply thrown away.  If we want to update our policy again, we would need to generate new trjectories once more. using the updated policy.

In fact, we need to compute  the gradient for the current policy, and to do that the trajectories need to be representativeof the current policy.

But we could just reuse  the recycled trajectories to compute gradients, and update the policy, agnain and again.

This is where importance sampling comes in. if we consider the trajectories collected by the old policy $P(\tau;\theta)$. And just by  chance, this trajectory can be collected by another new policy,  with a different probability $P(\tau;\theta')$

If we want to compute the average of some quantity, say $f(\tau)$. We could simply generate trajectories from the new policy, compute $f(\tau)$ and average them. mathematically it looks like

$$
\Sigma_{\tau} P(\tau;\theta')f(\tau)
$$

Now we could rearrange this equation, by multiplying and dividing by the same  number, $P(\tau;\theta)$ and rearrange the terms.

$$
\Sigma_{\tau}\overbrace{P(\tau;\theta)}^{sampling\space under\space old\space policy\space  \pi_{\theta}}\overbrace{\frac{P(\tau;\theta')}{P(\tau;\theta)}}^{re-weighting\space factor} f(\tau)
$$

written in this  way we can reinterpret the fist part as the coefficient for  sampling under  the old policy, with an extra re-weighting factor,in addition  to just averaging.

Intuitively, this tells us we  can use old trajectories for  computing averages  for new policy,  as long as we add this extra re-weighting factor, that takes  into account how under or over-represented each trajectory is  under  the new policy compared  to the  old  one.
