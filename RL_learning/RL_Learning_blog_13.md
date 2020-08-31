# Deep Reinforcement learning Learning blog 13 - Proximal Policy Optimization
## Intro
In  the previous blog, we summarized the idea of REINFORCE algorthm, in which we update the policy  function  weights by gradient information of expected reward function.

However, this most basic policy gradient method has several  problems  and we will  show   what they are  and the important tweaks in addition to it.

Some of these  important tweaks lead us to the proximal policy  Optimization (PPO) and Trust  Region Policy Optimization (TRPO). These tweaks have allowed faster and more stable learning.

This blog mainly referenced to the Udacity course and the orginal paper of [PPO](https://arxiv.org/abs/1707.06347)

## Constrainted  policy gradient methods (PPO)
Sometimes we would like to explore more good policies but we would not like the training process to forget totally the acceptable policy that we already have.

We can add constraint to our gradient-based algorithms, that differences between two policies at most some threshold $\delta$.
$$
J(\theta)=E_{\pi}[R(\tau)] \\
D(\pi(\cdot,\cdot,\theta),\pi(\cdot,\cdot,\theta'))\leq\delta
$$

The penalty term
$$
J(\theta)=E_{\pi}[R(\tau)]]-\beta D(\pi(\cdot,\cdot,\theta),\pi(\cdot,\cdot,\theta'))
$$
One of the measurement to calculate this distribution difference is KL-Divergence
$$
D_{KL}(p||q)=\int^{+\infty}_{-\infty}p(x)\log\frac{p(x)}{q(x)}dx
$$

## PPO (constrainted policy gradient method), beyond Reinforcement (Monte Carlo)

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
(\overbrace{\cdots+r_{t-1}}^{R^{past}_{t}}\overbrace{+r_{t}+\cdots}^{R^{future}_{t}})
$$
Because we have a Markov process,  the action at  time-step $t$ can  only affect  the future reward, so the past reward  shouldn't be contributing to the policy gradient. So to properly assign credit to the action  $a_t$, we should ignore the past reward. So a better  policy gradient would simply  have the future reward as the coefficient.
$$
g=\Sigma_{t}R^{future}_{t}\nabla_\theta\log\pi_{\theta}(a_{t}|s_{t})
$$

## Notes on Gradient Modification
It turns out that mathematically, ignoring past rewards might change the gradient for each  specific trajectory, but it doesn't change the averaged  gradient. So even though the gradient is different during training, on average we are still maximizing the average reward. In  fact, the resultant gradient is  less  noisy. So training using future rewards should speed things up.

## Importance Sampling
In the REINFORCE algorthm, we start with a policy, $\pi_\theta$, then using this policy to generate one or multiple trajectories $(s_t,a_t,r_t)$ to reduce noise. Afterwards, we compute a policy gradient, $g$, and update $\theta'\leftarrow\theta+\alpha g$.

At this point, the trajectories we've just generated are simply thrown away.  If we want to update our policy again, we would need to generate new trjectories once more, using the updated policy.

In fact, we need to compute the gradient for the current policy, and to do that the trajectories need to be representative of the current policy.

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

## The re-weighting factor

When we take a  closer look at the re-weighting factor.
$$
\frac{P(\tau;\theta')}{P(\tau;\theta)}=\frac{\pi_{\theta'}(a_1|s_1)\pi_{\theta'}(a_2|s_2)\pi_{\theta'}(a_3|s_3)\cdots}{\pi_{\theta}(a_1|s_1)\pi_{\theta}(a_2|s_2)\pi_{\theta}(a_3|s_3)\cdots}
$$

Statistically, each probability here contains a chain of products of each policy at different time-step, as the equation showed above.

If we estimate $P(\tau;\theta')$ and $P(\tau;\theta)$ using this equation without any treatement, there will be some problem.

For instance, when some policy gets quite close to 0, the re-weighting factor can become close to zero or infinity. This will make the re-weighting trick unreliable.

In order to cope with the problem, a trick by introducing a surrogate function was used in PPO.

## The surrogate Function (Proximal Policy)

Let's take a look at our policy gradient again, by re-writing the derivative of the log term, we have:

$$
g=\frac{P(\tau;\theta')}{P(\tau;\theta)}\sum_{t}\frac{\nabla_{\theta'}\pi_{\theta'}(a_t|s_t)}{\pi_{\theta'}(a_t|s_t)} R_{t}^{future}
$$

Then, by re-arranging these equations, we replace the $\frac{P(\tau;\theta')}{P(\tau;\theta)}$ term with the chains of interactions.

$$
g=\sum_{t}\frac{\cdots\pi_{\theta'}(a_t|s_t)\cdots}{\cdots\pi_{\theta}(a_t|s_t)\cdots}\frac{\nabla_{\theta'}\pi_{\theta'}(a_t|s_t)}{\pi_{\theta'}(a_t|s_t)} R_{t}^{future}
$$

Now, the idea of proximal policy comes in. Here we assume that if the old and the current policy is close enough to each other, we would like to treat all the factors in the "..." as a number very close to 1, and only leave the terms shown as below:

$$
g=\sum_{t}\frac{\nabla_{\theta'}\pi_{\theta'}(a_t|s_t)}{\pi_{\theta}(a_t|s_t)} R_{t}^{future}
$$

At last, we approximate the re-weighted factor with the output of the two policies by the same $(a_t,s_t)$.

With this approximated gradient, we can think of it as the gradient of a new object, called the surrogate function

$$
g=\nabla_{\theta'}L_{sur}(\theta',\theta)
$$

$$
L_{sur}(\theta',\theta)=\sum_{t}\frac{\pi_{\theta'}(a_t|s_t)}{\pi_{\theta}(a_t|s_t)} R_{t}^{future}
$$

therefore, we aim to maximize this surrogate function with the proximal gradient.

Now there left another important issue, if the two distributions of the policy differs too much, the assumptions that we made previously does not valid anymore. Hence there must be some constraints to avoid that from happening. 

To cope with this, one of the solutions is to introduce KL Divergence as regularization.

# KL Divergence as regularization (TRPO and PPO)

In the original papers of TRPO(predecessor of PPO) and PPO, despite all the complex mathmatical derivatioins, they mainly tried to add KL divergence between policies as constraint of optimization.

In PPO, the optimization objective is like this
$$
L_{PPO}^{\theta'}(\theta)=L^{\theta'}(\theta)-\beta KL(\theta,\theta')
$$

in which it makes the KL divergence term as a differentiable regularization term, and this makes the opimization process a bit easier.

In TRPO, it treat the KL divergence as a more general constraint condition, which is mathmatically more precise but very hard to optimize.

$$
L_{PPO}^{\theta'}(\theta)=L^{\theta'}(\theta)
$$
$$
KL(\theta,\theta')<\delta
$$

In practice, researchers found that the approach of PPO has very similar results as TRPO, yet the former is relatively much easier to implement. 



# Clipping Policy Update (PPO-2)

In PPO-2, we replace the KL divergence regularization with clipping Policy Update.

When the policy distribution differs to much, it is highly like that the policy in which you sample your trajectories will lead to cliff jumping effect in the current learning policy. In such case, it could be impossible to jump out of the bad policy plateau.

Hence, an intuitive idea to deal with cliff jumping is to give restrictions to surrogate function.

The formula of clipped surrogate function is 
$$
L_{sur}^{clip}(\theta,\theta')=\sum_{t}\min\{ \frac{\pi_{\theta'}}{\pi_{\theta}}R_{t}^{future},clip_{\epsilon}(\frac{\pi_{\theta'}}{\pi_{\theta}})R_{t}^{future}\}
$$

We want to make sure the two policy $\theta,\theta'$ is similar, which means their ratio is close to 1. Therefore $\epsilon$ usually be choosen as a small number.

The whole Clipping surrogate function could be implemented as follows:
```python

def clipped_surrogate(policy, old_probs, states, actions, rewards,
                      discount = 0.995, epsilon=0.1, beta=0.01):


    discounts=discount**np.arange(len(rewards))
    Reward=np.asarray(rewards)*discounts[:,np.newaxis]
    
    Reward_future=Reward[::-1].cumsum(axis=0)[::-1]

    R_mean=Reward_future.mean(axis=1)
    R_std=Reward_future.std(axis=1)+1e-10
    
    reward_normalized=(Reward_future-R_mean[:,np.newaxis])/R_std[:,np.newaxis]
    
    
    
    actions = torch.tensor(actions, dtype=torch.int8, device=device)
    old_probs = torch.tensor(old_probs, dtype=torch.float,device=device)
    rewards=torch.tensor(reward_normalized, dtype=torch.float, device=device)


    # convert states to policy (or probability)
    new_probs = pong_utils.states_to_prob(policy, states)
    new_probs = torch.where(actions == pong_utils.RIGHT, new_probs, 1.0-new_probs)
    
    ratio=new_probs/old_probs
    clip=torch.clamp(ratio,1-epsilon,1+epsilon)

    
    # include a regularization term
    # this steers new_policy towards 0.5
    # prevents policy to become exactly 0 or 1 helps exploration
    # add in 1.e-10 to avoid log(0) which gives nan
    entropy = -(new_probs*torch.log(old_probs+1.e-10)+ \
        (1.0-new_probs)*torch.log(1.0-old_probs+1.e-10))

    return torch.mean(beta*entropy+torch.min(rewards*ratio,rewards*clip))

```