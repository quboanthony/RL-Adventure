# Deep Reinforcement learning Learning blog 12 - Policy Gradient Method

This blog mainly referenced the Udacity DRLND course.

## Intro

In the previous blog, we have summarized several policy method in Reinforcement learning. These algorithms directly search  for the optimal object function value with random search in the parameter space and without knowing the gradients or minding the value function.

In this blog, we discuss another optimizing approach of policy-based methods, **the policy gradient methods**. Policy gradient methods estimate the policy function weights of optimal policy by gradient ascent.

##  The big picture
In the **black-box optimization method, such as Hill climbing**, it does not care what the objective is, or its structure. Therefore it could be
- easier to fall into the suboptimal area.
- takes forever to converge.

**The Policy Gradient method** update the parameters in the direction of increasing gradient, and iterate.
- policy:$\pi$
- Objective: $J(\theta)$
- Gradient:$\nabla_\theta J(\theta)$
- Udpdate: $\theta\leftarrow\theta+\alpha\nabla_\theta J(\theta)$

**It is possible to not find the gradient directly if the underlying policy is complicated.** In this case, you can estimate the gradient using finite differences.

### Finite differences
For $k\in[1,n]$:

Estimate:
$$
\frac{\partial J({\theta})}{\partial\theta_k}\approx \frac{J(\theta+\epsilon u_k)-J(\theta)}{\epsilon}
$$
where $\epsilon$ means a tiny step and $u_k$ means unit vector in $k^{th}$ dimension.

Collect all the partial derivatives into a single vector to get the combined derivative.
$$
\nabla_\theta J(\theta)=\langle\frac{\partial J({\theta})}{\partial\theta_1}\cdot\frac{\partial J({\theta})}{\partial\theta_2}\cdots\frac{\partial J({\theta})}{\partial\theta_m} \rangle
$$
Note that each of these policy evaluations may require at least to one full episode of interaction in the environment. And we are doing one evaluation for every parameter dimension at every iteration. This may take a long time to train (Not efficient!).

### Computing Gradients Analytically
$$
J(\theta)=E_{\pi}[R(\tau)] \\
\nabla_\theta J(\theta)=\nabla_\theta E_\pi [R(\tau)]
$$
Give the Analytically gradient of known Expected objective seems hard, but it is turns out a fairly well studied problem.

With the derivation below in the math part, we can have teh policy gradients in form of
$$
\nabla_\theta J(\theta)=\nabla_{\theta}E_{\theta}[R(\tau)]=E_{\pi}[\nabla_\theta(\log \pi(s,a,\theta))R(\tau)]
$$
where $E_{\pi}$ is the expectation
$\pi$ is the policy function which we can represent by a neural network.
$$
\nabla\theta=\alpha\nabla_\theta(\log\pi(s,a,\theta))R(\tau)
$$
### big procedure

![alt text](fig_blog_12/pg_reinforce.png "pg_reinforce")
During training, the policy gradient method will iteratively amending the weights of policy, increases the probaility of 'good' actions and decrease the probability of 'bad' actions.

LOOP:

  - Collect an  episode.

  - Change the weights of the policy network:

    - If WON, increase the  probability of each (state,action) combination.

    - If LOST, decrease the probaility of each (state,action) combination.

## Problem Setup

We consider the trajectories of a task, for example $\tau$ defined as follows:

$$
\tau=(s_0,a_0,s_1,a_1,s_2,a_2,s_3,\cdots,s_H,a_H,s_{H+1})
$$

the return  from this arbitrary trajectory $\tau$  is $R(\tau)$, defined as sum of all the rewards obtained in the trajectory,

$$
R(\tau)=r_1+r_2+r_3+\cdots+r_H+r_{H+1}
$$

Our Goal in the policy gradient methods, is to find the weights $\theta$  that maximize expected return, i.e. $\max_{\theta} U(\theta)$. By maximizing the expected return, on average, the agent experiences trajectories with high return.

The expected return $U(\theta)$ defined as follows:

$$
U(\theta)=\Sigma_{\tau}\mathrm{P}(\tau;\theta)R(\tau)
$$
where $\mathrm{P}(\tau;\theta)$ is the probaility of trajectory $\tau$,  $R(\tau)$ is the return from an arbitrary trajectory $\tau$. This calculates the expectation, i.e. the weighted average of all possible values that return $R(\tau)$ can take.

![alt text](fig_blog_12/Problem_setup_PG.png "problem_setup")

Here, we are using trajectories instead of episodes. The main reason for this is to consider both episodic and continuing tasks. That is to say, for many episodic  tasks, it makes sense to just the full episode since the rewards only deliver at the end of full episode.

## REINFORCE (Monte Carlo policy gradient)
REINFORCE is one of the algorithms  to estimate $\max_{\theta} U(\theta)$. It belongs to the **gradient ascent** method.

We learned that for policy  gradient methods, our goal is to  find the values of the weights $\theta$ in the neural network that maximize the expected  return $U$.

$$
U(\theta)=\Sigma_{\tau}\mathrm{P}(\tau;\theta)R(\tau)
$$

where $\tau$ is an arbitrary trajectory. **gradient ascent** is one way to determine the value of $\theta$ that maximizes this expected return function.

The connections between **gradient ascent** and **gradient descent** is That

- gradient descent is designed to find  the **minimum**  of a function, whereas gradient ascent will find the **maximum**.

- gradient descent  steps in the direction  of the **negative gradient**, whereas  gradient ascent steps in the direction of the **gradient**.

In each time step, we can update our parameter $\theta$ with :

$$
\theta\leftarrow\theta+\alpha\nabla_\theta U(\theta)
$$

However, by reviewing the definition  the expected return, we can see that it is impossible to calculate the gradient $\nabla_\theta U(\theta)$ directly, since we have consider every possible trajectory before calculating the gradient.

Therefore, we seek for an alternative solution.  One of them is to consider **a few trajectories** and the algorithms related is known as **REINFORCE**.

1. Use  the policy $\pi_{\theta}$ to collect $m$ trajectories $\{\tau^{(1)},\tau^{(2)},\tau^{(3)},\cdots,\tau^{(m)}\}$ with horizon $H$. We refer to the *i*-th trajectory as
$$
\tau^{(i)}=(s^{(i)}_{0},a^{(i)}_{0},\cdots,s^{(i)}_{H},a^{(i)}_{H},s^{(i)}_{H+1})
$$

2. Use the trajectories to estimate gradient $\nabla_\theta U(\theta)$:
$$
\nabla_\theta U(\theta)\approx \hat{g}:=\frac{1}{m}\Sigma^{m}_{i=1}\Sigma^{H}_{t=0}\nabla_\theta\log\pi_\theta(a_{t}^{(i)}|s_{t}^{(i)})R(\tau^{(i)})
$$
where $\hat{g}$ is the estimation of $\nabla_\theta U(\theta)$ .s

3. Update the weights of the policy:
$$
\theta\leftarrow+\alpha\hat{g}
$$

4. Loop over steps 1-3.

![alt text](fig_blog_12/reinforce.png "reinforce")

## Detrivation of the math

In this part, we would  like to go through the derivation of the estimated gradients

$$
\nabla_\theta U(\theta)\approx \hat{g}:=\frac{1}{m}\Sigma^{m}_{i=1}\Sigma^{H}_{t=0}\nabla_\theta\log\pi_\theta(a_{t}^{(i)}|s_{t}^{(i)})R(\tau^{(i)})
$$

### Likelihood  Ratio Policy Gradient
$$
\nabla_\theta U(\theta)=\nabla_\theta\Sigma_{\tau}P(\tau;\theta)R(\tau) \\
= \Sigma_{\tau}\nabla_\theta P(\tau;\theta)R(\tau)  \\
= \Sigma_{\tau}\frac{P(\tau;\theta)}{P(\tau;\theta)}\nabla_\theta P(\tau;\theta)R(\tau) \\
= \Sigma_{\tau}P(\tau;\theta)\frac{\nabla_\theta P(\tau;\theta)}{P(\tau;\theta)}R(\tau) \\
=\Sigma_{\tau}P(\tau;\theta)\nabla_\theta\log P(\tau;\theta)R(\tau)
$$

The simple trick $\nabla_\theta\log P(\tau;\theta)=\frac{\nabla_\theta P(\tau;\theta)}{P(\tau;\theta)}$ is referred to as **Likelihood ratio trick** or **REINFORCE trick**.

### Sample-Based Estimate
If  we approximate the likelihood  ratio policy gradient with a sample-based average, we have
$$
\nabla_\theta U(\theta)\approx\frac{1}{m}\Sigma^{m}_{i=1}\nabla_\theta\log\mathrm{P} (\tau^{(i)};\theta)R(\tau^{(i)})
$$
where each $\tau^{(i)}$ is  a sampled  trajectory.

### Finishing the Calculation

Here we still need to know how to calculate $\nabla_\theta\log\mathrm{P} (\tau^{(i)};\theta)$ in practice. What we have so far is a policy function $\tau_\theta$, which is estimated with the parameters $\theta$.  Hence we can furtherly finishing the calculation with finding the connection between $\nabla_\theta\log\mathrm{P} (\tau^{(i)};\theta)$ and $\tau_\theta$:

$$
\nabla_\theta\log\mathrm{P} (\tau^{(i)};\theta)=\nabla_\theta\log\left[\Pi^{H}_{t=0}\mathrm{P}(
  s^{(i)}_{t+1}|s^{(i)}_{t},a^{(i)}_{t})\pi_{\theta}(a^{(i)}_{t}|s^{(i)}_{t})\right] \\
=\nabla_\theta\left[ \Sigma^{H}_{t=0}\log\mathrm{P}(
  s^{(i)}_{t+1}|s^{(i)}_{t},a^{(i)}_{t}) + \Sigma^{H}_{t=0}\log\pi_{\theta}(a^{(i)}_{t}|s^{(i)}_{t})  \right] \\
=\nabla_\theta \Sigma^{H}_{t=0}\log\mathrm{P}(
  s^{(i)}_{t+1}|s^{(i)}_{t},a^{(i)}_{t}) + \nabla_\theta \Sigma^{H}_{t=0}\log\pi_{\theta}(a^{(i)}_{t}|s^{(i)}_{t}) \\
= \nabla_\theta \Sigma^{H}_{t=0}\log\pi_{\theta}(a^{(i)}_{t}|s^{(i)}_{t}) \\
= \Sigma^{H}_{t=0}\nabla_\theta\log\pi_{\theta}(a^{(i)}_{t}|s^{(i)}_{t})
$$

### Finally
With the results above, we have the derived the final equtions:

$$
\nabla_\theta U(\theta)\approx \hat{g}:=\frac{1}{m}\Sigma^{m}_{i=1}\Sigma^{H}_{t=0}\nabla_\theta\log\pi_\theta(a_{t}^{(i)}|s_{t}^{(i)})R(\tau^{(i)})
$$

## Code example

The policy function could be a neural network.
```python
class Policy(nn.Module):
    def __init__(self, s_size=4, h_size=16, a_size=2):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(s_size, h_size)
        self.fc2 = nn.Linear(h_size, a_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.forward(state).cpu()
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)
```
And the REINFORCE algorithms:
```python

def reinforce(n_episodes=1000, max_t=1000, gamma=1.0, print_every=100):
    scores_deque = deque(maxlen=100)
    scores = []
    for i_episode in range(1, n_episodes+1):
        saved_log_probs = []
        rewards = []
        state = env.reset()
        for t in range(max_t):
            action, log_prob = policy.act(state)
            saved_log_probs.append(log_prob)
            state, reward, done, _ = env.step(action)
            rewards.append(reward)
            if done:
                break
        scores_deque.append(sum(rewards))
        scores.append(sum(rewards))

        discounts = [gamma**i for i in range(len(rewards)+1)]
        R = sum([a*b for a,b in zip(discounts, rewards)])

        policy_loss = []
        for log_prob in saved_log_probs:
            policy_loss.append(-log_prob * R)
        policy_loss = torch.cat(policy_loss).sum()

        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        if i_episode % print_every == 0:
            print('Episode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
        if np.mean(scores_deque)>=195.0:
            print('Environment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_deque)))
            break

    return scores

```
