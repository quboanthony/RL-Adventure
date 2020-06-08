# Deep Reinforcement learning Learning blog 1 - introduction

## Intro

After graduated from the Udacity Deep Reinforcement Learning Nanodegree, I always feel like my pace was too fast and many things were missing. Hence I would like to summary what I have learning from the very beginning. Hoping that I could have a more structural understand about Reinforcement learning. This 'blog' will firstly serve as my personal notebook, therefore the content may seem to be crude and full of grammar mistakes (I am not a native speaker of English). I will keep polishing the contents and try to make it better and better.

The contents here mainly follow the roadmap of Udactiy course, and also include but will not be limited to the materials like ['SuttonBartoIPRLBook2ndEd'](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf), [UCL Deep Reinforcement Learning course](http://rail.eecs.berkeley.edu/deeprlcourse/), the excellent algorithm implementations like [ShangtongZhang](https://github.com/ShangtongZhang/DeepRL) and [rlcode](https://github.com/rlcode/reinforcement-learning) on Github.

Other good references are:

[Andrej Karpathy blog](karpathy.github.io/2016/05/31/rl/)

[Pinard's blog (in Chinese)](https://www.cnblogs.com/pinard/category/1254674.html)

[icml deep_rl_tutorial](https://icml.cc/2016/tutorials/deep_rl_tutorial.pdf)

## What is reinforcement learning?

Simply put, reinforcement learning is a part of machine learning, along with supervised learning and unsupervised learning.

Reinforcement Learning abstract the problem framework as an **agent** learning how to interact with an **environment**. The enviroment will feed in the agent's situation in it in the form of **state**. The environment will give agent the feedback as **reward** (real-time or deffered) as the agent takes action based on the current state, and the agent will also recieve a new **state**.

Unlike supervised learning, reinforcement learning does not have the structural labeled data for training. The reward could be seen as some sort of label but only depends on the environment and the action. Also the reward may not be associated with every state-action, the response may come later.

Unlike un-supervised Learning, reinforcement indeed have some response after all, just not the same form as in supervised learning.

The goal of the agent is to learn a **policy** (a series of actions under states) to **maximize expected cumulative reward** or the expected sum of rewards attained over all time steps. Note that we use the word **expected** here because in many cases, the rewards of all future steps cannot be determined, we want to calculate the expected value in statistical term.

Elements of RL:
1. agent & enviroment
2. policy
3. reward signal
4. value function
5. model of the enviroment

## Applications of reinforcement Learning

- Game like BackGammon, Go, Atari games, Dota2
- self-driving cars, ships, airplanes
- teach robot how to walk
- biology
- business
- telcommunications
- Finance


## RL tasks
- Episodic tasks
This type of tasks have a well-defined starting point and ending point, and an episode is the process from start to end.

- Continous tasks
This type of tasks continus forever.

## Element 1: agent and enviroment

The enviorment in an agent's observation at step $t$ is usually noted as $S_t$.

The agent's action at step $t$ is noted as $A_t$

When the agent takes an action $A_t$, its enviroment change from $S_t$ to $S_{t+1}$, and it will recieve a reward note as $R_{t+1}$.

## Element 2: policy
A policy is how the agent take action under time step $t$. It is often defined as a **conditional probability**, $\pi(a|s)=P(A_t|S_t)$.

## Element 3： reward signal
The agent recieve a reward at


## Multi-armed Bandits (active exploration)
Reinforcement learning *evaluates* the action taken rather than *instructs* by giving correct actions. The *instructive* action is the supervised learning approach, which the correct action is given as correct labels during training.

This leads to the need of active exploration. Most of the early works regards to this subject was done under a simplified setting which is called multi-armed bandits.

### K-armed Bandit Problem
Imagine you face a situation which only k choices of actions can be made, after each action you recieve a reward from a stationary probability distribution that depends on the action. The objective is to maximize the expected total rewards over some time period.

Each action has a expected or mean reward given. It can be called the *value* of that action.
$$
q_*(a)=E[R_t|A_t=a]
$$
where $A_t$ is the action taken at time $t$, $R_t$ is the reward received after the action was taken. If we know all about the values of actions, we can trivially solve the k-armed bandit problem: just select the actions with highest values.

Yet in fact we can only *estimate* the value of action at time $t$, the estimation noted as $Q_t(a)$. We can maintain this estimation during training.

When we maintain our estimation, there is always a estimation of value which is the hightest among all the others. If we take the action with the highest estimation, it is call a *greedy* approach. Which means we are **exploiting** our current knowledge of values of actions. Instead, if we choose other actions, we bring uncertainty into our estimations and we are **exploring**. When we
