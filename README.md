# E-DRL
Exploration Strategies for Deep Reinforcement Learning

I plan to explore the exploration strategies for DRL to overcome sparse reward games such as MONTEZUMA's REVENGE and PITFALL.
PITFALL is actually unsolved and all current models resulting in pessimistic behaviors (e.g., jumping around the starting point)

I'll first run baselines of Dueling DQN and A3C on Montezuma's Revenge and PITFALL. Here are the baselines:

I'll implemented counter-based approach on both tasks.

I'll try my approach on that.

Some ideas:

1. Do we really need time discount on reward since the reward are generally all long-term?

2. Since it gets a lot of negative rewards before it is possible to get to a positive reward, current models tend to punish these policies: which means it would not go any direction, and just stay at the starting point. How to overcome this? Is it good to clip or reduce the negative reward? Or how to improve the possibility of coming to a positive reward point? Or how to strengthen the experience when getting a positive reward?
