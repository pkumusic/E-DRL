Implementing DQN, DDQN, Dueling DQN and DDDQN with tensorpack.

Implementing Objecti-Sensitive DQNs (O-DQNs).

# Different methods to incorporate objects
## Combine/Separate
+ If we combine or separate the extracted objects to the same or separate images. TODO: Add examples. 
## Add/Swap
+ Whether we add the object features to the original images or completely swap. 
## Input/Feature
+ Where we input our feature. 

# For turtle machines
About 15 iters/s when using an empty GPU card. 

# From tensorpack
DQN typically took 2 days of training to reach a score of 400 on breakout game.
My Batch-A3C implementation only took <2 hours.
Both were trained on one GPU with an extra GPU for simulation.

The x-axis is the number of iterations, not wall time.
Iteration speed on Tesla M40 is about 9.7it/s for B-A3C.
D-DQN is faster at the beginning but will converge to 12it/s due of exploration annealing.

# How to use

To train DQNs:
```
python DQN-gym-train.py --env MsPacman-v0 --logdir test --gpu 0 --double t --dueling f
```

To visualize and submit to gyn:
```
python DQN-gym-run.py --env MsPacman-v0 --load <model-file> --gpu 0 --double t --dueling f --output folder --api <your-gym-api>
```
