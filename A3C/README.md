# Git Configuration
Initialization
~~~
git clone https://github.com/pkumusic/tensorpack.git
git remote add upstream https://github.com/ppwwyyxx/tensorpack.git
~~~
Update
~~~
git fetch upstream
git checkout master
git merge upgstream/master
~~~

# Deploy on Tian's Linux Machine
1. `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64`
2. 
~~~~
export PYTHONPATH=$PYTHONPATH:`readlink -f path/to/tensorpack`
export PYTHONPATH=$PYTHONPATH:`readlink -f /usr0/home/ttian1/ml/tensorpack`
~~~~

3. Install the requirements in tensorpack

# TODO List for the Deep Learning Project (Yuezhang, Tian, Jing)
1. Choose the game we wanna use. (Having different objects) (Ranked by preference for experiments)
    * MsPacman-v0: Pacman eat beans, pellets, and may eat monsters. 
    * Riverraid-v0: Has fuel and ship objects which have opposite effects.
    * Berzerk-v0: Walls, bullets and enemies to avoid, enemies to kill.
    * Pooyan-v0: The game I love when I was a child. Some objects inside. 
    * (Hard) MontezumaRevenge-v0: A very hard game for DRL where we need to get keys and its a long adventure game. (So the reward is delayed and sparse)
    * (Hard) Kangaroo-v0: A3C seems performs bad. Has several objects.
    * (Hard) Skiing-v0: The skier needs to across the areas inside the two flags to get scores.

2. Train the A3C model for the game and evaluate it on gym.

`./train-atari.py --env MsPacman-v0 --gpu 0`

3. Visualize and evaluate the hidden features of the network to see if it encodes any object/edge info.

4. Apply pre-trained edge/object detection techniques, combining it to A3C model.

5. Can we incorporate the object/edge detection objective to the DRL model?

# To train an Atari game in gym:

`./train-atari.py --env Breakout-v0 --gpu 0`

# To run a pretrained Atari model for 100 episodes:

1. Download models from [model zoo](https://drive.google.com/open?id=0B9IPQTvr2BBkS0VhX0xmS1c5aFk)
2. `ENV=NAME_OF_ENV ./run-atari.py --load "$ENV".tfmodel --env "$ENV"`

Models are available for the following gym atari environments (click links for videos):

+ [AirRaid](https://gym.openai.com/evaluations/eval_zIeNk5MxSGOmvGEUxrZDUw) (a bit flickering, don't know why)
+ [Alien](https://gym.openai.com/evaluations/eval_8NR1IvjTQkSIT6En4xSMA)
+ [Amidar](https://gym.openai.com/evaluations/eval_HwEazbHtTYGpCialv9uPhA)
+ [Assault](https://gym.openai.com/evaluations/eval_tCiHwy5QrSdFVucSbBV6Q)
+ [Asterix](https://gym.openai.com/evaluations/eval_mees2c58QfKm5GspCjRfCA)
+ [Asteroids](https://gym.openai.com/evaluations/eval_8eHKsRL4RzuZEq9AOLZA)
+ [Atlantis](https://gym.openai.com/evaluations/eval_Z1B3d7A1QCaQk1HpO1Rg)
+ [BankHeist](https://gym.openai.com/evaluations/eval_hifoaxFTIuLlPd38BjnOw)
+ [BattleZone](https://gym.openai.com/evaluations/eval_SoLit2bR1qmFoC0AsJF6Q)
+ [BeamRider](https://gym.openai.com/evaluations/eval_KuOYumrjQjixwL0spG0iCA)
+ [Berzerk](https://gym.openai.com/evaluations/eval_Yri0XQbwRy62NzWILdn5IA)
+ [Breakout](https://gym.openai.com/evaluations/eval_L55gczPrQJamMGihq9tzA)
+ [Carnival](https://gym.openai.com/evaluations/eval_xJSOlo2lSWaH1wHEOX5vw)
+ [Centipede](https://gym.openai.com/evaluations/eval_mc1Kp5e6R42rFdjeMLzkIg)
+ [ChopperCommand](https://gym.openai.com/evaluations/eval_tYVKyh7wQieRIKgEvVaCuw)
+ [CrazyClimber](https://gym.openai.com/evaluations/eval_bKeBg0QwSgOm6A0I0wDhSw)
+ [DemonAttack](https://gym.openai.com/evaluations/eval_tt21vVaRCKYzWFcg1Kw)
+ [DoubleDunk](https://gym.openai.com/evaluations/eval_FI1GpF4TlCuf29KccTpQ)
+ [ElevatorAction](https://gym.openai.com/evaluations/eval_SqeAouMvR0icRivx2xprZg)
+ [FishingDerby](https://gym.openai.com/evaluations/eval_pPLCnFXsTVaayrIboDOs0g)
+ [Frostbite](https://gym.openai.com/evaluations/eval_qtC3taKFSgWwkO9q9IM4hA)
+ [Gopher](https://gym.openai.com/evaluations/eval_KVcpR1YgQkEzrL2VIcAQ)
+ [Gravitar](https://gym.openai.com/evaluations/eval_QudrLdVmTpK9HF5juaZr0w)
+ [IceHockey](https://gym.openai.com/evaluations/eval_8oWCTwwGS7OUTTGRwBPQkQ)
+ [Jamesbond](https://gym.openai.com/evaluations/eval_mLF7XPi8Tw66pnjP73JsmA)
+ [JourneyEscape](https://gym.openai.com/evaluations/eval_S9nQuXLRSu7S5x21Ay6AA)
+ [Kangaroo](https://gym.openai.com/evaluations/eval_TNJiLB8fTqOPfvINnPXoQ)
+ [Krull](https://gym.openai.com/evaluations/eval_dfOS2WzhTh6sn1FuPS9HA)
+ [KungFuMaster](https://gym.openai.com/evaluations/eval_vNWDShYTRC0MhfIybeUYg)
+ [MsPacman](https://gym.openai.com/evaluations/eval_kpL9bSsS4GXsYb9HuEfew)
+ [NameThisGame](https://gym.openai.com/evaluations/eval_LZqfv706SdOMtR4ZZIwIsg)
+ [Phoenix](https://gym.openai.com/evaluations/eval_uzUruiB3RRKUMvJIxvEzYA)
+ [Pong](https://gym.openai.com/evaluations/eval_8L7SV59nSW6GGbbP3N4G6w)
+ [Pooyan](https://gym.openai.com/evaluations/eval_UXFVI34MSAuNTtjZcK8N0A)
+ [Qbert](https://gym.openai.com/evaluations/eval_wekCJkrWQm9NrOUzltXg)
+ [Riverraid](https://gym.openai.com/evaluations/eval_OU4x3DkTfm4uaXy6CIaXg)
+ [RoadRunner](https://gym.openai.com/evaluations/eval_wINKQTwxT9ipydHOXBhg)
+ [Robotank](https://gym.openai.com/evaluations/eval_Gr5c0ld3QACLDPQrGdzbiw)
+ [Seaquest](https://gym.openai.com/evaluations/eval_N2624y3NSJWrOgoMSpOi4w)
+ [SpaceInvaders](https://gym.openai.com/evaluations/eval_Eduozx4HRyqgTCVk9ltw)
+ [StarGunner](https://gym.openai.com/evaluations/eval_JB5cOJXFSS2cTQ7dXK8Iag)
+ [Tennis](https://gym.openai.com/evaluations/eval_gDjJD0MMS1yLm1T0hdqI4g)
+ [Tutankham](https://gym.openai.com/evaluations/eval_gDjJD0MMS1yLm1T0hdqI4g)
+ [UpNDown](https://gym.openai.com/evaluations/eval_KmkvMJkxQFSED20wFUMdIA)
+ [VideoPinball](https://gym.openai.com/evaluations/eval_PWwzNhVFR2CxjYvEsPfT1g)
+ [WizardOfWor](https://gym.openai.com/evaluations/eval_1oGQhphpQhmzEMIYRrrp0A)
+ [Zaxxon](https://gym.openai.com/evaluations/eval_TIQ102EwTrHrOyve2RGfg)

Note that atari game settings in gym are quite different from DeepMind papers, so the scores are not comparable. The most notable differences are:
+ In gym, each action is randomly repeated 2~4 times.
+ In gym, inputs are RGB instead of greyscale.
+ In gym, an episode is limited to 10000 steps.
+ The action space also seems to be different.

Also see the DQN implementation [here](../Atari2600)
