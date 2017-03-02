This is the A3C method improved with pseudo count exploration reward. By very simple density model, the agent is possible to get 500 score at about 20 epochs (180k updates)

Some parameters to tune with, here is the default settings:
* Exploration rate from ((0,1), (80, 2), (100, 3), (120, 4), (140, 5))
* Pseudo count methods. 'joint' means each pixel is seemed as independent. 'CTS' means CTS based methods. I'll report training speed later.
* Frame Size. Downsampling size for density model. 42 as default
* MAX_DOWNSAMPLED_VAL. rescaled value for density model. default 128.
* beta * ((count+alpha) ** power). beta=0.05, alpha=0.01, power=-0.5. This results in 0.5 score for any image with 0 count.
* The rewards are clipped between (-1,1)
* Network Architecture: default is same as in Nature DQN. 1 is the CNN with max pooling. 

Training speed:
* baseline A3C: 9.6 iter/s 
* joint, 42, 128: 9.8 iter/s (Almost no overhead)

Example of running script
~~~
python A3C-gym-train.py --gpu 1 --pc joint --env MontezumaRevenge-v0 --logdir /serverdata1/music_sandbox/E-DRL/montezuma/A3C-PC-joint
~~~


A3C-PC-joint-128-500score-downsample42-value128

