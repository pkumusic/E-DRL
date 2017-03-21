# E-DRL
Exploration Strategies for Deep Reinforcement Learning
## Weely Discussion
1. Experiment Count-based Exploration on pitfall and montezuma's Revenge. [Link](https://github.com/Itsukara/async_deep_reinforce) 乐章
2. Improve density models for count-based exploration (PixelCNN, GANs, VAEs, etc) 尚航
3. Other exploration strategies [paper](http://www.cs.mcgill.ca/~cs526/roger.pdf) (e.g., Competence Maps, Multiple-Hypothesis Heuristics) 江帆
4. Baselines and possible improvements [Starcraft](https://arxiv.org/abs/1609.02993),[Gradient Boosting](https://arxiv.org/abs/1603.04119),[Bootstrapped DQN](https://arxiv.org/abs/1602.04621),[VIME](https://arxiv.org/abs/1605.09674),[Incentivizing](https://arxiv.org/abs/1507.00814),[BBQ Nets](https://arxiv.org/abs/1608.05081), [Reward Design](https://arxiv.org/abs/1604.07095), [Under-appreciated Rewards](https://arxiv.org/abs/1611.09321), [MBIE-EB](http://www.sciencedirect.com/science/article/pii/S0022000008000767)  弘扬，李军
5. Novel ideas to incorporate exploration into network structure

## Highlights:
* Achieved 2500 score (ranked 1) on OpenAI gym Montezuma's Revenge game [Demo](https://gym.openai.com/evaluations/eval_zxZ4J4nTRw2snOY7umcfqw)


## Plans

Some ideas:
1. Combining better density models for count-based approach. (e.g., PixelCNN)
2. Using auxiliary tasks for exploration.
3. Using Thompson sampling 
4. Design network for exploration

We plan to explore the exploration strategies for DRL to overcome sparse reward games such as MONTEZUMA's REVENGE and PITFALL.
PITFALL is actually unsolved and all current models resulting in pessimistic behaviors (e.g., jumping around the starting point)

We'll first run baselines of Dueling DQN and A3C on Montezuma's Revenge and PITFALL. Here are the baselines:

We'll implemented counter-based approach on both tasks.

We'll try my approach on that.

Some ideas:

1. Do we really need time discount on reward since the reward are generally all long-term?

2. Since it gets a lot of negative rewards before it is possible to get to a positive reward, current models tend to punish these policies: which means it would not go any direction, and just stay at the starting point. How to overcome this? Is it good to clip or reduce the negative reward? Or how to improve the possibility of coming to a positive reward point? Or how to strengthen the experience when getting a positive reward?

3. Based on count-based methods, what we actually need is a density estimator for an unseen image. Could we use Variational Autoencoder, Generative Adversarial Nets, Bolzman Machines or any other approaches to build the estimator?

4. Is it possible to directly embed the exploration strategy into DQN training?

## Installation notes:

* Python version: 2.7 
* Setting up virtual machine:
 ~~~
virtualenv drl
source drl/bin/activate
 ~~~
* Tensorflow version is 0.11.0rc0, https://www.tensorflow.org/versions/r0.11/get_started/os_setup
~~~
export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.11.0rc0-cp27-none-linux_x86_64.whl
pip install --upgrade $TF_BINARY_URL
~~~
* Export Tensorpack. A history modified version of tensorpack contained in the repo.:
~~~
export PYTHONPATH=$PYTHONPATH:path/to/E-DRL
~~~
* Install cv2:
~~~
pip install opencv-python (This is just a binding. May not work. May need to install from source)
~~~
* Install other dependencies:
~~~
pip install -r requirements.txt
pip install gym
pip install gym[atari]
~~~
* It should work:
~~~
python DQN-gym-train.py --env SpaceInvaders-v0 -g 0 --double f --dueling f --logdir DQN
~~~