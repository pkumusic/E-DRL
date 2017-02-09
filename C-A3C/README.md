# async_deep_reinforce

Asynchronous deep reinforcement learning + Pseudo-count based reward + On-highscore-learning

## About

This code is fork from [miyosuda's code](https://github.com/miyosuda/async_deep_reinforce). I added many functions for my Deep Learning experiments. Of which, pseudo-count based reward based on following DeepMind's paper and on-highscore-learning (my original) enable over 1500 point average score in Montezuma's Revenge, which is higher than the paper as for A3C.
 
[https://arxiv.org/abs/1606.01868 (Unifying Count-Based Exploration and Intrinsic Motivation, DeepMind)](https://arxiv.org/abs/1606.01868)

"on-highscore-learning" is my original idea, which learn from state-action-rewards-history when getting highscore. But in evaluation of Montezuma's Revenge, I set option to reset highscore in every episode, so learning occured in every score. (I'm changing this now. In new version, only highscore episode will be selected automatically based on history of scores) 

## Slide

See following slide (in English) for explanation of this project.

[http://www.slideshare.net/ItsukaraIitsuka/drl-challenge-on-montezumas-revenge](http://www.slideshare.net/ItsukaraIitsuka/drl-challenge-on-montezumas-revenge
)

See following slide (in Japanese) for Japanese explanation.

[http://www.slideshare.net/ItsukaraIitsuka/deepmind20166-unifying-countbased-exploration-and-intrinsic-motivation-pseudocount-montezumas-revenge](http://www.slideshare.net/ItsukaraIitsuka/deepmind20166-unifying-countbased-exploration-and-intrinsic-motivation-pseudocount-montezumas-reveng0e)

## Learning curve of Montezuma's Revenge

The following graph is the average score of Montezuma's Revenge.

![learning result after 39M steps](https://cdn-ak.f.st-hatena.com/images/fotolife/I/Itsukara/20160902/20160902172341.png)

0 - 30M steps: Pseudo-count based reward is ON.

30 - 40M steps: Above + on-highscore-learning is ON.

## Best Learning curve of Monezuma's Revenge

The following graph is the best Learning Curve of Montezuma's Revenge (2016/10/7). Best score is 2500 and peak average score is more than 1500 point. 
 
![best learning result](https://cdn-ak.f.st-hatena.com/images/fotolife/I/Itsukara/20161007/20161007134815.png)


## Explored Rooms

- My result
The following picture indicates the rooms explored in my all trainings. 

![explored rooms in my trainings](https://cdn-ak.f.st-hatena.com/images/fotolife/I/Itsukara/20161205/20161205234507.png)

This is better than DeepMind result (see next picture). This was achieved in OpenAI Gym environment only. In ALE environment, although the average score is higher than OpenAI Gym, the number of explored rooms is less than that of OpenAI Gym.

- DeepMind's result
The rooms expolred in DeepMind paper (all in all).

![explored rooms in DeepMind's trainings](https://cdn-ak.f.st-hatena.com/images/fotolife/I/Itsukara/20161225/20161225040856.png)



## Play movie

The following is a play movie of Montezuma's Revenge after training 50M steps. Its score is 2600.

[![](http://img.youtube.com/vi/tts3wOPnKQE/0.jpg)](https://www.youtube.com/watch?v=tts3wOPnKQE)

## How to prepare environment

This code needs Anaconda, tensorflow, opencv3 and Arcade Learning Environment (ALE). After download of gcp-install-a3c-env.tgz, you can use scrips in "gcp-install" directory. Run following.
 
    $ sudo apt-get install git
    $ git clone https://github.com/Itsukara/async_deep_reinforce.git
    $ mkdir Download
    $ cp async_deep_reinforce/gcp-install/gcp-install-a3c-env.tgz Download/
    $ cd Download/
    $ tar zxvf gcp-install-a3c-env.tgz
    $ bash -x install-Anaconda.sh
    $ . ~/.bashrc
    $ bash -x install-tensorflow.sh
    $ bash -x install-opencv3.sh
    $ bash -x install-ALE.sh
    $ bash -x install-additionals.sh
    $ cd ../async_deep_reinforce
    $ ./run-option montezuma-c-avg-greedy-rar025

When program requests input, just hit Enter or input "y" or "yes" and hit Enter. But as for Anaconda, you have to input "q" when License and "--More--" is displayed.

I built the environment using my scripts on Ubuntu 14.04LTS 64bit in Google Cloud Platform, Amazon EC2 and Microsoft Azure. 

## How to train

To train,

    $ ./run-option montezuma-c-max-greedy-rar025

To display game screen played by the program,

    $ python a3c_display.py --rom=montezuma_revenge.bin --display=True

To create play movie without displaying the game screen,

    $ python a3c_display.py --rom=montezuma_revenge.bin --record-screen-dir=screen
    $ run-avconv-all screen # you need avconv

## Run options

As for options, see options.py.

## How to reproduce OpenAI Gym Result

I uploaded evaluation result in OpenAI Gym. See ["OpenAI Gym evaluation page"](https://gym.openai.com/evaluations/eval_mMQJe1nsS7OA3U72AvvJEQ). I'd appreciate if you cloud review my evaluation.

To repuroduce OpenAP Gym result,

   $ ./run-option-gym montezuma-j-tes30-b0020-ff-fs2

Play screens are recorded in following directory,

- screen.new-room : screens when entered new room are recored
- screen.new-record : screens when achieved new score are recorded

## Status of code

The source code is still under development and may chage frequently. Currently, I'm searching best parameters to speed-up learning and get higher score. In this search, I'm adding new functions to change behavior of the program. So, it might be degraded sometimes. Sorry for that in advance.

## Sharing experiment result

I'd appreciate if you could write your experiment result to thread ["Experiment Results"](https://github.com/Itsukara/async_deep_reinforce/issues/3) in Issues. 

## Blog

I'm writing blog on this program. See following (in Japanese):

[http://itsukara.hateblo.jp/ (Itsukara's Blog)](http://itsukara.hateblo.jp/)

## How to refer

I'd appreciate if you woud refer my code in your blog or paper as following:

https://github.com/Itsukara/async_deep_reinforce (On-Highscore-Learning code based on A3C+ and Pseudo-count developed by Itsukara) 

## Acknowledgements

- [@miosuda](https://github.com/miyosuda/async_deep_reinforce) for providing very fast A3C program.

