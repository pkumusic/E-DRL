# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import random
import os
import pickle

from game_state import GameState
from game_ac_network import GameACFFNetwork, GameACLSTMNetwork
from a3c_training_thread import A3CTrainingThread
from rmsprop_applier import RMSPropApplier

import options
options = options.options

def choose_action(pi_values):
  pi_values -= np.finfo(np.float32).epsneg
  action_samples = np.random.multinomial(options.num_experiments, pi_values)
  return action_samples.argmax(0)


# use CPU for display tool
device = "/cpu:0"

if options.use_lstm:
  global_network = GameACLSTMNetwork(options.action_size, -1, device)
else:
  global_network = GameACFFNetwork(options.action_size, device)

sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)

saver = tf.train.Saver()
checkpoint = tf.train.get_checkpoint_state(options.checkpoint_dir)
# for pseudo-count
psc_info = {"psc_n":0, "psc_vcount":None}
if checkpoint and checkpoint.model_checkpoint_path:
  saver.restore(sess, checkpoint.model_checkpoint_path)
  print("checkpoint loaded:", checkpoint.model_checkpoint_path)
  tokens = checkpoint.model_checkpoint_path.split("-")
  # set global step
  global_t = int(tokens[1])
  print(">>> global step set: ", global_t)
  # for pseudo-count
  if options.psc_use:
    psc_fname = options.checkpoint_dir + '/' + 'psc.' + str(global_t)
    if os.path.exists(psc_fname):
      with open(psc_fname, "rb") as f:
        psc_info = pickle.load(f)
      print("psc_info loaded:", psc_fname)
    else:
      print("psc_info does not exist and not loaded:", psc_fname)


game_state = GameState(0, options, display=options.display, no_op_max=30, thread_index=0)

# for pseudo-count
if options.psc_use:
  game_state.psc_set_psc_info(psc_info)

if options.use_gym and (options.record_screen_dir is not None):
  game_state.set_record_screen_dir(options.record_screen_dir)

for episode in range(options.num_episode_record):
  episode_record_dir = None
  if (not options.use_gym) and (options.record_screen_dir is not None):
    episode_dir = options.rom.split(".")[0] + "-e{:03d}".format(episode)
    episode_record_dir = os.path.join(options.record_screen_dir, episode_dir)
    os.makedirs(episode_record_dir)
    game_state.set_record_screen_dir(episode_record_dir)

  steps = 0
  reward = 0
  while True:
    pi_values = global_network.run_policy(sess, game_state.s_t)

    action = choose_action(pi_values)
    game_state.process(action)
    if game_state.reward != 0:
      reward += game_state.reward
      print("SCORE=", reward)

    # terminate if the play time is too long
    steps += 1
    terminal = game_state.terminal
    if steps > options.max_play_steps:
      terminal =  True

    if terminal:
      game_state.reset()
      print("Game finised with score=", reward)
      break
    else:
      game_state.update()

  if (not options.use_gym) and (options.record_screen_dir is not None):
    new_episode_record_dir = episode_record_dir + "-r{:04d}-s{:04d}".format(reward, steps)
    os.rename(episode_record_dir, new_episode_record_dir)

if options.use_gym:
  game_state.close_record_screen_dir()
