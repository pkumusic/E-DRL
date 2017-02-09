# -*- coding: utf-8 -*-
import tensorflow as tf
import threading

import signal
import math
import os
import time
import pickle

from game_ac_network import GameACFFNetwork, GameACLSTMNetwork
from a3c_training_thread import A3CTrainingThread
from rmsprop_applier import RMSPropApplier

import options
options = options.options

def log_uniform(lo, hi, rate):
  log_lo = math.log(lo)
  log_hi = math.log(hi)
  v = log_lo * (1-rate) + log_hi * rate
  return math.exp(v)

device = "/cpu:0"
if options.use_gpu:
  device = "/gpu:0"

initial_learning_rate = log_uniform(options.initial_alpha_low,
                                    options.initial_alpha_high,
                                    options.initial_alpha_log_rate)

global_t = 0

stop_requested = False

# for thread syncroization
th0_ready = threading.Event()
all_ready = threading.Event()
th0_finish = threading.Event()

th0_ready.clear()
all_ready.clear()
th0_finish.clear()
num_ready = 0


if options.use_lstm:
  global_network = GameACLSTMNetwork(options.action_size, -1, device)
else:
  global_network = GameACFFNetwork(options.action_size, device)


training_threads = []

learning_rate_input = tf.placeholder("float")

grad_applier = RMSPropApplier(learning_rate = learning_rate_input,
                              decay = options.rmsp_alpha,
                              momentum = 0.0,
                              epsilon = options.rmsp_epsilon,
                              clip_norm = options.grad_norm_clip,
                              device = device)

for i in range(options.parallel_size):
  training_thread = A3CTrainingThread(i, global_network, initial_learning_rate,
                                      learning_rate_input,
                                      grad_applier, options.max_time_step,
                                      device = device, options = options)
  training_threads.append(training_thread)

# prepare session
sess = tf.Session(config=tf.ConfigProto(log_device_placement=False,
                                        allow_soft_placement=True))

init = tf.initialize_all_variables()
sess.run(init)

# summary for tensorboard
score_input = tf.placeholder(tf.int32)
tf.scalar_summary("score", score_input)

summary_op = tf.merge_all_summaries()
summary_writer = tf.train.SummaryWriter(options.log_file, sess.graph_def)

# init or load checkpoint with saver
saver = tf.train.Saver(max_to_keep = options.max_to_keep)
checkpoint = tf.train.get_checkpoint_state(options.checkpoint_dir)
# for pseudo-count
psc_info = None
all_gs_info = [None for i in range(options.parallel_size)]
if checkpoint and checkpoint.model_checkpoint_path:
  saver.restore(sess, checkpoint.model_checkpoint_path)
  print("checkpoint loaded:", checkpoint.model_checkpoint_path)
  tokens = checkpoint.model_checkpoint_path.split("-")
  # set global step
  global_t = int(tokens[1])
  print(">>> global step set: ", global_t)
  # set wall time
  wall_t_fname = options.checkpoint_dir + '/' + 'wall_t.' + str(global_t)
  with open(wall_t_fname, 'r') as f:
    wall_t = float(f.read())
  # for pseudo-count
  if options.psc_use:
    # psc_info of thread0 (for compatibility)
    psc_fname = options.checkpoint_dir + '/' + 'psc.' + str(global_t)
    if os.path.exists(psc_fname):
      with open(psc_fname, "rb") as f:
        psc_info = pickle.load(f)
      print("psc_info loaded:", psc_fname)
    else:
      print("psc_info does not exist and not loaded:", psc_fname)
    # gs_info of all thread
    gs_fname = options.checkpoint_dir + '/' + 'gs.' + str(global_t)
    if os.path.exists(gs_fname):
      with open(gs_fname, "rb") as f:
        all_gs_info = pickle.load(f)
      print("all_gs_info loaded:", gs_fname)
    else:
      print("all_gs_info does not exist and not loaded:", gs_fname)

  next_save_steps = (global_t + options.save_time_interval)//options.save_time_interval * options.save_time_interval
else:
  print("Could not find old checkpoint")
  # set wall time
  wall_t = 0.0
  next_save_steps = options.save_time_interval


def save_data(training_threads):
  if not os.path.exists(options.checkpoint_dir):
    os.mkdir(options.checkpoint_dir)  

  # need copy of global_t because it might be changed in other thread
  global_t_copy = global_t

  # write wall time
  wall_t = time.time() - start_time
  wall_t_fname = options.checkpoint_dir + '/' + 'wall_t.' + str(global_t_copy)
  with open(wall_t_fname, 'w') as f:
    f.write(str(wall_t))

  # write psc_info
  if options.psc_use:
    # write psc_info of thread0 (for compatibility)
    game_state = training_threads[0].game_state
    psc_n = game_state.psc_n
    psc_vcount = game_state.psc_vcount
    psc_fname = options.checkpoint_dir + '/' + 'psc.' + str(global_t_copy)
    with open(psc_fname, "wb") as f:
      pickle.dump({"psc_n":psc_n, "psc_vcount":psc_vcount}, f)
    # write game_state info of all thread (all_gs_info)
    all_gs_info = []
    for i in range(options.parallel_size):
      game_state = training_threads[i].game_state
      psc_n = game_state.psc_n
      psc_vcount = game_state.psc_vcount
      rooms = game_state.rooms
      episode = game_state.episode
      gs_info = {"psc_n":psc_n, "psc_vcount":psc_vcount, "rooms":rooms, "episode":episode}
      all_gs_info.append(gs_info)
    gs_fname = options.checkpoint_dir + '/' + 'gs.' + str(global_t_copy)
    with open(gs_fname, "wb") as f:
      pickle.dump(all_gs_info, f)

  saver.save(sess, options.checkpoint_dir + '/' + 'checkpoint', global_step = global_t_copy)

  print('@@@ Data saved at global_t={}'.format(global_t_copy))

#@profile
def train_function(parallel_index):
  global global_t
  global next_save_steps
  global num_ready
  
  training_thread = training_threads[parallel_index]
  # set start_time
  start_time = time.time() - wall_t
  training_thread.set_start_time(start_time)

  # for pseudo-count
  if options.psc_use:
    training_thread.game_state.psc_set_psc_info(psc_info)
    gs_info = all_gs_info[parallel_index]
    if gs_info is not None:
      training_thread.game_state.psc_set_gs_info(gs_info) 

  best_average_score = 0
  while True:
    if global_t > next_save_steps or \
      global_t > options.end_time_step or \
      stop_requested:

      if parallel_index == 0:
        if options.sync_thread:
          all_ready.clear()
          th0_finish.clear()
          num_ready = 1
          th0_ready.set()
          all_ready.wait()
          th0_ready.clear()
          
        next_save_steps += options.save_time_interval
 
        if global_t > options.end_time_step or \
          stop_requested:
          save_data(training_threads)
        elif options.save_best_avg_only:
          average_score = training_thread.episode_scores.average()
          print("%%% best_average_score={:.5f}, average_score={:.5f}".format(best_average_score, average_score))
          if average_score > best_average_score:
            best_average_score = average_score
            print("%%% NEW best_average_score={:.5f}".format(best_average_score))
            save_data(training_threads)
          else:
            print("%%% no update of best_average_score")
        else:
          save_data(training_threads)
        
        if options.sync_thread:
          th0_finish.set()

      else:
        if options.sync_thread:
          th0_ready.wait()
          num_ready += 1
          if num_ready == options.parallel_size:
            all_ready.set()
          th0_finish.wait()

      if global_t > options.end_time_step or \
        stop_requested:
        break

    diff_global_t, _ = training_thread.process(sess, global_t, summary_writer,
                                               summary_op, score_input)
    global_t += diff_global_t
     

def gym_eval_function(parallel_index):
  global global_t
  global next_save_steps
  
  training_thread = training_threads[parallel_index]
  # set start_time
  start_time = time.time()
  training_thread.set_start_time(start_time)

  # for pseudo-count
  if options.psc_use:
    training_thread.game_state.psc_set_psc_info(psc_info)
    gs_info = all_gs_info[parallel_index]
    if gs_info is not None:
      training_thread.game_state.psc_set_gs_info(gs_info) 

  env = training_thread.game_state.gym
  env.monitor.start(options.record_screen_dir)
  env.reset()
  spec = env.spec

  trials = spec.trials
  trials_in_thread = trials // options.parallel_size
  if parallel_index < trials % options.parallel_size:
    trials_in_thread += 1

  timestep_limit = spec.timestep_limit
  options.max_play_steps = timestep_limit // options.frames_skip_in_gs
  
  for _ in range(trials_in_thread):
    while True:
      if stop_requested:
        break

      diff_global_t, terminal_end = training_thread.process(sess, global_t, summary_writer,
                                                            summary_op, score_input)
      global_t += diff_global_t
      if terminal_end:
        break

  env.monitor.close()
 
    
def signal_handler(signal, frame):
  global stop_requested
  print('You pressed Ctrl+C!')
  stop_requested = True
  
if options.gym_eval:
  eval_threads = []
  for i in range(options.parallel_size):
    eval_threads.append(threading.Thread(target=gym_eval_function, args=(i,)))

  global_t = 0

  for t in eval_threads:
    t.start()

else:
  train_threads = []
  for i in range(options.parallel_size):
    train_threads.append(threading.Thread(target=train_function, args=(i,)))
    
  signal.signal(signal.SIGINT, signal_handler)

  # set start time
  start_time = time.time() - wall_t

  for t in train_threads:
    t.start()

  print('Press Ctrl+C to stop')

  for t in train_threads:
    t.join()
