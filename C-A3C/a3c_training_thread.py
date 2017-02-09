# -*- coding: utf-8 -*-
import numpy as np
import random
import time
import cv2
import os
import lzma
from collections import deque
from sortedcontainers import SortedList

from accum_trainer import AccumTrainer
from game_state import GameState
from game_ac_network import GameACFFNetwork, GameACLSTMNetwork

import options
options = options.options

class Episode_scores(object):
  def __init__(self, options):
    self.maxlen = options.score_averaging_length
    self.threshold = options.score_highest_ratio
    self.episode_scores = deque()
    self.episode_scores.append(0) # to avoid 0-div in first averaging
    self.episode_scores_sum = 0
    self.sorted_scores = SortedList()
    self.sorted_scores.add(0) # align to episode_scores
    self.num_episode = 0
    self.options = options

  def add(self, n, global_t, thread_index):
    self.episode_scores_sum += n
    self.episode_scores.append(n)
    self.sorted_scores.add(-n) # trick to use SortedList in reverse order
    if len(self.episode_scores) > self.maxlen:
      oldest = self.episode_scores.popleft()
      self.sorted_scores.remove(-oldest)
      self.episode_scores_sum -= oldest
    self.num_episode += 1
    if self.num_episode % self.options.average_score_log_interval == 0:
      print("@@@ Average Episode score = {:.6f}, s={:9d},th={}".format(self.average(), global_t, thread_index))

  def average(self):
    return self.episode_scores_sum / len(self.episode_scores)

  def is_highscore(self, n):
    sorted_scores = self.sorted_scores
    num_scores = len(sorted_scores)
    sorted_scores.add(-n)
    index = sorted_scores.index(-n)
    highest_ratio = (index + 1) / num_scores
    sorted_scores.remove(-n)
    return highest_ratio <= self.threshold


class A3CTrainingThread(object):
  def __init__(self,
               thread_index,
               global_network,
               initial_learning_rate,
               learning_rate_input,
               grad_applier,
               max_global_time_step,
               device,
               options):

    self.thread_index = thread_index
    self.learning_rate_input = learning_rate_input
    self.max_global_time_step = max_global_time_step
    self.options = options

    if options.use_lstm:
      self.local_network = GameACLSTMNetwork(options.action_size, thread_index, device)
    else:
      self.local_network = GameACFFNetwork(options.action_size, device)

    self.local_network.prepare_loss(options.entropy_beta)

    # TODO: don't need accum trainer anymore with batch
    self.trainer = AccumTrainer(device)
    self.trainer.prepare_minimize( self.local_network.total_loss,
                                   self.local_network.get_vars() )
    
    self.accum_gradients = self.trainer.accumulate_gradients()
    self.reset_gradients = self.trainer.reset_gradients()
  
    self.apply_gradients = grad_applier.apply_gradients(
      global_network.get_vars(),
      self.trainer.get_accum_grad_list() )

    self.sync = self.local_network.sync_from(global_network)
    
    self.game_state = GameState(random.randint(0, 2**16), options, thread_index = thread_index)
    
    self.local_t = 0

    self.initial_learning_rate = initial_learning_rate

    self.episode_reward = 0

    self.indent = "         |" * self.thread_index
    self.steps = 0
    self.no_reward_steps = 0
    self.terminate_on_lives_lost = options.terminate_on_lives_lost and (self.thread_index != 0)

    if self.options.train_episode_steps > 0:
      self.max_reward = 0.0
      self.max_episode_reward = 0.0
      self.episode_states = []
      self.episode_actions = []
      self.episode_rewards = []
      self.episode_values = []
      self.episode_liveses = []
      self.episode_scores = Episode_scores(options)
      self.tes = self.options.train_episode_steps
      if self.options.tes_list is not None:
        self.tes = self.options.tes_list[thread_index]
        print("[DIVERSITY]th={}:tes={}".format(thread_index, self.tes))
    self.initial_lives = self.game_state.initial_lives
    self.max_history = int(self.tes * self.options.tes_extend_ratio * 2.1)

    if self.options.record_new_record_dir is not None:
      if self.thread_index == 0:
        if not os.path.exists(self.options.record_new_record_dir):
          os.makedirs(self.options.record_new_record_dir)
      self.episode_screens = []

    if self.options.record_new_room_dir is not None:
      if self.thread_index == 0:
        if not os.path.exists(self.options.record_new_room_dir):
          os.makedirs(self.options.record_new_room_dir)
      self.episode_screens = []

    self.greediness = options.greediness
    self.repeat_action_ratio = options.repeat_action_ratio
    self.prev_action = 0

    
    

  def _anneal_learning_rate(self, global_time_step):
    learning_rate = self.initial_learning_rate * (self.max_global_time_step - global_time_step) / self.max_global_time_step
    if learning_rate < 0.0:
      learning_rate = 0.0
    return learning_rate

  def choose_action(self, pi_values, global_t):
    # Add greediness for broader exploration
    r = random.random()
    if r < self.greediness:
      action =  int(r * len(pi_values))
    elif r < self.repeat_action_ratio:
      action = self.prev_action
    else:
      # Increase randomness of choice if no reward term is too long
      if self.no_reward_steps > self.options.no_reward_steps:
        randomness = (self.no_reward_steps - self.options.no_reward_steps) * self.options.randomness
        pi_values += randomness
        pi_values /= sum(pi_values)
        if self.local_t % self.options.randomness_log_interval == 0:
          elapsed_time = time.time() - self.start_time
          print("t={:6.0f},s={:9d},th={}:{}randomness={:.8f}".format(
                elapsed_time, global_t, self.thread_index, self.indent, randomness))

      pi_values -= np.finfo(np.float32).epsneg
      action_samples = np.random.multinomial(self.options.num_experiments, pi_values)
      action = action_samples.argmax(0)

    self.prev_action = action
    return action

  def _record_score(self, sess, summary_writer, summary_op, score_input, score, global_t):
    summary_str = sess.run(summary_op, feed_dict={
      score_input: score
    })
    summary_writer.add_summary(summary_str, global_t)
    
  def set_start_time(self, start_time):
    self.start_time = start_time

  #@profile
  def process(self, sess, global_t, summary_writer, summary_op, score_input):
    states = []
    actions = []
    rewards = []
    values = []
    liveses = [self.game_state.lives]
    if self.tes > 0:
      if self.episode_liveses == []:
        self.episode_liveses.append(self.game_state.lives)

    terminal_end = False

    # reset accumulated gradients
    sess.run( self.reset_gradients )

    # copy weights from shared to local
    sess.run( self.sync )

    start_local_t = self.local_t

    if self.options.use_lstm:
      start_lstm_state = self.local_network.lstm_state_out
    
    # t_max times loop
    for i in range(self.options.local_t_max):
      pi_, value_ = self.local_network.run_policy_and_value(sess, self.game_state.s_t)
      action = self.choose_action(pi_, global_t)

      states.append(self.game_state.s_t)
      actions.append(action)
      values.append(value_)
      liveses.append(self.game_state.lives)

      if (self.thread_index == 0) and (self.local_t % self.options.log_interval == 0):
        print("pi={} (thread{})".format(pi_, self.thread_index))
        print(" V={} (thread{})".format(value_, self.thread_index))

      # process game
      self.game_state.process(action)

      # receive game result
      reward = self.game_state.reward
      terminal = self.game_state.terminal

      self.episode_reward += reward
      if reward > 0 and \
         (self.options.rom == "montezuma_revenge.bin" or self.options.gym_env == "MontezumaRevenge-v0"):
        elapsed_time = time.time() - self.start_time
        print("t={:6.0f},s={:4.0f},th={}:{}r={:3.0f}RM{:02d}| NEW-SCORE".format(
              elapsed_time, global_t, self.thread_index, self.indent, self.episode_reward,
              self.game_state.room_no))

      # pseudo-count reward
      if self.options.psc_use:
        reward += self.game_state.psc_reward

      # add basic income after some no reward steps
      if self.no_reward_steps > self.options.no_reward_steps:
        reward += self.options.basic_income

      # clip reward
      if self.options.reward_clip > 0.0:
        reward = np.clip(reward, -self.options.reward_clip, self.options.reward_clip)
      rewards.append( reward )

      # collect episode log
      if self.tes > 0:
        self.episode_states.append(self.game_state.s_t)
        self.episode_actions.append(action)
        self.episode_rewards.append(reward)
        self.episode_values.append(value_)
        self.episode_liveses.append(self.game_state.lives)
        if len(self.episode_states) > self.max_history * 2:
          self.episode_states = self.episode_states[-self.max_history:]
          self.episode_actions = self.episode_actions[-self.max_history:]
          self.episode_rewards = self.episode_rewards[-self.max_history:]
          self.episode_values = self.episode_values[-self.max_history:]
          self.episode_liveses = self.episode_liveses[-self.max_history-1:]
        # requirement for OpenAI Gym: --clear-history-on-death=False
        if self.options.clear_history_on_death and (liveses[-2] > liveses[-1]):
          self.episode_states = []
          self.episode_actions = []
          self.episode_rewards = []
          self.episode_values = []
          self.episode_liveses = self.episode_liveses[-2:]
 
      self.local_t += 1

      if self.options.record_new_record_dir is not None \
         or self.options.record_new_room_dir is not None:
        screen = self.game_state.uncropped_screen
        if self.options.compress_frame:
          screen = lzma.compress(screen.tobytes(), preset=0)
        self.episode_screens.append(screen)

      # terminate if the play time is too long
      self.steps += 1
      if self.steps > self.options.max_play_steps:
        terminal = True

      # requirement for OpenAI Gym: --terminate-on-lives-lost=False
      # terminate if lives lost
      if self.terminate_on_lives_lost and (liveses[-2] > liveses[-1]):
        terminal = True

      # count no reward steps
      if self.game_state.reward == 0.0:
        self.no_reward_steps += 1
      else:
        self.no_reward_steps = 0

      # s_t1 -> s_t
      self.game_state.update()
      
      if self.local_t % self.options.score_log_interval == 0:
        elapsed_time = time.time() - self.start_time
        print("t={:6.0f},s={:9d},th={}:{}r={:3.0f}RM{:02d}| l={:.0f},v={:.5f},pr={:.5f}".format(
              elapsed_time, global_t, self.thread_index, self.indent,
              self.episode_reward, self.game_state.room_no,
              self.game_state.lives, value_, self.game_state.psc_reward))

      # if self.game_state.room_no != self.game_state.prev_room_no:
      #   elapsed_time = time.time() - self.start_time
      #   print("t={:6.0f},s={:9d},th={}:{}RM{:02d}>RM{:02d}| l={:.0f},v={:.5f},pr={:.5f}".format(
      #         elapsed_time, global_t, self.thread_index, self.indent, 
      #         self.game_state.prev_room_no, self.game_state.room_no,
      #         self.game_state.lives, value_, self.game_state.psc_reward))

      if self.tes > 0:
        if self.game_state.lives < self.episode_liveses[-2]:
          elapsed_time = time.time() - self.start_time
          print("t={:6.0f},s={:9d},th={}:{}l={:.0f}>{:.0f}RM{:02d}|".format(
                elapsed_time, global_t, self.thread_index, self.indent, 
                self.episode_liveses[-2], self.game_state.lives, self.game_state.room_no))

      # seperate steps after getting reward
      if self.game_state.reward > 0:
        if not terminal:
          break

      if terminal:
        terminal_end = True
        elapsed_time = time.time() - self.start_time
        end_mark = "end" if self.terminate_on_lives_lost else "END"
        print("t={:6.0f},s={:9d},th={}:{}r={:3.0f}@{}|".format(
              elapsed_time, global_t, self.thread_index, self.indent, self.episode_reward, end_mark))

        self._record_score(sess, summary_writer, summary_op, score_input,
                           self.episode_reward, global_t)
          
        if self.tes > 0:
          if self.options.record_new_room_dir is not None \
             and self.game_state.new_room >= 0:
            dirname = "s{:09d}-th{}-r{:03.0f}-RM{:02d}".format(global_t,  self.thread_index,\
                       self.episode_reward, self.game_state.new_room)
            dirname = os.path.join(self.options.record_new_room_dir, dirname)
            os.makedirs(dirname)
            for index, screen in enumerate(self.episode_screens):
              filename = "{:06d}.png".format(index)
              filename = os.path.join(dirname, filename)
              screen_image = screen
              if self.options.compress_frame:
                screen_image = np.frombuffer(lzma.decompress(screen), dtype=np.uint8).reshape((210, 160))
              cv2.imwrite(filename, screen_image)
            print("@@@ New Room record screens saved to {}".format(dirname))

          if self.episode_reward > self.max_episode_reward:
            if self.options.record_new_record_dir is not None:
              dirname = "s{:09d}-th{}-r{:03.0f}-RM{:02d}".format(global_t,  self.thread_index,\
                         self.episode_reward, self.game_state.room_no)
              dirname = os.path.join(self.options.record_new_record_dir, dirname)
              os.makedirs(dirname)
              for index, screen in enumerate(self.episode_screens):
                filename = "{:06d}.png".format(index)
                filename = os.path.join(dirname, filename)
                screen_image = screen
                if self.options.compress_frame:
                  screen_image = np.frombuffer(lzma.decompress(screen), dtype=np.uint8).reshape((210, 160))
                cv2.imwrite(filename, screen_image)
              print("@@@ New Record screens saved to {}".format(dirname))
            self.max_episode_reward = self.episode_reward
            if self.options.record_all_non0_record:
              self.max_episode_reward = 0

          self.max_reward = 0.0
          self.episode_states = []
          self.episode_actions = []
          self.episode_rewards = []
          self.episode_values = []
          self.episode_liveses = []
          self.episode_scores.add(self.episode_reward, global_t, self.thread_index)
          if self.options.record_new_record_dir is not None \
             or self.options.record_new_room_dir is not None:
            self.episode_screens= []

        self.episode_reward = 0
        self.steps = 0
        self.no_reward_steps = 0
        self.game_state.reset()
        if self.options.use_lstm:
          self.local_network.reset_state()
        break

    if self.thread_index == 0 and self.local_t % self.options.performance_log_interval < self.options.local_t_max:
      elapsed_time = time.time() - self.start_time
      steps_per_sec = global_t / elapsed_time
      print("### Performance : {} STEPS in {:.0f} sec. {:.0f} STEPS/sec. {:.2f}M STEPS/hour".format(
            global_t,  elapsed_time, steps_per_sec, steps_per_sec * 3600 / 1000000.))

    if self.options.gym_eval:
      diff_local_t = self.local_t - start_local_t
      return diff_local_t, terminal_end

    # don't train if following condition
    # requirement for OpenAI Gym: --terminate-on-lives-lost=False
    if self.options.terminate_on_lives_lost and (self.thread_index == 0) and (not self.options.train_in_eval):
      return 0, terminal_end
    else:
      if self.tes > 0:
        _ = self.episode_scores.is_highscore(self.episode_reward)
        if self.episode_reward > self.max_reward:
          self.max_reward = self.episode_reward
          if True:
            tes = self.tes
            # requirement for OpenAI Gym: --test-extend=False
            if self.options.tes_extend and self.initial_lives != 0:
              tes *= self.options.tes_extend_ratio * (self.game_state.lives / self.initial_lives)
              if self.game_state.lives == self.initial_lives:
                tes *= 2
              tes = int(tes)
            tes = min(tes, len(self.episode_states))
            print("[OHL]SCORE={:3.0f},s={:9d},th={},lives={},steps={},tes={},RM{:02d}".format(self.episode_reward,  global_t, self.thread_index, self.game_state.lives, self.steps, tes, self.game_state.room_no))
            if tes == 0:
              states = []
              actions = []
              rewards = []
              values = []
              liveses = self.episode_liveses[-1:]
            else:
              states = self.episode_states[-tes:]
              actions = self.episode_actions[-tes:]
              rewards = self.episode_rewards[-tes:]
              values = self.episode_values[-tes:]
              liveses = self.episode_liveses[-tes-1:]
            if self.options.clear_history_after_ohl:
              self.episode_states = []
              self.episode_actions = []
              self.episode_rewards = []
              self.episode_values = []
              self.episode_liveses = self.episode_liveses[-2:]

      if len(states) > 0:
        R = 0.0
        if not terminal_end:
          R = self.local_network.run_value(sess, self.game_state.s_t)

        actions.reverse()
        states.reverse()
        rewards.reverse()
        values.reverse()

        batch_si = []
        batch_a = []
        batch_td = []
        batch_R = []

        lives = liveses.pop()
        # compute and accmulate gradients
        for(ai, ri, si, Vi) in zip(actions, rewards, states, values):
          # Consider the number of lives
          if (not self.options.use_gym) and self.initial_lives != 0.0 and not self.terminate_on_lives_lost:
            prev_lives = liveses.pop()
            if prev_lives > lives:
              weight = self.options.lives_lost_weight
              rratio = self.options.lives_lost_rratio
              R *= rratio * ( (1.0 - weight) + weight * (lives / prev_lives) )
              ri = self.options.lives_lost_reward
              lives = prev_lives

          R = ri + self.options.gamma * R
          td = R - Vi
          a = np.zeros([self.options.action_size])
          a[ai] = 1

          batch_si.append(si)
          batch_a.append(a)
          batch_td.append(td)
          batch_R.append(R)

        if self.options.use_lstm:
          batch_si.reverse()
          batch_a.reverse()
          batch_td.reverse()
          batch_R.reverse()

          sess.run( self.accum_gradients,
                    feed_dict = {
                      self.local_network.s: batch_si,
                      self.local_network.a: batch_a,
                      self.local_network.td: batch_td,
                      self.local_network.r: batch_R,
                      self.local_network.initial_lstm_state: start_lstm_state,
                      self.local_network.step_size : [len(batch_a)] } )
        else:
          sess.run( self.accum_gradients,
                    feed_dict = {
                      self.local_network.s: batch_si,
                      self.local_network.a: batch_a,
                      self.local_network.td: batch_td,
                      self.local_network.r: batch_R} )
          
        cur_learning_rate = self._anneal_learning_rate(global_t)

        sess.run( self.apply_gradients,
                  feed_dict = { self.learning_rate_input: cur_learning_rate } )

      # return advanced local step size
      diff_local_t = self.local_t - start_local_t
      return diff_local_t, terminal_end
    
