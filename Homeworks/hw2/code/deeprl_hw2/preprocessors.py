"""Suggested Preprocessors."""

import numpy as np
from PIL import Image
import numpy as np
from collections import deque

class Preprocessor:
    """Preprocessor base class.

    This is a suggested interface for the preprocessing steps. You may
    implement any of these functions. Feel free to add or change the
    interface to suit your needs.

    Preprocessor can be used to perform some fixed operations on the
    raw state from an environment. For example, in ConvNet based
    networks which use image as the raw state, it is often useful to
    convert the image to greyscale or downsample the image.

    Preprocessors are implemented as class so that they can have
    internal state. This can be useful for things like the
    AtariPreproccessor which maxes over k frames.

    If you're using internal states, such as for keeping a sequence of
    inputs like in Atari, you should probably call reset when a new
    episode begins so that state doesn't leak in from episode to
    episode.
    """
    def process_state_for_network(self, state):
        """Preprocess the given state before giving it to the network.

        Should be called just before the action is selected.

        This is a different method from the process_state_for_memory
        because the replay memory may require a different storage
        format to reduce memory usage. For example, storing images as
        uint8 in memory is a lot more efficient thant float32, but the
        networks work better with floating point images.

        Parameters
        ----------
        state: np.ndarray
          Generally a numpy array. A single state from an environment.

        Returns
        -------
        processed_state: np.ndarray
          Generally a numpy array. The state after processing. Can be
          modified in anyway.

        """
        return state

    def process_state_for_memory(self, state):
        """Preprocess the given state before giving it to the replay memory.

        Should be called just before appending this to the replay memory.

        This is a different method from the process_state_for_network
        because the replay memory may require a different storage
        format to reduce memory usage. For example, storing images as
        uint8 in memory and the network expecting images in floating
        point.

        Parameters
        ----------
        state: np.ndarray
          A single state from an environmnet. Generally a numpy array.

        Returns
        -------
        processed_state: np.ndarray
          Generally a numpy array. The state after processing. Can be
          modified in any manner.

        """
        return state

    def process_batch(self, samples):
        """Process batch of samples.

        If your replay memory storage format is different than your
        network input, you may want to apply this function to your
        sampled batch before running it through your update function.

        Parameters
        ----------
        samples: list(tensorflow_rl.core.Sample)
          List of samples to process

        Returns
        -------
        processed_samples: list(tensorflow_rl.core.Sample)
          Samples after processing. Can be modified in anyways, but
          the list length will generally stay the same.
        """
        return samples

    def process_reward(self, reward):
        """Process the reward.

        Useful for things like reward clipping. The Atari environments
        from DQN paper do this. Instead of taking real score, they
        take the sign of the delta of the score.

        Parameters
        ----------
        reward: float
          Reward to process

        Returns
        -------
        processed_reward: float
          The processed reward
        """
        return reward

    def reset(self):
        """Reset any internal state.

        Will be called at the start of every new episode. Makes it
        possible to do history snapshots.
        """
        pass


class HistoryPreprocessor(Preprocessor):
    """Keeps the last k states.

    Useful for domains where you need velocities, but the state
    contains only positions.

    When the environment starts, this will just fill the initial
    sequence values with zeros k times.

    Parameters
    ----------
    history_length: int
      Number of previous states to prepend to state being processed.

    """

    def __init__(self, history_length=1):
        self.history = deque(maxlen=history_length)
        self.history_length = history_length

    def process_state_for_network(self, state):
        """You only want history when you're deciding the current action to take."""
        # Initialize
        if not self.history:
            for i in xrange(self.history_length):
                self.history.append(np.zeros_like(state, dtype='float32'))
        self.history.append(state)
        return np.stack(self.history, axis=-1)

    def reset(self):
        """Reset the history sequence.

        Useful when you start a new episode.
        """
        self.history.clear()

    def get_config(self):
        return {'history_length': self.history_length}


class AtariPreprocessor(Preprocessor):
    """Converts images to greyscale and downscales.

    Based on the preprocessing step described in:

    @article{mnih15_human_level_contr_throug_deep_reinf_learn,
    author =	 {Volodymyr Mnih and Koray Kavukcuoglu and David
                  Silver and Andrei A. Rusu and Joel Veness and Marc
                  G. Bellemare and Alex Graves and Martin Riedmiller
                  and Andreas K. Fidjeland and Georg Ostrovski and
                  Stig Petersen and Charles Beattie and Amir Sadik and
                  Ioannis Antonoglou and Helen King and Dharshan
                  Kumaran and Daan Wierstra and Shane Legg and Demis
                  Hassabis},
    title =	 {Human-Level Control Through Deep Reinforcement
                  Learning},
    journal =	 {Nature},
    volume =	 518,
    number =	 7540,
    pages =	 {529-533},
    year =	 2015,
    doi =        {10.1038/nature14236},
    url =	 {http://dx.doi.org/10.1038/nature14236},
    }

    You may also want to max over frames to remove flickering. Some
    games require this (based on animations and the limited sprite
    drawing capabilities of the original Atari).

    Parameters
    ----------
    new_size: 2 element tuple
      The size that each image in the state should be scaled to. e.g
      (84, 84) will make each image in the output have shape (84, 84).
    """

    def __init__(self, new_size):
        self.new_size = new_size

    def process_state_for_memory(self, state):
        """Scale, convert to greyscale and store as uint8.

        We don't want to save floating point numbers in the replay
        memory. We get the same resolution as uint8, but use a quarter
        to an eigth of the bytes (depending on float32 or float64)

        We recommend using the Python Image Library (PIL) to do the
        image conversions.
        """
        # img = Image.fromarray(state, 'RGB')
        # img = img.convert('L')
        # img = img.resize(self.new_size)
        # return np.asarray(img, dtype='uint8')
        pass

    def process_state_for_network(self, state):
        """Scale, convert to greyscale and store as float32.

        Basically same as process state for memory, but this time
        outputs float32 images.
        """
        img = Image.fromarray(state, 'RGB')
        img = img.convert('L')
        img = img.resize(self.new_size)
        return np.asarray(img, dtype='float32')[np.newaxis, :]

    def process_batch(self, samples):
        """The batches from replay memory will be uint8, convert to float32.

        Same as process_state_for_network but works on a batch of
        samples from the replay memory. Meaning you need to convert
        both state and next state values.
        """
        batch = []
        for sample in samples:
            curr_batch = []
            state = np.asarray(sample.state, dtype='float32')
            next_state = np.asarray(sample.next_state, dtype='float32')
            curr_batch.append(state)
            curr_batch.append(sample.action)
            curr_batch.append(sample.reward)
            curr_batch.append(sample.is_terminal)
            curr_batch.append(next_state)
            batch.append(curr_batch)
        return batch

    def process_reward(self, reward):
        """Clip reward between -1 and 1."""
        if reward > 0:
            return 1
        elif reward < 0:
            return -1
        else:
            return reward


class PreprocessorSequence(Preprocessor):
    """You may find it useful to stack multiple prepcrocesosrs (such as the History and the AtariPreprocessor).

    You can easily do this by just having a class that calls each preprocessor in succession.

    For example, if you call the process_state_for_network and you
    have a sequence of AtariPreproccessor followed by
    HistoryPreprocessor. This this class could implement a
    process_state_for_network that does something like the following:

    state = atari.process_state_for_network(state)
    return history.process_state_for_network(state)
    """
    def __init__(self, preprocessors):
        self.preprocessors = preprocessors

    def process_state_for_network(self, state):
        for preprocessor in self.preprocessors:
            state = preprocessor.process_state_for_network(state)
        return state

    def process_batch(self, samples):
        for preprocessor in self.preprocessors:
            samples = preprocessor.process_batch(samples)
        return samples

