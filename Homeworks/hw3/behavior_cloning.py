# Author: Music Lee <yuezhanl@andrew.cmu.edu>

import gym
from deeprl_hw3.imitation import load_model, generate_expert_training_data, test_cloned_policy, wrap_cartpole
import keras

def main():
    model_config_path = "CartPole-v0_config.yaml"
    model_weight_path = "CartPole-v0_weights.h5f"
    env = gym.make('CartPole-v0')
    #env = wrap_cartpole(env)
    clone_model  = load_model(model_config_path=model_config_path)
    expert_model = load_model(model_config_path=model_config_path, model_weights_path=model_weight_path)
    states, actions = generate_expert_training_data(expert_model, env, num_episodes=100, render=True)
    optimizer = keras.optimizers.Adam()
    clone_model.compile(optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    clone_model.fit(states, actions, epochs=50)
    test_cloned_policy(env, expert_model, num_episodes=5, render=False)
    test_cloned_policy(env, clone_model, num_episodes=5, render=False)




if __name__ == "__main__":
    main()