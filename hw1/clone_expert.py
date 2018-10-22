#!/usr/bin/env python

"""
Code to load an expert policy and generate roll-out data for behavioral cloning.
Example usage:
    python run_expert.py experts/Humanoid-v1.pkl Humanoid-v1 --render \
            --num_rollouts 20

Author of this script and included expert policies: Jonathan Ho (hoj@openai.com)
"""

import os
import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy

def build_model(num_actions):
    model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(512, activation=tf.nn.relu),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Dense(num_actions)
    ])
    model.compile(optimizer='adam', loss='mean_absolute_error')
    return model

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_data', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                    help='Number of expert roll outs')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--train_epochs', type=int, default=10)
    args = parser.parse_args()

    with tf.Session():
        tf_util.initialize()

        with open(os.path.join('expert_data', args.envname + '.pkl'), "rb") as file:
          expert_data = pickle.load(file)

        # clone the observations
        observations, actions = expert_data['observations'], expert_data['actions']
        print('actions', actions.shape)
        model = build_model(num_actions=actions.shape[-1])
        model.fit(observations, actions[:, 0, :], epochs=args.train_epochs)

        # rollout the cloned model
        import gym
        env = gym.make(args.envname)
        max_steps = args.max_timesteps or env.spec.timestep_limit

        returns = []
        for i in range(args.num_rollouts):
            print('iter', i)
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0
            while not done:
                obs = np.expand_dims(obs, 0)
                action = model.predict(obs)
                obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1
                if args.render:
                    env.render()
                if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                if steps >= max_steps:
                    break

            returns.append(totalr)


        print('returns', returns)
        print('mean', np.mean(returns), 'std', np.std(returns))

if __name__ == '__main__':
    main()
