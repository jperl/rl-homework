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

def rollout(args, env, model):
    max_steps = args.max_timesteps or env.spec.timestep_limit

    returns = []
    for i in range(args.num_rollouts):
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

    print('rollout mean', np.mean(returns), 'std', np.std(returns))


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--num_rollouts', type=int, default=20)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--train_epochs', type=int, default=1)
    args = parser.parse_args()

    print('loading and building expert policy')
    expert_policy_fn = load_policy.load_policy(args.expert_policy_file)
    print('loaded and built')

    with tf.Session():
        tf_util.initialize()

        with open(os.path.join('expert_data', args.envname + '.pkl'), "rb") as file:
          expert_data = pickle.load(file)

        actions_shape = expert_data['actions'].shape
        print('actions', actions_shape)

        # train on the observations
        observations, actions = expert_data['observations'].tolist(), expert_data['actions'].tolist()
        model = build_model(num_actions=actions_shape[-1])

        model.fit(np.array(observations), np.array(actions)[:, 0, :], epochs=args.train_epochs)

        import gym
        env = gym.make(args.envname)
        max_steps = args.max_timesteps or env.spec.timestep_limit

        # aggregate more data per epoch
        returns = []
        for epoch in range(args.num_epochs):
            # decay beta over epochs
            beta = 1 / np.sqrt(epoch + 1)

            print('epoch', epoch, 'beta', beta)

            for i in range(args.num_rollouts):
                obs = env.reset()
                done = False
                steps = 0

                while not done:
                    use_policy = np.random.choice(2, p=[beta, 1 - beta])

                    if use_policy:
                      action = model.predict(np.expand_dims(obs, 0))
                    else:
                      # use expert
                      action = expert_policy_fn(obs[None,:])
                      observations.append(obs)
                      actions.append(action)

                    obs, r, done, _ = env.step(action)

                    steps += 1
                    if args.render:
                        env.render()
                    if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                    if steps >= max_steps:
                        break

            model.fit(np.array(observations)[-5000:], np.array(actions)[-5000:, 0, :], epochs=args.train_epochs)
            rollout(args, env, model)


if __name__ == '__main__':
    main()
