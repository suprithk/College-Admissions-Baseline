import numpy as np
import torch
import os
import tqdm
import gym

from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.env_checker import check_env

from college_admissions import *

NUM_TIMESTEPS = 200
NUM_EPISODES = 10_000

def train(train_timesteps, env):
    model = PPO('MultiInputPolicy', env, verbose=1, tensorboard_log="./runs")
    model.learn(train_timesteps, tb_log_name="first run")
    return model

def evaluate(model, num_episodes):
    env = model.get_env()
    all_rewards = []
    for i in range(num_episodes):
        episode_rewards = []
        done = False
        obs = env.reset()
        while not done:
            action, _states = model.predict(obs)
            obs, reward, done, info = env.step(action, obs)
            episode_rewards.append(reward)

        all_rewards.append(sum(episode_rewards))

    mean_ep_reward = np.mean(all_rewards)
    # plot the rewards, delta for fairness


def main():
    print("LESS GOOO")
    env = CollegeEnv()
    # check_env(env, warn=True)

    model = train(NUM_TIMESTEPS, env)
    evaluate(model, NUM_EPISODES)

if __name__ == "__main__":
    main()


# def evaluate(env, agent, num_eps, num_timesteps, name, seeds, eval_path, algorithm=None):
#     reward_fn = getReward

#     for ep in range(num_eps):
#         # Fill in the sets for each element in observation space
#         ep_data = {

#         }

#         obs = env.reset()
#         done = False

#         print(f'{name} EPISODE {ep}:')

#         for t in tqdm.trange(num_timesteps):
#             action = None
#             if isinstance(agent, MLEGreedyAgent):
#                     action = agent.act(obs, done)
#             elif isinstance(agent, PPO):
#                     action = agent.predict(obs)[0]

#             obs, _, done, _ = env.step(action)

#             # Update admission variables for ep_data (optional)
#             # TODO 

#             r = get_reward(self, action, obs)

#             # append functions into ep_data
#             # TODO

#         # Store episodic data in eval data
#         # TODO
#     #return mean episodic reward
#     return eval_data 