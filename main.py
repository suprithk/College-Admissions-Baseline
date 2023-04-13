import numpy as np
# import torch
import os
import tqdm
import gym

from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.env_checker import check_env

from college_admissions import *

NUM_TIMESTEPS = 200
NUM_EPISODES = 100

global count 
count = 0

def train(train_timesteps, env):
    model = PPO('MultiInputPolicy', env, verbose=1) # tensorboard_log="./runs")
    model.learn(train_timesteps) #tb_log_name="first run")
    return model

def evaluate(model, num_episodes):
    global count

    env = model.get_env()
    all_rewards = []
    for i in range(num_episodes):
        print(i)
        episode_rewards = []
        done = False
        obs = env.reset()
        count = 0
        while not done:
            action, _states = model.predict(obs)
            obs, reward, done, info = env.step(action)
            count +=1 
            if (count == 200):
                break
            episode_rewards.append(reward)

        all_rewards.append(sum(episode_rewards))

    for x in all_rewards:
        print(x)
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
