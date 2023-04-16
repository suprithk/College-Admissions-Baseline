# TO DO
# 3. Track and graph  fairness metric 



import numpy as np
import torch
import os
import tqdm
import gym

from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from stable_baselines3.common.callbacks import CheckpointCallback

from college_admissions import *


# TRAIN CONSTANTS
TRAIN_TIMESTEPS = 50_000

SAVE_FREQ = TRAIN_TIMESTEPS / 2
SAVE_DIR = './models/'

# EXP_DIR = './experiments/ppo/'

# EVALUATE CONSTANTS
NUM_EPISODES = 10
EPISODE_TIMESTEPS = 1000


global curr_timestep
curr_timestep = 0

def train(train_timesteps, env):

    env = Monitor(env)
    env = DummyVecEnv([lambda: env])

    model = PPO('MultiInputPolicy', env, verbose=1, tensorboard_log="./runs/")

    checkpoint_callback = CheckpointCallback(save_freq=SAVE_FREQ, save_path=SAVE_DIR, name_prefix='ppo_model')

    model.learn(train_timesteps, callback=checkpoint_callback, tb_log_name="first run")
    model.save(SAVE_DIR + '/final_model')
    return model


# analyze reward and fairness
def evaluate(model, num_episodes, episode_timesteps):
    global curr_timestep

    env = model.get_env()
    all_rewards = []
    for i in range(num_episodes):
        print("Episode " + str(i + 1))
        episode_rewards = []
        done = False
        obs = env.reset()
        curr_timestep = 0
        while not done:
            action, _states = model.predict(obs)
            obs, reward, done, info = env.step(action)
            curr_timestep +=1 
            if (curr_timestep == episode_timesteps):
                break
            episode_rewards.append(reward)

        all_rewards.append(sum(episode_rewards))

    for x in all_rewards:
        print(x)
    mean_ep_reward = np.mean(all_rewards)
    # plot the rewards, delta for fairness using tensorboard writer scalar


def main():
    print("LESS GOOO")
    env = CollegeEnv()

    check_env(env, warn=True)

    print("############################## Training PPO ##############################")
    model = train(TRAIN_TIMESTEPS, env)

    
    print("############################## Evaluating PPO ##############################")
    evaluate(model, NUM_EPISODES, EPISODE_TIMESTEPS)

    # print("############################## Evaluating PPO_5000t ##############################")
    # model = PPO.load('./models/ppo_model_5000_steps.zip', env=env)
    # evaluate(model, NUM_EPISODES, EPISODE_TIMESTEPS)


if __name__ == "__main__":
    main()
