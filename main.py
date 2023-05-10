import numpy as np
import torch
import os
import tqdm
import gym
import os
import sys
import shutil

from sb3.ppo import PPO
# from stable_baselines3 import PPO
from stable_baselines3 import DQN
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from stable_baselines3.common.callbacks import CheckpointCallback

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

from config import *
from college_admissions import *
from students import *
from ppo_wrapper_env import *
from graphing.plot_a_mu_over_time import *
from graphing.plot_d_mu_over_time import *
from graphing.plot_delta_over_time import *
from graphing.plot_reward_over_time import *
from graphing.plot_threshold_over_time import *


global curr_timestep
curr_timestep = 0

def train(train_timesteps, env):

    env = Monitor(env)
    env = DummyVecEnv([lambda: env])

    model = PPO('MlpPolicy', env, verbose=1, tensorboard_log="./runs/")
    # model = SAC('MultiInputPolicy', env, verbose=1, tensorboard_log="./runs/")

    checkpoint_callback = CheckpointCallback(save_freq=SAVE_FREQ, save_path=SAVE_DIR, name_prefix='ppo_model')

    model.learn(train_timesteps, callback=checkpoint_callback, tb_log_name="first run")
    model.save(SAVE_DIR + '/final_model')
    return model



def evaluate(model, num_episodes, episode_timesteps):
    global curr_timestep

    env = model.get_env()
    d_mu_vals = []
    a_mu_vals = []
    disadvantaged_acceptances = []
    advantaged_acceptances = []
    thresholds = []
    delta_incomes = []

    for i in range(num_episodes):
        print("Episode " + str(i + 1))
        done = False
        obs = env.reset()
        curr_timestep = 0
        while True:
            action, _states = model.predict(obs)
            obs, reward, done, info = env.step(action)
            
            curr_timestep +=1

            # access information
            # if on advantaged student
            if (curr_timestep % 2 == 1):
                advantaged_acceptances.append(info[0]["num_advantaged_accepted"])
                a_mu_vals.append(info[0]['a_mu'])
            else:
                disadvantaged_acceptances.append(info[0]["num_disadvantaged_accepted"])
                d_mu_vals.append(info[0]['d_mu'])
            # regardless, append threshold into threshold
            thresholds.append(info[0]['threshold'])
            delta_incomes.append(info[0]['delta_income'])

            # if on last episode 
            if (curr_timestep == episode_timesteps):
                break

    total_group_applications = len(disadvantaged_acceptances)

    # add 1 to each to account for divide by zero
    disadvantaged_acceptances = np.array(disadvantaged_acceptances) 
    advantaged_acceptances = np.array(advantaged_acceptances)
    a_mu_vals = np.array(a_mu_vals)
    d_mu_vals = np.array(d_mu_vals)

    # How many of each were accepted at end of episode
    print("Given 1000 students of each")
    print("advantaged students accepted: " + str(advantaged_acceptances[len(advantaged_acceptances) - 1]))
    print("disadvantaged students accepted: " + str(disadvantaged_acceptances[len(disadvantaged_acceptances) - 1]))

    writer.close()
    plot_a_mu_over_time(a_mu_vals)
    plot_d_mu_over_time(d_mu_vals)
    plot_delta_over_time(delta_incomes)
    # plot_reward_over_time(episode_rewards)
    plot_threshold_over_time(thresholds)


def main():
    env = CollegeEnv()
    env = PPOEnvWrapper(env)

    check_env(env, warn=True)

    print("############################## Training PPO ##############################")
    model = train(TRAIN_TIMESTEPS, env)


    print("############################## Evaluating PPO ##############################")
    evaluate(model, NUM_EPISODES, EVALUATE_EPISODE_TIMESTEPS)

if __name__ == "__main__":
    main()
