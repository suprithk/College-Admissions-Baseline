# TO DO
# 4. Finally, tweek environment (change reward, fairness metric, etc)
# Tweek reward: Currently,the agent is mostly rejecting everyone
# We got to make it so that it values it threshold heavily but also wants to accept people

# Our environment is not getting reset when we do evaluate our incomes are not going back or our threshold

# Do we need to have a delta function so that pocar can run?


import numpy as np
import torch
import os
import tqdm
import gym
import os
import sys
import shutil

from stable_baselines3 import PPO
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


global curr_timestep
curr_timestep = 0

def train(train_timesteps, env):

    # env = Monitor(env)
    # env = DummyVecEnv([lambda: env])

    model = PPO('MultiInputPolicy', env, verbose=1, tensorboard_log="./runs/")

    checkpoint_callback = CheckpointCallback(save_freq=SAVE_FREQ, save_path=SAVE_DIR, name_prefix='ppo_model')

    model.learn(train_timesteps, callback=checkpoint_callback, tb_log_name="first run")
    model.save(SAVE_DIR + '/final_model')
    return model



def evaluate(model, num_episodes, episode_timesteps):
    global curr_timestep

    env = model.get_env()
    all_rewards = []
    d_mu_vals = []
    a_mu_vals = []
    disadvantaged_acceptances = []
    advantaged_acceptances = []
    thresholds = []

    for i in range(num_episodes):
        print("Episode " + str(i + 1))
        episode_rewards = []
        done = False
        obs = env.reset()
        # print("threshold at timestep 0: " + str(obs['threshold']))
        curr_timestep = 0
        while True:
            action, _states = model.predict(obs)
            obs, reward, done, info = env.step(action)
            # if (curr_timestep == 0):
                # print("threshold at timestep 1: " + str(info[0]['threshold']))
                # print("a_mu at timestep 1: " + str(info[0]['a_mu']))
                # # print("d_mu at timestep 0: " + str(info[0]['d_mu']))
                # print("num_advantaged_accepted: " + str(info[0]["num_advantaged_accepted"]))
                # print("num_disadvantaged_accepted: " + str(info[0]["num_disadvantaged_accepted"]))
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

            # append episode reward
            episode_rewards.append(reward)
            # if on last episode 
            if (curr_timestep == episode_timesteps):
                break

        all_rewards.append(sum(episode_rewards))

    total_group_applications = len(disadvantaged_acceptances)

    # add 1 to each to account for divide by zero
    disadvantaged_acceptances = np.array(disadvantaged_acceptances) + 1
    advantaged_acceptances = np.array(advantaged_acceptances) + 1
    a_mu_vals = np.array(a_mu_vals)
    d_mu_vals = np.array(d_mu_vals)

    # Plot Fairness metric CHANGE TO ABSOLUTE DIFFERENCE (close to 0 means fair) INCLUDE INCOME
    fairness_constant = disadvantaged_acceptances / advantaged_acceptances - 1
    # Should we also track income, only income, or both ?
    income_gap = a_mu_vals - d_mu_vals

    for i in range(total_group_applications):
        writer.add_scalar('Delta Acceptances', fairness_constant[i], i)
        writer.add_scalar('A_mu over Time', a_mu_vals[i], i)
        writer.add_scalar('D_mu over Time', d_mu_vals[i], i)
        writer.add_scalar('Delta Income', income_gap[i], i)

    # Plot thresholds
    for i in range(len(thresholds)):
        writer.add_scalar('Threshold over Time', thresholds[i], i)

    # How many of each were accepted at end of episode
    print("Given 1000 students of each")
    print("advantaged students accepted: " + str(advantaged_acceptances[len(advantaged_acceptances) - 1]))
    print("disadvantaged students accepted: " + str(disadvantaged_acceptances[len(disadvantaged_acceptances) - 1]))

    writer.close()


def main():
    env = CollegeEnv()

    check_env(env, warn=True)

    # # check if runs exists if so, clear it to start new run
    # if os.path.isdir('./runs/'):
    #     shutil.rmtree('./runs/')

    print("############################## Training PPO ##############################")
    model = train(TRAIN_TIMESTEPS, env)


    print("############################## Evaluating PPO ##############################")
    evaluate(model, NUM_EPISODES, EVALUATE_EPISODE_TIMESTEPS)

    # print("############################## Evaluating PPO_5000t ##############################")
    # model = PPO.load('./models/ppo_model_150000_steps.zip', env=env)
    # evaluate(model, NUM_EPISODES, EPISODE_TIMESTEPS)


if __name__ == "__main__":
    main()
