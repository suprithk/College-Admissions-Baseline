from college_admissions import getReward
import numpy as np
import torch
import os
import tqdm

# Create agents in a seperate directory
# TODO
from agents import MLEGreedyAgent, MLEGreedyAgentParams
from agents import PPO

def train(train_timesteps, env):
    model = None
    should_load = False


def evaluate(env, agent, num_eps, num_timesteps, name, seeds, eval_path, algorithm=None):
    reward_fn = getReward

    for ep in range(num_eps):
        # Fill in the sets for each element in observation space
        ep_data = {

        }

        obs = env.reset()
        done = False

        print(f'{name} EPISODE {ep}:')

        for t in tqdm.trange(num_timesteps):
            action = None
            if isinstance(agent, MLEGreedyAgent):
                    action = agent.act(obs, done)
            elif isinstance(agent, PPO):
                    action = agent.predict(obs)[0]

            obs, _, done, _ = env.step(action)

            # Update admission variables for ep_data (optional)
            # TODO 

            r = get_reward(self, action, obs)

            # append functions into ep_data
            # TODO

        # Store episodic data in eval data
        # TODO
    #return mean episodic reward
    return eval_data 
