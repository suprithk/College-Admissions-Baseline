import gym
import numpy as np
from students import *

import main

class CollegeEnv(gym.Env):

    NUM_STEPS = 10_000 # unused

    EP_LENGTH = 100 # unused

    global scores_sum
    scores_sum = 0

    global student_count
    student_count = 0

    def __init__(self):
        self.observation_space = gym.spaces.Dict({
            # continuous score from 0 to 4
            'gpa' : gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            # discrete label associated with advantage - 0,1
            'label' : gym.spaces.Discrete(2),
            # income of each student ranging from 0 to 10 mil
            'income' : gym.spaces.Box(low=0, high=10_000_000, shape=(1,), dtype=np.float32),
            # threshold for admission
            'threshold' : gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        })
        self.done = False
        self.current_step = 0

        # Initialize previous observation
        self.prev_obs = None

        # action is accepting or rejecting the student application
        self.action_space = gym.spaces.Discrete(2)

    def reset(self):
        # Reset the environment and give obs an advantaged applicant
        initialThreshold = .8
        temp_income = advantaged_income()

        obs = {
            'gpa' : get_manipulated_gpa(temp_income, initialThreshold),
            'label' : 1,
            'income' : np.array(temp_income),
            'threshold' : np.array(initialThreshold)
        }
        # Reset ep_steps 
        self.done = False
        self.current_step = 0

        # Initialize previous observation
        self.prev_obs = obs

        # Return the initial Observation
        return obs

    def step(self, action):
        # increment timestep
        self.current_step += 1

        # Get new obs
        threshold = self.threshold(action)
        if (self.prev_obs['label'] == 0):
            income_temp = advantaged_income()
            obs = {
            'gpa' : get_manipulated_gpa(income_temp, threshold),
            'label' : 1,
            'income' : np.array(income_temp),
            'threshold' : np.array(threshold)
            }
        else:
            income_temp = disadvantaged_income()
            obs = {
            'gpa' : get_manipulated_gpa(income_temp, threshold),
            'label' : 0,
            'income' : np.array(income_temp),
            'threshold' : np.array(threshold)
            }

        reward = self.get_reward(action, obs)
        done = False # there is no end case for this env
        info = {}  # optional info
        self.prev_obs = obs # update prev_obs
        
        return obs, reward, done, info

    def get_reward(self, action, obs):
        # 1: acceptance -> return the threshold + inc/dec
        # 0: rejection -> do nothing 
        if (action != 0):
            diff = obs['threshold'] - self.prev_obs['threshold'] # change in threshold
            return obs['threshold'] + diff
        else:
            return 0

    def threshold(self, action):
        # return the average attendee score
        global scores_sum
        global student_count
        if (action == 1):
            scores_sum += self.prev_obs['gpa']
            student_count += 1
            return scores_sum/student_count
        else:
            return self.prev_obs['threshold'] # keep same threshold

    def render(self, mode='human'):
        pass
