import gym
import numpy as np
from students import *

import main
NUM_STEPS = 10_000 # unused

EP_LENGTH = 1000

class CollegeEnv(gym.Env):
 
    global scores_sum
    scores_sum = 0

    global student_count
    student_count = 0

    def __init__(self):
        self.observation_space = gym.spaces.Dict({
            # continuous score from 0 to 1
            'gpa' : gym.spaces.Box(low=np.array([0], dtype=np.float32), high=np.array([1], dtype=np.float32), shape=(1,), dtype=np.float32),
            # discrete label associated with advantage - 0,1
            'label' : gym.spaces.Discrete(2),
            # income of each student ranging from 0 to 10 mil
            'income' : gym.spaces.Box(low=np.array([0], dtype=np.float32), high=np.array([10_000_000], dtype=np.float32), shape=(1,), dtype=np.float32),
            # threshold for admission
            'threshold' : gym.spaces.Box(low=np.array([0], dtype=np.float32), high=np.array([1], dtype=np.float32), shape=(1,), dtype=np.float32)
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
            'gpa' : np.array([get_manipulated_gpa(temp_income, initialThreshold)], dtype=np.float32),
            'label' : 1,
            'income' : np.array([temp_income], dtype=np.float32),
            'threshold' : np.array([initialThreshold], dtype=np.float32)
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
            'gpa' : np.array([get_manipulated_gpa(income_temp, threshold)], dtype=np.float32),
            'label' : 1,
            'income' : np.array([income_temp], dtype=np.float32),
            'threshold' : np.array(threshold, dtype=np.float32)
            }
        else:
            income_temp = disadvantaged_income()
            obs = {
            'gpa' : np.array([get_manipulated_gpa(income_temp, threshold)], dtype=np.float32),
            'label' : 0,
            'income' : np.array([income_temp], dtype=np.float32),
            'threshold' : np.array(threshold, dtype=np.float32)
            }

        reward = float(self.get_reward(action, obs))

        if (self.current_step == EP_LENGTH):
            done = True
        else:
            done = False 

        info = {}  # optional info
        self.prev_obs = obs # update prev_obs
        
        return obs, reward, done, info

    def get_reward(self, action, obs):
        # 1: acceptance -> return the threshold + inc/dec
        # 0: rejection -> do nothing 
        if (action != 0):
            diff = obs['threshold'] - self.prev_obs['threshold'] # change in threshold
            return obs['threshold'] + (diff * obs['threshold'])
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
