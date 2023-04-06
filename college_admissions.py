import gym
import numpy as np
from students import *

class CollegeEnv(gym.Env):

    NUM_STEPS = 10_000

    EP_LENGTH = 100

    scores_sum = 0

    student_count = 0

    def __init__(self):
        self.observation_space = gym.spaces.Dict({
            # continuous score from 0 to 4
            'gpa' : gym.spaces.Box(low=0, high=4, shape=(1,), dtype=np.float32),
            # discrete label associated with advantage - 0,1
            'label' : gym.spaces.Discrete(2),
            # income of each student ranging from 0 to 10 mil
            'income' : gym.spaces.Box(low=0, high=10_000_000, shape=(1,), dtype=np.float32),
            # threshold for admission
            'threshold' : gym.spaces.Box(low=0, high=4, shape=(1,), dtype=np.float32)
        })
        self.done = False
        self.current_step = 0

        # Initialize previous observation
        self.prev_obs = None

        # action is accepting or rejecting the student application
        self.action_space = gym.spaces.Discrete(2)

    def reset(self):
        # For each tuple, regenerate its test score via a Gaussian Dist.
        # Keep their group unchanged
        # regenerate their income using two separate income Gaussian Dist. 
        initialThreshold = [3.0,]
        obs = {
            'gpa' : sample_gpa(),
            'label' : 1,
            'income' : [sample_advantaged(),],
            'threshold' : initialThreshold
        }
        # Reset ep_steps 
        self.done = False
        # Return the initial Observation

        # Initialize previous observation
        self.prev_obs = obs

        return obs

    def step(self, action, obs):
        threshold = self.threshold(action, obs)
        if (obs['label'] == 0):
            obs = {
            'gpa' : sample_gpa(),
            'label' : 1,
            'income' : np.array(sample_advantaged()),
            'threshold' : np.array(threshold)
            }
        else:
            obs = {
            'gpa' : sample_gpa(),
            'label' : 0,
            'income' : np.array(sample_disadvantaged()),
            'threshold' : np.array(threshold)
            }
        reward = self.get_reward(action, obs)  # reward
        done = False  # termination flag
        info = {}  # optional info
        self.prev_obs = obs
        return obs, reward, done, info

    def get_reward(self, action, obs):
        # Do nothing if rejected
        if (action != 0):
            diff = obs['threshold'][0] - self.prev_obs['threshold'][0] # change in threshold
            if (diff > 0):
                return 1 * obs['threshold'][0]
            elif (diff < 0):
                return -1 * obs['threshold'][0]
            else:
                return 0

    def threshold(self, action, obs):
        global scores_sum
        global student_count
        if (action == 1):
            scores_sum += obs['gpa']
            student_count += 1
            return scores_sum/student_count
        else:
            return obs['threshold'][0] # keep same threshold

    def render(self, mode='human'):
        pass
   
# TOD
# 1. How to code the logic for manipulating scores
# 2. Reset function for initial obs
# 3. main.py
# 3.1 tensorboard callbacks 
# bruh
# bruh 
