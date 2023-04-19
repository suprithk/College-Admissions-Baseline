import gym
import numpy as np

from students import *
from config import *
import main


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

        self.num_disadvantaged_accepted = 0
        self.num_advantaged_accepted = 0

    def reset(self):
        # Reset the environment and give obs an advantaged applicant
        initialThreshold = .1
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

        self.num_disadvantaged_accepted = 0
        self.num_advantaged_accepted = 0

        # Set environment variables back as well
        global d_mu
        d_mu = initial_d_mu
        global a_mu
        a_mu = initial_a_mu

        # Return the initial Observation
        return obs

    def step(self, action):
        # increment timestep
        self.current_step += 1
        # Get new threshold
        threshold = self.threshold(action)

        info = {}

        # Add threshold to info
        info['threshold'] = threshold

        # Increase or decrease income mu
        # If disadvantaged student
        if (self.prev_obs['label'] == 0):
            # If rejected
            if (action == 0):
                decrease_d_mu()
            else:
                self.num_disadvantaged_accepted += 1
                increase_d_mu()
            
            # put new d_mu into info
            info['d_mu'] = get_d_mu()
            info['num_disadvantaged_accepted'] = self.num_disadvantaged_accepted
        # If advantaged student
        else:
            # If rejected
            if (action == 0):
                decrease_a_mu()
            else:
                self.num_advantaged_accepted += 1
                increase_a_mu()
            
            # put new a_mu into info
            info['a_mu'] = get_a_mu()
            info['num_advantaged_accepted'] = self.num_advantaged_accepted

        # Get new observation
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

        if (self.current_step == EPISODE_TIMESTEPS):
            done = True
        else:
            done = False 

        self.prev_obs = obs # update prev_obs
        
        return obs, reward, done, info

    def get_reward(self, action, obs):
        # 1: acceptance -> return the threshold + inc/dec
        # 0: rejection -> do nothing 
        diff = obs['threshold'] - self.prev_obs['threshold'] # change in threshold if acceptance

        # if we should not accept (greedily)
        if (diff <= 0):
            # if accept, punish, otherwise reward
            if (action != 0):
                return -1
            else:
                return 1
        # if we should accept
        else:
            # if accept, reward, otherwise punish
            if (action != 0):
                return 1
            else:
                return -1

        # if (action != 0):
        #     diff = obs['threshold'] - self.prev_obs['threshold'] # change in threshold
        #     return obs['threshold'] + (diff * 10) # really want to increase threshold
        # else:
        #     return 0

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
