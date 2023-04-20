import gym
import numpy as np

from students import *
from config import *
import main


class CollegeEnv(gym.Env):
 
    # global scores_sum
    # scores_sum = 0

    # global student_count
    # student_count = 0

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

        self.scores_sum = 0
        self.student_count = 0

        self.a_mu = 200_000
        self.d_mu = 50_000

    def reset(self):
        # Reset the environment and give obs an advantaged applicant
        initialThreshold = .6
        temp_income = advantaged_income(self.a_mu)

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
        self.prev_obs = obs              # deep copy ?

        self.num_disadvantaged_accepted = 0
        self.num_advantaged_accepted = 0

        self.scores_sum = 0
        self.student_count = 0

        # Set environment variables back as well      these are not gettin set back to original
        self.a_mu = 200_000
        self.d_mu = 50_000

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
                self.d_mu = decrease_d_mu(self.d_mu)
            else:
                self.num_disadvantaged_accepted += 1
                self.d_mu = increase_d_mu(self.d_mu)
            
            # put new d_mu into info
            info['d_mu'] = self.d_mu
            info['num_disadvantaged_accepted'] = self.num_disadvantaged_accepted
        # If advantaged student
        else:
            # If rejected
            if (action == 0):
                self.a_mu = decrease_a_mu(self.a_mu)
            else:
                self.num_advantaged_accepted += 1
                self.a_mu = increase_a_mu(self.a_mu)
            
            # put new a_mu into info
            info['a_mu'] = self.a_mu
            info['num_advantaged_accepted'] = self.num_advantaged_accepted

        # Get new observation
        if (self.prev_obs['label'] == 0):
            income_temp = advantaged_income(self.a_mu)
            obs = {
            'gpa' : np.array([get_manipulated_gpa(income_temp, threshold)], dtype=np.float32),
            'label' : 1,
            'income' : np.array([income_temp], dtype=np.float32),
            'threshold' : np.array(threshold, dtype=np.float32)
            }
        else:
            income_temp = disadvantaged_income(self.d_mu)
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

        self.prev_obs = obs # update prev_obs        deep copy???
        
        return obs, reward, done, info

    def get_reward(self, action, obs):
        # 1: acceptance -> return the threshold + inc/dec
        # 0: rejection -> do nothing 
        diff = obs['threshold'] - self.prev_obs['threshold'] # change in threshold if acceptance


        # return obs['threshold']
        # if we should not accept (greedily)
        if (diff <= 0):
            # if accept, punish, otherwise reward
            if (action != 0):
                return -.1
            else:
                return .1
        # if we should accept
        else:
            # if accept, reward, otherwise punish
            if (action != 0):
                return obs['threshold']
            else:
                return -.1

        # if (action != 0):
        #     diff = obs['threshold'] - self.prev_obs['threshold'] # change in threshold
        #     return obs['threshold'] + (diff * 10) # really want to increase threshold
        # else:
        #     return 0


# FIX this, as we progress in timesteps, accepting a student regardless of gpa
# does not affect our threshold hardly enough, so the agent just begins to accept whoever
# regardless of gpa
    def threshold(self, action):
        # return the average attendee score
        # if we reject, do nothing, otherwise update threshold
        if (action == 0):
            return self.prev_obs['threshold']
        else:
            self.scores_sum += self.prev_obs['gpa']
            self.student_count += 1
            return self.scores_sum/self.student_count

    def render(self, mode='human'):
        pass
