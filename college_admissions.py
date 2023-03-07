import gym
import numpy as np

class CollegeEnv(gym.Env):

    NUM_STEPS = 10_000

    EP_LENGTH = 100

    current_step = 0

    def __init__(self, n = 10):
        self.observation_space = gym.spaces.Tuple([
            gym.spaces.Tuple([
                # continuous score from 0 to 4
                gym.spaces.Box(low=0, high=4, shape=(1,), dtype=np.float32),
                # discrete label associated with advantage - 0,1
                gym.spaces.Discrete(2),
                # income of each student ranging from 0 to 1
                gym.spaces.Box(low=0, high=10_000_000, shape=(1,), dtype=np.float32)
            ])
            for _ in range(n)
        ])
        self.done = False
        self.current_step = 0
        # TODO
        # incorporate threshold 

        # action is accepting or rejecting the student application
        self.action_space = gym.spaces.MultiBinary(n)

    def reset(self):
        # obs = ((0.1, 0.2, 0.3), 0, 0.0)  # initial observation
        self.rng = default.rng
	
	    for _ in range(n)
	        Student = np.array[(self.rng * self.max_Test_Score), 
                            (self.rng * self.max_Label), 
                            (self.rng * self.max_Income)]
	
	    return np.array[self.Students]
        # return obs

    def step(self, action):
        # TODO
        # regenerate scores, keep the same label, change income based on income func
        # reward is sum of all accepted scores
        # run through all students and calculate the sum

        self.done = self.current_step == self.EP_LENGTH
        obs = ((0.2, 0.4, 0.6), (0.3, 0.2, 0.5), (0.2, 0.4, 0.6))  # new observation
        reward = self.get_reward(action)  # reward
        done = False  # termination flag
        info = {}  # optional info
        return obs, reward, done, info

    def get_reward(self, action):
        # TODO
        # take action
        print("lol")

    def get_new_income(self, action):
        # TODO
        # calculate income based on a probability
        # can be done by checking if an rng falls below a threshold
        print("lol")


# TODO
# 1. Agent does not have control over threshold, just who it can accept
# 2. How to code the logic for manipulating scores
# 3. Income function
# 4. How do you give rewards based on observation
# 5. Fairness equation