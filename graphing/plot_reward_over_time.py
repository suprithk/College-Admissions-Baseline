import numpy as np
from matplotlib import pyplot as plt

def plot_reward_over_time(reward_over_time):
    plt.title(f'Reward Over Time')
    plt.xlabel('Time')
    plt.ylabel('Reward')

    i = range(len(reward_over_time))
    plt.plot(i, reward_over_time, color="blue")

    plt.show()
    plt.close()
