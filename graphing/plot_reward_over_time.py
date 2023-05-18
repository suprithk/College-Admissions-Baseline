import numpy as np
from matplotlib import pyplot as plt

def plot_reward_over_time(reward_over_time):
    agent_names = list(reward_over_time.keys())
    plt.title(f'Reward Over Time', fontsize = 28)
    plt.xlabel('Time', fontsize = 20)
    plt.ylabel('Reward', fontsize = 20)

    for agent in agent_names:
        plt.plot(range(len(reward_over_time[agent])), reward_over_time[agent], linewidth = 4)
    plt.legend(agent_names, fontsize='x-large')

    plt.show()
    plt.close()
