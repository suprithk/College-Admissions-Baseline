import numpy as np
from matplotlib import pyplot as plt

def plot_delta_over_time(delta_over_time):
    agent_names = list(delta_over_time.keys())
    plt.title(f'Delta Over Time', fontsize = 28)
    plt.xlabel('Timestep', fontsize = 20)
    plt.ylabel('Delta', fontsize = 20)

    for agent in agent_names:
        plt.plot(range(len(delta_over_time[agent])), delta_over_time[agent], linewidth = 4)
    plt.legend(agent_names, fontsize='x-large')

    plt.show()
    plt.close()
