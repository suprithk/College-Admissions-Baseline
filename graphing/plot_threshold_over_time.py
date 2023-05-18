import numpy as np
from matplotlib import pyplot as plt 

def plot_threshold_over_time(threshold_overtime):
    agent_names = list(threshold_overtime.keys())
    plt.title(f'Threshold Over Time', fontsize = 28)
    plt.xlabel('Timestep', fontsize = 20)
    plt.ylabel('Threshold', fontsize = 20)

    for agent in agent_names:
        plt.plot(range(len(threshold_overtime[agent])), threshold_overtime[agent], linewidth = 2)
    plt.legend(agent_names, fontsize='x-large')


    plt.show()
    plt.close()