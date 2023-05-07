import numpy as np
from matplotlib import pyplot as plt 

def plot_threshold_over_time(threshold_overtime):
    plt.title(f'Threshold Over Time')
    plt.xlabel('Timestep')
    plt.ylabel('Threshold')

    i = range(len(threshold_overtime))
    plt.plot(i, threshold_overtime, color="orange")

    plt.show()
    plt.close()