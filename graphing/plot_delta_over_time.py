import numpy as np
from matplotlib import pyplot as plt

def plot_delta_over_time(delta_over_time):
    plt.title(f'Delta Over Time')
    plt.xlabel('Timestep')
    plt.ylabel('Delta')

    i = range(len(delta_over_time))
    plt.plot(i, delta_over_time, color="blue")

    plt.show()
    plt.close()
