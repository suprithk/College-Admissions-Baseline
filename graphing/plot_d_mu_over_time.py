import numpy as np
from matplotlib import pyplot as plt 

def plot_d_mu_over_time(d_mu_over_time):
    plt.title(f'Disadvantaged Mean Income Over Time')
    plt.xlabel('Timestep')
    plt.ylabel('Income')

    i = range(len(d_mu_over_time))
    plt.plot(i, d_mu_over_time, color="pink")

    plt.show()
    plt.close()