import numpy as np
from matplotlib import pyplot as plt

def plot_a_mu_over_time(a_mu_over_time):
    plt.title(f'Advantaged Mean Income Over Time')
    plt.xlabel('Timestep')
    plt.ylabel('Income')

    i = range(len(a_mu_over_time))
    plt.plot(i, a_mu_over_time, color="blue")

    plt.show()
    plt.close()
