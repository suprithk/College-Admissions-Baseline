import gym
import scipy.stats 
from scipy.stats import norm
from scipy.stats import truncnorm
import numpy as np

from config import *


# This file includes 3 core distribution used in the sampling
# of students. 
#   1. GPA gaussian distribution
#   2. Disadvantaged income distribution 
#   3. Advantaged income distribution 


# Create a Gaussian distribution for Disadvantaged student family incomes
def disadvantaged_income(d_mu):
  D_sigma = D_SIGMA
  min, max = 1, 10_000_000
  D_gaussian = truncnorm((min - d_mu) / D_sigma, (max - d_mu) / D_sigma, 
                         loc=d_mu, scale=D_sigma)
  # Generate a single data point

  return D_gaussian.rvs()

# Create a Gaussian distribution for Advantaged student family incomes
def advantaged_income(a_mu):
  A_sigma = A_SIGMA
  min, max = 1, 10_000_000
  A_gaussian = truncnorm((min - a_mu) / A_sigma, (max - a_mu) / A_sigma, 
                         loc=a_mu, scale=A_sigma)
  # Generate a single data point
  return A_gaussian.rvs()

# Create a Gaussian distribution for student GPA
def sample_gpa():
  gpa_mu = GPA_MU
  gpa_sigma = GPA_SIGMA
  min, max = 0, 1
  gpa_gaussian = truncnorm((min - gpa_mu) / gpa_sigma, (max - gpa_mu) / gpa_sigma, 
                           loc=gpa_mu, scale=gpa_sigma)
  
  # Generate a single gpa data point
  return gpa_gaussian.rvs()

# Functions to decrease or increase d_mu
def decrease_d_mu(d_mu):
  if (d_mu <= 10_000):
    d_mu = 10_000
  else:
    d_mu = d_mu * .999
  return d_mu

def increase_d_mu(d_mu):
  if (d_mu >= 10_000_000):
    d_mu = 10_000_000
  else:
    d_mu = d_mu * 1.001
  return d_mu

# Functions to decrease or increase a_mu
def decrease_a_mu(a_mu):
  if (a_mu <= 10_000):
    a_mu = 10_000
  else:
    a_mu = a_mu * .999
  return a_mu

def increase_a_mu(a_mu):
  if (a_mu >= 10_000_000):
    a_mu = 10_000_000
  else:
    a_mu = a_mu * 1.001
  return a_mu


def get_manipulated_gpa(income, threshold):
    
    unmanipulated_gpa = sample_gpa()
    # if income is below this, cannot manipulate
    if income <= 75_000:
      return unmanipulated_gpa
    
    gpa_increase = .08 * np.log2((income)/50_000) # income to gpa increase function
    manipulated_gpa = unmanipulated_gpa + gpa_increase

    if (manipulated_gpa >= threshold):
      if (manipulated_gpa > 1):
        return 1.0
      else:
        return manipulated_gpa
    else:
      return unmanipulated_gpa
