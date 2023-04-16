import gym
import scipy.stats 
from scipy.stats import norm
from scipy.stats import truncnorm
import numpy as np

d_mu = 50_000
a_mu = 150_000
# This file includes 3 core distribution used in the sampling
# of students. 
#   1. GPA gaussian distribution
#   2. Disadvantaged income distribution 
#   3. Advantaged income distribution 


# Create a Gaussian distribution for Disadvantaged student family incomes
def disadvantaged_income():
  D_sigma = 10_000
  min, max = 0, 10_000_000
  D_gaussian = truncnorm((min - d_mu) / D_sigma, (max - d_mu) / D_sigma, 
                         loc=d_mu, scale=D_sigma)
  # Generate a single data point

  return D_gaussian.rvs()

# Create a Gaussian distribution for Advantaged student family incomes
def advantaged_income():
  A_sigma = 10_000
  min, max = 0, 10_000_000
  A_gaussian = truncnorm((min - a_mu) / A_sigma, (max - a_mu) / A_sigma, 
                         loc=a_mu, scale=A_sigma)
  # Generate a single data point
  return A_gaussian.rvs()

# Create a Gaussian distribution for student GPA
def sample_gpa():
  gpa_mu = .70
  gpa_sigma = 0.1
  min, max = 0, 1
  gpa_gaussian = truncnorm((min - gpa_mu) / gpa_sigma, (max - gpa_mu) / gpa_sigma, 
                           loc=gpa_mu, scale=gpa_sigma)
  
  # Generate a single gpa data point
  return gpa_gaussian.rvs()

# Functions to decrease or increase d_mu
def decrease_d_mu():
  global d_mu
  d_mu = d_mu * .95

def increase_d_mu():
  global d_mu
  d_mu = d_mu * 1.05

# Functions to decrease or increase a_mu
def decrease_a_mu():
  global a_mu
  a_mu = a_mu * .95

def increase_a_mu():
  global a_mu
  a_mu = a_mu * 1.05

def get_manipulated_gpa(income, threshold):
    
    unmanipulated_gpa = sample_gpa()
    gpa_increase = np.log10(income) / 40
    manipulated_gpa = unmanipulated_gpa + gpa_increase
    if (manipulated_gpa >= threshold):
      if (manipulated_gpa > 1):
        return 1
      else:
        return manipulated_gpa
    else:
      return unmanipulated_gpa
