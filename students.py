import gym
import scipy.stats 
from scipy.stats import norm
from scipy.stats import truncnorm

d_mu = 70_000
a_mu = 150_000
# This file includes 3 core distribution used in the sampling
# of students. 
#   1. GPA gaussian distribution
#   2. Disadvantaged income distribution 
#   3. Advantaged income distribution 


# Create a Gaussian distribution for Disadvantaged student family incomes
def sample_disadvantaged():
  D_sigma = 10_000
  D_gaussian = norm(d_mu, D_sigma)

  # Generate a single data point
  return D_gaussian.rvs()

# Create a Gaussian distribution for Advantaged student family incomes
def sample_advantaged():
  A_sigma = 10_000
  A_gaussian = norm(a_mu, A_sigma)

  # Generate a single data point
  return A_gaussian.rvs()

# Create a Gaussian distribution for student GPA
def sample_gpa():
  gpa_mu = 3.00
  gpa_sigma = 0.40
  min, max = 0, 4
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