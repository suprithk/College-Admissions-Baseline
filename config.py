# TRAIN CONSTANTS
TRAIN_TIMESTEPS = 100_000 # Eventually be 1_000_000
EPISODE_TIMESTEPS = 10_000 # Eventaully will be 25_000

SAVE_FREQ = TRAIN_TIMESTEPS / 2
SAVE_DIR = './models/'

# EVALUATE CONSTANTS
NUM_EPISODES = 1      
EVALUATE_EPISODE_TIMESTEPS = EPISODE_TIMESTEPS # Eventually will be 25_000

# ENVIRONMENT CONSTANTS
# initial_threshold = .6

# initial_d_mu = 50_000
D_SIGMA = 10_000

# initial_a_mu = 500_000
A_SIGMA = 10_000

GPA_MU = .6
GPA_SIGMA = .05

REGULARIZE_ADVANTAGE = True  # Regularize advantage?
# Weights for advantage, value-thresholding, and decrease-in-violation terms in Eq. 3 of the paper
BETA_0 = 1
BETA_1 = 0.325
BETA_2 = 0.325

# Threshold for delta
OMEGA = 0.05