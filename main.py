import numpy as np
import importlib
import matplotlib.pyplot as plt


import dynamic_prog as dp
importlib.reload(dp)


# ########################## Question 1.1 #########################################################################

# Number of states
nstates = 3

# Number of actions
nactions = 3

# List of transition matrices and reward matrices
pmats = []
rvecs = []

# Initialize transition matrices and rewrd matrices to 0
for k in range(0, nstates):
    pmats.append(np.zeros((nstates, nstates)))
    rvecs.append(np.zeros((nactions, )))

# Transition matrix for state s0
pmats[0][0, 0] = 0.55
pmats[0][0, 1] = 0.45
pmats[0][1, 0] = 0.3
pmats[0][1, 1] = 0.7
pmats[0][2, 0] = 1

# Transition matrix for state s1
pmats[1][0, 0] = 1
pmats[1][1, 1] = 0.4
pmats[1][1, 2] = 0.6
pmats[1][2, 1] = 1

# Transition matrix for state s0
pmats[2][0, 1] = 1
pmats[2][1, 1] = 0.6
pmats[2][1, 2] = 0.4
pmats[2][2, 2] = 1

# Fill non negative values for rewards vectors
rvecs[0][2] = 5 / 100
rvecs[2][1] = 1
rvecs[2][2] = 9 / 10

# Set discount parameter to 0.95
gamma = 0.95

# The optimal policy vector that we have guessed and its corresponding value vector
pi_star = np.array([1, 1, 2])
v_star = dp.policy_evaluation(rvecs, pmats, pi_star, gamma)


# ########################## Question 1.2 #########################################################################

# Question 1.2: Value iteration
# Set nitial values to 0
v0 = np.zeros((nstates, ))
# Perform value iteration and record value vectors history in hist_vit
pi_vit, hist_vit = dp.value_iteration(rvecs, pmats, v0, gamma, epsilon=0.01)
# Get optimality gap along iterations in infinite norm
opti_gap_vit = dp.optimality_gap(hist_vit, v_star)
# Plot optimality gap
plt.figure()
plt.plot(opti_gap_vit, marker="o")
plt.ylabel("$||v - v^*||_{\infty}$")
plt.xlabel("Value iteration")


# ########################## Question 1.3 #########################################################################

# Initialize pi0 to [a0, a0, a0]
pi0 = np.zeros((nstates, ), dtype=int)
# Perform policy iteration and record value vectors history in hist_pit
pi_pit, hist_pit = dp.policy_iteration(rvecs, pmats, pi0, gamma)
# Get optimality gap along iterations in infinite norm
opti_gap_pit = dp.optimality_gap(hist_pit, v_star)
# Plot optimality gap
plt.figure()
plt.plot(opti_gap_pit, marker="o")
plt.ylabel("$||v - v^*||_{\infty}$")
plt.xlabel("iterations")
plt.title("Policy iteration")