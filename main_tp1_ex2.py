import gridworld
import numpy as np
import importlib
import matplotlib.pyplot as plt

import funcs_exo2 as exo2

# Reload module, for dev
importlib.reload(exo2)

# Plot parameters
plt.rcParams.update({"font.size": 20})

# Get environement
env = gridworld.GridWorld1


# ########################## Question 1.4 #########################################################################

# Number of MC iterations
nmc = 1000

# Max length of trajectories
Tmax = 10

# Discount factor
gamma = 0.95

# here the v-function and q-function to be used for question 4
v_q4 = [0.87691855, 0.92820033, 0.98817903, 0.00000000, 0.67106071, -0.99447514, 0.00000000, -0.82847001, -0.87691855,
        -0.93358351, -0.99447514]

# Estimate mu
nest_mu0 = 1000
mu0_mc = exo2.mc_estimate_mu0(env, nest_mu0)

# Define the policy
pol = [0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 3]

# Set range for plot of J and J^pi
nmax = 10000
pace = 10
ngrid = np.arange(100, nmax, pace)

# Estimate J and J^pi
j_mcs = exo2.j_mc_estimates(ngrid, pol, mu0_mc, env, Tmax, gamma)
j_pi = np.dot(mu0_mc, v_q4)

# Plot the result
plt.plot(ngrid, j_mcs - j_pi)


# ########################## Question 1.5 #########################################################################

# Params
nactions = 4
nstates = 11

# Number of episodes
nits = 100

# Exploration parameter
eps = 0.2

# Initialize Q matrix and nvisits matrix with ones
# We set the value of impossible (state, action)s to - np.inf
Q = - np.inf * np.ones((nstates, nactions))
nvisits = np.ones((nstates, nactions))
for i in range(0, 11):
    for j in env.state_actions[i]:
        Q[i, j] = 1

# Perform Q-learning and compute value function by MC at each episode
Q, nvisits, vns, rewards = exo2.Qlearning(Q, nvisits, env, nits, Tmax, eps, gamma, nmc=10000)

# Optimal value function
v_star = np.array([0.877, 0.928, 0.988, 0, 0.824, 0.928, 0, 0.778, 0.824, 0.877, 0.828])

# ||v_t - v*||_{\infty}
opti_diff = np.max(np.abs(vns - v_star.reshape((11, 1))), axis=0)
# Plot the difference
plt.figure()
plt.plot(opti_diff, marker="o",  markersize=10)
plt.xlabel("Episode (=n)")
plt.ylabel("$||v_n - v^*||_{\infty}$")

# Plot cumulative mean of rewards
plt.figure()
plt.plot(np.cumsum(rewards)/np.arange(1, rewards.shape[0] + 1), marker="o", markersize=10)
plt.title("Cumulative mean of reward as a function of Q learning episode")
plt.xlabel("Episode")
plt.ylabel("Cumulative mean of rewards")



