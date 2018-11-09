import gridworld
import numpy as np
import time
import importlib
import matplotlib.pyplot as plt

import reinf_learning as rl
importlib.reload(rl)

env = gridworld.GridWorld1






# ########################## Question 1.4 #########################################################################

# Number of MC iterations
nmc = 1000
# Max length of trajectories
Tmax = 10
# DIscount factor
gamma = 0.95

# here the v-function and q-function to be used for question 4
v_q4 = [0.87691855, 0.92820033, 0.98817903, 0.00000000, 0.67106071, -0.99447514, 0.00000000, -0.82847001, -0.87691855,
        -0.93358351, -0.99447514]
# Estimate mu
nest_mu0 = 1000
mu0_mc = rl.mc_estimate_mu0(env, nest_mu0)

nmax = 10000
pace = 10
ngrid = np.arange(100, nmax, pace)

# Estimate J and J^pi
j_mcs = rl.j_mc_estimates(ngrid, mu0_mc, env, Tmax, gamma)
j_pi = np.dot(mu0_mc, v_q4)

plt.plot(ngrid, j_mcs - j_pi)