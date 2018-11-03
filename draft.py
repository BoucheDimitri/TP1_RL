import numpy as np


################ VALUE ITERATION ##############################""

def optimal_bellman_operator(rvecs, pmats, w, gamma):
    ns = len(rvecs)
    tauw = np.zeros((ns, ))
    for i in range(0, nstates):
        tauw[i] = np.max(rvecs[i] + gamma * np.dot(pmats[i], w))
    return tauw


def greedy_policy(rvecs, pmats, w, gamma):
    ns = len(rvecs)
    pi = np.zeros((ns, ))
    for i in range(0, nstates):
        pi[i] = np.argmax(rvecs[i] + gamma * np.dot(pmats[i], w))
    return pi


def value_iteration(rvecs, pmats, w0, gamma, maxits=10000, epsilon=0.01):
    wt = np.copy(w0)
    for k in range(0, maxits):
        wtplus1 = optimal_bellman_operator(rvecs, pmats, wt, gamma)
        if np.max(np.abs(wt - wtplus1)) < epsilon:
            print(str(epsilon) + "-optimal policy reached after " + str(k) + " iterations.")
            return greedy_policy(rvecs, pmats, wtplus1, gamma)
        wt = wtplus1
        return greedy_policy(rvecs, pmats, wtplus1, gamma)


def r_pi(rvecs, pi):
    ns = len(rvecs)
    rpi = np.zeros((ns, ))
    for i in range(0, ns):
        rpi[i] = rvecs[i][pi[i]]
    return rpi


def p_pi(pmats, pi):
    ns = len(pmats)
    ppi = np.zeros((ns, ns))
    for i in range(0, ns):
        for j in range(0, ns):
            ppi[i, j] = pmats[i][pi[i], j]
    return ppi


def policy_evaluation(rvecs, pmats, pi, gamma):
    ns = len(pmats)
    rpi = r_pi(rvecs, pi)
    ppi = p_pi(pmats, pi)
    A = np.eye(ns) - gamma * ppi
    Ainv = np.linalg.inv(A)
    return np.dot(Ainv, rpi)


def policy_iteration(rvecs, pmats, pi0, gamma, maxit=100):
    vks = []
    for k in range(0, maxit):
        vk = policy_evaluation(rvecs, pmats, pi0, gamma)
        vks.append(vk)
        pikplus1 = greedy_policy(rvecs, pmats, vk, gamma)
        if k!=0:
            if vks[k + 1] == vks[k]:
                return pikplus1
    return pikplus1





################ FILL MDP ##################################

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
