import numpy as np


def optimal_bellman_operator(rvecs, pmats, v, gamma):
    ns = len(rvecs)
    tauv = np.zeros((ns, ))
    for i in range(0, ns):
        tauv[i] = np.max(rvecs[i] + gamma * np.dot(pmats[i], v))
    return tauv


def greedy_policy(rvecs, pmats, v, gamma):
    ns = len(rvecs)
    pi = np.zeros((ns, ))
    for i in range(0, ns):
        pi[i] = np.argmax(rvecs[i] + gamma * np.dot(pmats[i], v))
    return pi.astype(int)


def value_iteration(rvecs, pmats, v0, gamma, maxits=10000, epsilon=0.01):
    vt = np.copy(v0)
    ns = v0.shape[0]
    vhist = list()
    vhist.append(vt.reshape((ns, 1)))
    for k in range(0, maxits):
        vtplus1 = optimal_bellman_operator(rvecs, pmats, vt, gamma)
        vhist.append(vtplus1.reshape((ns, 1)))
        if np.max(np.abs(vt - vtplus1)) < epsilon:
            print(str(epsilon) + "-optimal policy reached after " + str(k) + " iterations.")
            pistar = greedy_policy(rvecs, pmats, vtplus1, gamma)
            return pistar, vhist
        vt = vtplus1
    pistar = greedy_policy(rvecs, pmats, vtplus1, gamma)
    return pistar, vhist


def optimality_gap(vhist, vopti):
    vhist_mat = np.concatenate(vhist, axis=1)
    ns = vopti.shape[0]
    diff = vhist_mat - vopti.reshape((ns, 1))
    return np.max(np.abs(diff), axis=0)


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
    vks = [np.array([np.inf, np.inf, np.inf])]
    ns = pi0.shape[0]
    pikplus1 = np.copy(pi0)
    for k in range(0, maxit):
        vk = policy_evaluation(rvecs, pmats, pikplus1, gamma)
        vks.append(vk.reshape((ns, 1)))
        pikplus1 = greedy_policy(rvecs, pmats, vk, gamma)
        if np.all(vks[k + 1] == vks[k]):
            print("Yes")
            return pikplus1, vks[1:]
        print(k)
    return pikplus1, vks[1:]


