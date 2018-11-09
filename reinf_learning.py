import numpy as np


def collect_trajs(env, n, Tmax, gamma):
    records_dict = {}
    for i in range(0, n):
        s0 = np.random.randint(0, 11)
        s = s0
        states = [s]
        rewards = []
        actu = 1
        for t in range(0, Tmax):
            # Using env.step with terminal state each time led to and error:
            # AssertionError: line 66, in step assert action in self.state_actions[state]
            if s == 3 or s == 6:
                r = 0
                rewards.append(r * actu)
                break
            elif s == 0 or s == 1 or s == 2 or s == 5 or s == 7 or s == 8 or s == 9:
                a = 0
            else:
                a = 3
            s, r, term = env.step(s, a)
            states.append(s)
            rewards.append(r * actu)
            actu *= gamma
            if term:
                break
        records_dict[i] = np.array(states), np.array(rewards)
    return records_dict


def concat_collected_trajs(records_dict1, records_dict2):
    n1 = len(records_dict1.keys())
    n2 = len(records_dict2.keys())
    for i in range(0, n2):
        records_dict1[n1 + i] = records_dict2[i]
    return records_dict1


def mc_estimate_Vn(collected_trajs):
    Vn_estimates = np.zeros((11, ))
    ntrajs = np.zeros((11, ))
    for i in collected_trajs.keys():
        s0 = collected_trajs[i][0][0]
        Vn_estimates[s0] += np.sum(collected_trajs[i][1])
        ntrajs[s0] += 1
    return Vn_estimates / ntrajs


def mc_estimate_mu0(env, n):
    mu0_mc = np.zeros((11, ))
    for i in range(0, n):
        s0 = env.reset()
        mu0_mc[s0] += 1
    return mu0_mc / np.sum(mu0_mc)


def j_mc_estimates(ngrid, mu0_mc, env, Tmax, gamma):
    jmcs = np.zeros((ngrid.shape[0]))
    for i in range(0, ngrid.shape[0]):
        if i == 0:
            records_dict1 = collect_trajs(env, ngrid[i], Tmax, gamma)
        else:
            records_dict2 = collect_trajs(env, ngrid[i] - ngrid[i - 1], Tmax, gamma)
            records_dict1 = concat_collected_trajs(records_dict1, records_dict2)
        v_mc = mc_estimate_Vn(records_dict1)
        jmcs[i] = np.dot(mu0_mc, v_mc)
    return jmcs



