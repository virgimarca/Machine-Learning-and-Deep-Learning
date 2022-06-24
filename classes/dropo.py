import numpy as np
import nevergrad as ng
from scipy.stats import truncnorm
from scipy.stats import multivariate_normal


class Dropo(object):
    def __init__(self, sim_env, lambda_t, epsilon, K, instances):
        self.sim_env = sim_env
        self.sim_env.reset()
        self.lambda_t = lambda_t
        self.epsilon = epsilon
        self.K = K

        with open('/content/classes/data.npy', 'rb') as f:
            self.states = np.load(f, allow_pickle=True)
            self.next_states = np.load(f, allow_pickle=True)
            self.actions = np.load(f, allow_pickle=True)
            self.terminals = np.load(f, allow_pickle=True)

        self.mu_lowerbound = np.array([2, 2, 2]).T
        self.mu_upperbound = np.array([10, 10, 10]).T
        self.cov_lowerbound = np.array([1e-5, 1e-5, 1e-5]).T
        self.cov_upperbound = np.array([1, 1, 1]).T

        self.instances_to_use = instances

    def optimize(self, budget):

        search_space = []
        for i in range(len(self.sim_env.get_parameters()) - 1):
            # MEAN
            search_space.append(ng.p.Scalar(init=2).set_bounds(lower=0, upper=4))
        for i in range(len(self.sim_env.get_parameters()) - 1):
            # STANDARD DEVIATION
            search_space.append(ng.p.Scalar(init=2).set_bounds(lower=0, upper=4))

        params = ng.p.Tuple(*search_space)
        instrum = ng.p.Instrumentation(x=params)
        optim = ng.optimizers.CMA(parametrization=instrum, budget=budget)
        res = optim.minimize(self.function)
        return self.denormalize(np.array(res.value[1]['x'][0:3]), np.array(res.value[1]['x'][3:])), self.function(
            **res.kwargs)

    def truncnormal(self, mu, cov, size):
        l = []
        # because we keep the first mass fixed to 2.53429174
        l.append((2.53429174) * np.ones(size))
        for i, mean in enumerate(mu):
            std = cov[i]
            LB = self.mu_lowerbound[i]
            UB = self.mu_upperbound[i]

            csi = truncnorm.rvs(-2, 2, loc=mean, scale=std, size=size)
            for i, sample in enumerate(csi):
                if (sample < LB) or (sample > UB):
                    while (sample < LB) or (sample > UB):
                        sample = truncnorm.rvs(-2, 2, loc=mean, scale=std)
                    csi[i] = sample
            l.append(csi)

        return np.array(l).T

    def denormalize(self, mu, cov):
        mu = self.mu_lowerbound + (self.mu_upperbound - self.mu_lowerbound) * mu / 4
        cov = self.cov_lowerbound * (self.cov_upperbound / self.cov_lowerbound) ** (cov / 4)
        return mu, cov

    def function(self, x):
        mu = np.array(x[0:3])
        cov = np.array(x[3:])
        L = 0
        mu, cov = self.denormalize(mu, cov)
        csi = self.truncnormal(mu, cov, self.K * (self.instances_to_use))
        lambda_t = self.lambda_t
        for t in range(self.instances_to_use):
            states_csi = []
            next_states = self.next_states[t + lambda_t - 1]

            for k in range(self.K):
                self.sim_env.reset()
                self.sim_env.set_parameters(csi[t * self.K + k])
                self.sim_env.set_mujoco_state(self.states[t])

                for j in range(t, t + lambda_t):
                    state, reward, done, _ = self.sim_env.step(self.actions[j])

                states_csi.append(state)

            states_csi = np.array(states_csi)
            mean = np.mean(states_csi, axis=0)
            cov_matrix = np.cov(states_csi, rowvar=0) + np.diagflat(self.epsilon * np.ones(mean.shape[0]))

            multi_normal = multivariate_normal(mean=mean, cov=cov_matrix, allow_singular=True)

            L_t = multi_normal.logpdf(next_states)
            L = L + L_t

        return -1 * L
