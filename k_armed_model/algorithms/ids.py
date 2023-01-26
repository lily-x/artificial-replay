import numpy as np
from .algorithm import Algorithm
from .common import conf_r, rd_argmax

class IDS(Algorithm):
    '''
        Implementation of the IDS algorithm which is agnostic of the dataset
        param VIDS: boolean, if True choose arm which delta**2/v quantity. Default: False
    '''
    def __init__(self, true_means, dataset, num_arms, VIDS=False):
        self.dataset = dataset # save the dataset (although is ignored)
        self.K = num_arms
        self.true_means = true_means
        self.VIDS = VIDS

        self.M = 1000
        self.threshold = 0.99
        self.regret = 0 # keeps track of its regret
        self.regret_iterations = 0 # and current index in the regret

        self.flag = False
        self.alpha = np.ones(self.K) # initializes posterior alpha and beta
        self.beta = np.ones(self.K)
        self.thetas = np.array([np.random.beta(self.alpha[k], self.beta[k], self.M) for k in range(self.K)])
        self.Maap, self.p_a = np.zeros((self.K, self.K)), np.zeros(self.K)


    def reset(self, dataset):
        self.regret = 0 # resets the estimates back to zero
        self.regret_iterations = 0

        self.flag = False
        self.alpha = np.ones(self.K) # initializes posterior alpha and beta
        self.beta = np.ones(self.K)
        self.thetas = np.array([np.random.beta(self.alpha[k], self.beta[k], self.M) for k in range(self.K)])
        self.Maap, self.p_a = np.zeros((self.K, self.K)), np.zeros(self.K)

    def update_obs(self, action, reward_obs):
        self.alpha[action] += reward_obs
        self.beta[action] += (1 - reward_obs)
        self.thetas[action] = np.random.beta(self.alpha[action], self.beta[action], self.M)


    '''
        https://github.com/DBaudry/Information_Directed_Sampling
    '''

    def IDSAction(self, delta, g):
        """
        Implementation of IDSAction algorithm as defined in Russo & Van Roy, p. 242
        :param delta: np.array, instantaneous regrets
        :param g: np.array, information gains
        :return: int, arm to pull
        """
        Q = np.zeros((self.K, self.K))
        IR = np.ones((self.K, self.K)) * np.inf
        q = np.linspace(0, 1, 1000)
        for a in range(self.K - 1):
            for ap in range(a + 1, self.K):
                if g[a] < 1e-6 or g[ap] < 1e-6:
                    return rd_argmax(-g)
                da, dap, ga, gap = delta[a], delta[ap], g[a], g[ap]
                qaap = q[rd_argmax(-(q * da + (1 - q) * dap) ** 2 / (q * ga + (1 - q) * gap))]
                IR[a, ap] = (qaap * (da - dap) + dap) ** 2 / (qaap * (ga - gap) + gap)
                Q[a, ap] = qaap
        amin = rd_argmax(-IR.reshape(self.K * self.K))
        a, ap = amin // self.K, amin % self.K
        b = np.random.binomial(1, Q[a, ap])
        arm = int(b * a + (1 - b) * ap)
        return arm

    def computeIDS(self):
        """
        Implementation of SampleIR (algorithm 4 in Russo & Van Roy, p. 242) applied for Bernoulli Bandits with
        beta prior. Here integrals are no more approximated using a grid on [0, 1] but in sampling thetas according
        to their respective posterior distributions.
        :param Maap: np.array, M(a|a') as defined in Russo & Van Roy's paper
        :param p_a: np.array, probability p* of choosing each arm supposing the latter is the optimal one
        :param thetas: np.array, posterior samples
        :param M: int, number of samples
        :param VIDS: boolean, if True choose arm which delta**2/v quantity
        :return: int, np.array, arm chose and p*
        """
        mu, theta_hat = np.mean(self.thetas, axis=1), np.argmax(self.thetas, axis=0)
        for a in range(self.K):
            mu[a] = np.mean(self.thetas[a])
            for ap in range(self.K):
                t = self.thetas[ap, np.where(theta_hat == a)]
                self.Maap[ap, a] = np.nan_to_num(np.mean(t))
                if ap == a:
                    self.p_a[a] = t.shape[1]/self.M
        if np.max(self.p_a) >= self.threshold:
            # Stop learning policy
            self.optimal_arm = np.argmax(self.p_a)
            arm = self.optimal_arm
        else:
            rho_star = sum([self.p_a[a] * self.Maap[a, a] for a in range(self.K)])
            delta = rho_star - mu
            if self.VIDS:
                v = np.array([sum([self.p_a[ap] * (self.Maap[a, ap] - mu[a]) ** 2 for ap in range(self.K)])
                              for a in range(self.K)])
                arm = rd_argmax(-delta ** 2 / v)
            else:
                g = np.array([sum([self.p_a[ap] * (self.Maap[a, ap] * np.log(self.Maap[a, ap]/mu[a]+1e-10) +
                                              (1-self.Maap[a, ap]) * np.log((1-self.Maap[a, ap])/(1-mu[a])+1e-10))
                                   for ap in range(self.K)]) for a in range(self.K)])
                arm = self.IDSAction(delta, g)
        return arm


    def pick_action(self, t):
        """
        Implementation of the Information Directed Sampling with approximation of integrals using MC sampling
        for Bernoulli Bandit Problems with beta prior
        :param T: int, time horizon
        :param M: int, number of samples. Default: 10 000
        :param VIDS: boolean, if True choose arm which delta**2/v quantity. Default: False
        :return: np.arrays, reward obtained by the policy and sequence of chosen arms
        """
        if not self.flag:
            if np.max(self.p_a) >= self.threshold:
                # Stop learning policy
                self.flag = True
                action = self.optimal_arm
            else:
                action = self.computeIDS()
        else:
            action = self.optimal_arm
        return action
