import numpy as np
from .algorithm import Algorithm
from .common import conf_r, rd_argmax
from .ids import IDS


class HistoricalIDS(IDS):
    '''
        Implementation of the IDS algorithm which is agnostic of the dataset
        param VIDS: boolean, if True choose arm which delta**2/v quantity. Default: False
    '''
    def __init__(self, true_means, dataset, K, VIDS=False):
        self.dataset = dataset # save the dataset (although is ignored)
        self.K = K
        self.true_means = true_means
        self.VIDS = VIDS

        self.M = 1000
        self.threshold = 0.99
        self.regret = 0 # keeps track of its regret
        self.regret_iterations = 0 # and current index in the regret

        self.flag = False
        self.alpha = np.ones(self.K) # initializes posterior alpha and beta
        self.beta = np.ones(self.K)
        for k in range(self.K):
            self.alpha[k] = 1 + np.sum(dataset[k])
            self.beta[k] = 1 + len(dataset[k]) - np.sum(dataset[k])

        self.thetas = np.array([np.random.beta(self.alpha[k], self.beta[k], self.M) for k in range(self.K)])
        self.Maap, self.p_a = np.zeros((self.K, self.K)), np.zeros(self.K)

    def reset(self, dataset):
        self.regret = 0 # resets the estimates back to zero
        self.regret_iterations = 0

        self.flag = False
        self.alpha = np.ones(self.K) # initializes posterior alpha and beta
        self.beta = np.ones(self.K)
        for k in range(self.K):
            self.alpha[k] = 1 + np.sum(dataset[k])
            self.beta[k] = 1 + len(dataset[k]) - np.sum(dataset[k])
        self.thetas = np.array([np.random.beta(self.alpha[k], self.beta[k], self.M) for k in range(self.K)])
        self.Maap, self.p_a = np.zeros((self.K, self.K)), np.zeros(self.K)
