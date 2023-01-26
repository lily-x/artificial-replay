import numpy as np
from .algorithm import Algorithm
from .ucb import UCB
from .common import conf_r


class HistoricalUCB(UCB): # inherits the original UCB algorithm
    '''
        Implementation of the UCB algorithm which uses the historical dataset to start off
        the action selection frequencies and the means
    '''
    def __init__(self, T, true_means, dataset, K, MONOTONE_FLAG = False):
        self.horizon = T
        self.dataset = dataset
        self.true_means = true_means
        self.K = K
        self.MONOTONE_FLAG = MONOTONE_FLAG
        self.ucb = np.inf*np.ones(self.K)
            # same as original algorithm but updates the mean and selection frequency to that of the dataset
        self.means = np.asarray([max(0.0, np.mean(dataset[k])) for k in range(self.K)])
        self.selection = np.asarray([len(dataset[k]) for k in range(self.K)])

        self.regret = 0
        self.regret_iterations = 0

    def reset(self, dataset):
        self.regret = 0
        self.regret_iterations = 0
            # similar difference in the reset function
        self.means = np.asarray([max(0.0,np.mean(dataset[k])) for k in range(self.K)])
        self.selection = np.asarray([len(dataset[k]) for k in range(self.K)])
