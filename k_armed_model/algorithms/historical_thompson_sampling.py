import numpy as np
from .algorithm import Algorithm
from .thompson_sampling import ThompsonSampling
from .common import conf_r


class HistoricalThompsonSampling(ThompsonSampling): # inherits the original TS algorithm
    '''
        Implementation of the TS algorithm which uses the historical dataset to start off
        the alpha beta posterior parameters
    '''
    def __init__(self, true_means, dataset, num_arms):
        self.dataset = dataset
        self.true_means = true_means
        self.K = num_arms
            # same as original algorithm but updates the alpha and beta parameters from the dataset
        self.alpha = np.zeros(self.K)
        self.beta = np.zeros(self.K)
        for k in range(self.K):
            self.alpha[k] = 1 + np.sum(dataset[k])
            self.beta[k] = 1 + len(dataset[k]) - np.sum(dataset[k])
        self.regret = 0
        self.regret_iterations = 0

    def reset(self, dataset):
        self.regret = 0
        self.regret_iterations = 0
            # similar difference in the reset function
        for k in range(self.K):
            self.alpha[k] = 1 + np.sum(dataset[k])
            self.beta[k] = 1 + len(dataset[k]) - np.sum(dataset[k])
