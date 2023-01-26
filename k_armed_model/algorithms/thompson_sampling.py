import numpy as np
from .algorithm import Algorithm
from .common import conf_r, rd_argmax

class ThompsonSampling(Algorithm):
    '''
        Implementation of the TS algorithm which is agnostic of the dataset
    '''
    def __init__(self, true_means, dataset, K):
        self.dataset = dataset # save the dataset (although is ignored)
        self.K = K
        self.true_means = true_means

        self.alpha = np.ones(self.K) # initializes posterior alpha and beta
        self.beta = np.ones(self.K)


        self.regret = 0 # keeps track of its regret
        self.regret_iterations = 0 # and current index in the regret


    def reset(self, dataset):
        self.regret = 0 # resets the estimates back to zero
        self.regret_iterations = 0

        self.alpha = np.ones(self.K) # initializes posterior alpha and beta
        self.beta = np.ones(self.K)

    def update_obs(self, action, reward_obs):
        self.alpha[action] += reward_obs
        self.beta[action] += (1 - reward_obs)

    def pick_action(self, t):
        samples_list = [np.random.beta(self.alpha[i], self.beta[i]) for i in range(self.K)]
        return rd_argmax(samples_list) # returns the argmax
