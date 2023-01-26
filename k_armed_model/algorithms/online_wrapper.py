import numpy as np
from .algorithm import Algorithm

class OnlineWrapper(Algorithm):

    def __init__(self, true_means, dataset, N, K, algorithm):
        self.dataset = dataset
        self.N = N  # num historical pulls
        self.K = K
        self.algorithm = algorithm
        self.true_means = true_means

        self.algorithm.reset(dataset)

        self.dataset_index = np.zeros(self.K) # keeps track of which datapoints have been used in the historical dataset
        self.regret = 0 # tracker for cumulative regret
        self.regret_iterations = 0 # tracker for number of online calls when sampling an arm


    def reset(self, dataset, true_means=None):
        self.regret = 0 # resets all of the quantities
        self.regret_iterations = 0
        self.dataset = dataset
        self.dataset_index = np.zeros(self.K)
        self.algorithm.reset(dataset)

        if true_means is not None:
            self.true_means = true_means

    def update_obs(self, arm, reward_obs):
        self.algorithm.update_obs(arm, reward_obs) # appeals to the sub-algorithm to update based on the observation

    def one_step(self, t):
        flag = False
        arm = self.algorithm.pick_action(self.regret_iterations) # calls its sub algorithm to pick an arm

        # Check if we have value in dataset that we can use and feed back to algorithm
        if self.dataset_index[arm] < len(self.dataset[arm]):
            obs = self.dataset[arm][int(self.dataset_index[arm])]# gets a value from the dataset
            self.dataset_index[arm] += 1 # updates index within the dataset
        else:
        # Otherwise, take online sample:
            obs = np.random.binomial(p=self.true_means[arm], n=1)
            self.regret += np.max(self.true_means) - self.true_means[arm]
            self.regret_iterations += 1
            self.algorithm.regret_iterations += 1
            flag = True
        self.update_obs(arm, obs)
        return flag

    def used_all_history(self):
        ''' returns whether we've gone through all historical data '''
        for k in range(self.K):
            if self.dataset_index[k] < len(self.dataset[k]):
                return False

        return True

    def history_use_percentage(self):
        return np.sum(self.dataset_index) / self.N
