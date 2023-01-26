import numpy as np
from collections import Counter

def generate_dataset_two_arms(n, mean_arms, alpha):
    '''
    generate historical samples for setting with K=2 arms
    alpha - percentage of dataset coming from first arm
    '''
    dataset = {'0': [], '1': []}
    for _ in range(int(alpha*n)):
        dataset['0'].append(np.random.binomial(p=mean_arms[0], n=1))
    for _ in range(n - int(alpha*n)):
        dataset['1'].append(np.random.binomial(p=mean_arms[1], n=1))
    return dataset



class Environment:
    def __init__(self, K, N, T):
        '''
        MAB simulator for K-armed bandit with N historical samples

        returns dictionary: k -> array of reward from historical pulls
        '''
        self.K = K
        self.N = N
        self.T = T

        self.mean_arms = np.random.rand(K)

        # historical pulls
        self.history = {}

        historical_distrib = np.random.rand(K)  # distribution of pulls to each arm in history; not based on reward
        historical_distrib /= historical_distrib.sum()
        arms_pulled = np.random.choice(K, size=N, p=historical_distrib)

        for k in range(K):
            num_pulls = np.sum(arms_pulled == k)
            rewards = np.random.binomial(n=1, p=self.mean_arms[k], size=num_pulls)
            self.history[k] = rewards

        self.history_pos = np.zeros(K)  # track position of views in historical data

        self.online_data = {}
        for k in range(K):
            self.online_data[k] = np.random.binomial(n=1, p=self.mean_arms[k], size=self.T)

        self.online_pulls = []  # tracker for online pulls



    def pull_arm(self, k):
        ''' return observed reward from pulling arm k '''
        
        counter = Counter(e for e,_ in self.online_pulls)
        num_samples = dict(counter)[k]
        reward = self.online_data[k][num_samples+1]
        # reward = np.random.binomial(n=1, p=self.mean_arms[k])
        self.online_pulls.append((k, reward))


    def get_historical_pull(self, k):
        ''' returns None if we have exhausted all historical samples from arm k '''
        if self.history_pos[k] >= len(self.history[k]):
            return None

        reward = self.history[k][self.history_pos[k]]
        self.history_pos[k] += 1
        return reward

    def reset_algo(self):
        self.online_pulls = []
        self.history_pos = np.zeros(self.K)  # track position of views in historical data



    def reset_data(self, reset_reward_flag = False):
        if reset_reward_flag:
            self.mean_arms = np.random.rand(self.K)

            # historical pulls
            self.history = {}

            historical_distrib = np.random.rand(self.K)  # distribution of pulls to each arm in history; not based on reward
            historical_distrib /= historical_distrib.sum()
            arms_pulled = np.random.choice(self.K, size=self.N, p=historical_distrib)

            for k in range(self.K):
                num_pulls = np.sum(arms_pulled == k)
                rewards = np.random.binomial(n=1, p=self.mean_arms[k], size=num_pulls)
                self.history[k] = rewards

            self.history_pos = np.zeros(self.K)  # track position of views in historical data
        
        self.online_data = {}
        for k in range(self.K):
            self.online_data[k] = np.random.binomial(n=1, p=self.mean_arms[k], size=self.T)



class SimpleEnvironment(Environment):
    def __init__(self, delta, alpha, N):
        '''
        MAB simulator for 2-armed bandit with N historical samples

        returns dictionary: k -> array of reward from historical pulls
        '''
        self.K = 2
        self.N = N

        self.mean_arms = np.asarray([1/2, 1/2-delta])

        # historical pulls
        self.history = {}

        historical_distrib = np.asarray([alpha, 1-alpha])  # distribution of pulls to each arm in history; not based on reward
        arms_pulled = np.random.choice(self.K, size=N, p=historical_distrib)

        for k in range(self.K):
            num_pulls = np.sum(arms_pulled == k)
            rewards = np.random.binomial(n=1, p=self.mean_arms[k], size=num_pulls)
            self.history[k] = rewards

        self.history_pos = np.zeros(self.K)  # track position of views in historical data

        self.online_pulls = []  # tracker for online pulls
