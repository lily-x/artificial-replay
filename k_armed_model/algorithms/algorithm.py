import numpy as np

class Algorithm(object):
    '''
        Algorithm class providing generic implementation of the following functions:
        - reset (i.e. resetting estimates back to their original for a new iteration)
        - update_obs (update estimates based on observed reward)
        - selecting an action
        - one-step (performs a one-step loop of picking an arm, getting sample, calculating regret)

    '''
    def __init__(self, true_means, dataset, num_arms):
        self.dataset = dataset
        self.true_means = true_means
        self.regret = 0
        self.num_arms = num_arms
        self.regret_iterations = 0

    def reset(self, dataset):
        self.regret = 0
        self.regret_iterations = 0

    def update_config(self, config):
        raise NotImplementedError
        self.config = config

    def update_obs(self, action, reward):
        pass

    def pick_action(self, t):
        pass

    def one_step(self, t):
        '''
            One-Step loop for an online algorithm.
            Starts by picking an action according to the current timestep
            Updates the number of regret iterations, and updates regret based on the selected arm
            Obtains an observation sampled from the true distribution of the selected action
            Updates the observation
            Returns true (indicating that an online sample was taken and regret should be added)
        '''
        action = self.pick_action(self.regret_iterations)
        self.regret_iterations += 1
        self.regret += np.max(self.true_means) - self.true_means[action]
        obs = np.random.binomial(n=1,p=self.true_means[action])
        self.update_obs(action, obs)
        return True
