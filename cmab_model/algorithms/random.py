import numpy as np

class RandomAlgorithm(object):
    ''' random action in CMAB-CRA setting '''
    def __init__(self, env):
        # saves parameters as part of the object
        self.env     = env
        self.N       = env.N
        self.budget  = env.budget
        self.effort_levels = env.effort_levels

        # track number of iterations and cumulative regret
        self.regret = 0
        # self.reward = []
        self.regret_iterations = 0

    def reset(self, dataset):
        ''' Reset regret and number of iterations back to zero '''
        self.regret = 0
        self.regret_iterations = 0

        self.env.reset()

    def update_obs(self, action, reward):
        pass

    def pick_action(self, t):
        raise NotImplementedError
