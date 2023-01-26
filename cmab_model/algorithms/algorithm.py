import numpy as np

class Algorithm(object):
    '''
        Algorithm class providing generic implementation of the following functions:
        - reset (i.e. resetting estimates back to their original for a new iteration)
        - update_obs (update estimates based on observed reward)
        - selecting an action
        - one-step (performs a one-step loop of picking an arm, getting sample, calculating regret)

        Takes in Environment class that contains
        - dataset:       historical dataset
        - mean_reward:   function taking an action vector and returning the mean reward for that action
        - obs_distr:     function taking an action vector and returning a vector of samples from the true underlying rewards
        - optimal:       optimal one-step mean reward
        - effort_levels: vector of possible effort levels for the action space
    '''
    def __init__(self, env):
        '''
            N: Max possible locations to visit
            Dataset: A historical dataset

            Budget: Budget for effort
        '''
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
        '''
            Action: List of type [ (loc_1, effort_1), (loc_2, effort_2), ...]
            Reward: List of type [r_1, r_2, ...]
            where each entry is sampled according to the true underlyling distribution for that particular point
        '''
        pass

    def pick_action(self, t):
        '''
            Has the algorithm "pick" an action, returns:
            act = [ (loc_1, effort_1), (loc_2, effort_2), ...]
            where loc_i = [x_i, y_i], and effort_1 is the actual value
        '''
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

        assert len(action) >= 1  # action must pull at least one arm
        self.regret_iterations += 1

        obs_reward = []
        tot_reward = 0
        for (loc, effort) in action:
            reward = self.env.mean_reward(loc, effort)
            obs_reward.append(reward)
            tot_reward += reward

        self.regret += self.env.optimal - np.sum(tot_reward)
        self.update_obs(action, obs_reward)

        return tot_reward, True
