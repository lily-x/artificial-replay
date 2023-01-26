import math
import numpy as np
from .algorithm import Algorithm
from .utils import common

DEBUG = False

class FixedDiscretization(Algorithm):
    """
    Fixed discretization algorithm  implemented for green-security enviroments
    with continuous domain and finite effort levels, with a discretization induced by l_inf metric

    Extra Attributes:
        T: (int) number of timesteps
        Mean_Reward: A function taking an action vector and returning the mean reward for that action
        Obs_Distr: A function taking an action vector and returning a vector of samples from the true underlying rewards
        Effort Levels: A vector of possible effort levels for the action space
    """


    def __init__(self, env, T, grid_discretization):
        super().__init__(env)

        self.T = T

        self.grid_discretization = grid_discretization
        self.n_grid_points       = len(self.grid_discretization)

        # Tensor of [i,j,effort] estimates for num visits and the mean reward
        self.cum_reward = np.zeros([self.n_grid_points, self.n_grid_points, len(self.effort_levels)])
        self.num_visits = np.zeros([self.n_grid_points, self.n_grid_points, len(self.effort_levels)])

        # track previous UCB to ensure monotone decreasing UCB
        self.prev_ucb = np.ones([self.n_grid_points, self.n_grid_points, len(self.effort_levels)])

    def get_discrete_loc(self, loc):
        ''' gets the indices of location in grid discretization '''
        discrete_loc = [np.argmin(np.abs(self.grid_discretization - np.asarray(loc[0])), axis=0),
                        np.argmin(np.abs(self.grid_discretization - np.asarray(loc[1])), axis=0)]
        return np.array(discrete_loc)

    def get_discrete_effort(self, effort):
        ''' gets the index of chosen effort level '''
        return np.argmin(np.abs(self.effort_levels - np.asarray(effort)))

    def reset(self):
        ''' Resets the agent by setting all parameters back to zero '''
        self.regret = 0
        self.regret_iterations = 0
        self.cum_reward = np.zeros([self.n_grid_points, self.n_grid_points, len(self.effort_levels)])
        self.num_visits = np.zeros([self.n_grid_points, self.n_grid_points, len(self.effort_levels)])
        self.prev_ucb   = np.ones([self.n_grid_points, self.n_grid_points, len(self.effort_levels)])

    def update_single_obs(self, action, reward):
        ''' update reward estimate from single observation for a (location, effort) pair '''
        if DEBUG: print(f'Observed data: {loc}, beta: {effort}, reward: {reward}')

        loc, effort = action

        # First find the (i,j) in discretization from the actual chosen location
        loc_discrete    = self.get_discrete_loc(loc)
        effort_discrete = self.get_discrete_effort(effort)

        dim = (loc_discrete[0], loc_discrete[1], effort_discrete)

        self.cum_reward[dim] += reward  # update reward estimate
        self.num_visits[dim] += 1 # update number of visits

        if DEBUG: print(f'Reward estimate: {self.cum_reward[dim] / self.num_visits[dim]}, num_visits: {self.num_visits[dim]}')


    def update_obs(self, actions, rewards):
        ''' update reward estimates for a combinatorial action '''
        if DEBUG: print(f'Updating observations')

        for i, action in enumerate(actions): # loop through each location + effort value that was passed in the action
            self.update_single_obs(action, rewards[i])


    def pick_action(self, t):
        ''' Formulates the ILP for the knapsack problem and solves for the optimal action vector '''

        region_ucbs = np.zeros([self.n_grid_points ** 2, len(self.effort_levels)])
        index = 0
        for i in range(self.n_grid_points):
            for j in range(self.n_grid_points):
                for k in range(len(self.effort_levels)):
                    if self.num_visits[i,j,k] > 0:
                        reward_estimate = self.cum_reward[i,j,k] / self.num_visits[i,j,k]
                        ucb_estimate = reward_estimate + common.conf_r(self.T, t, self.num_visits[i,j,k])
                        # ensure monotone
                        if ucb_estimate > self.prev_ucb[i,j,k]:
                            ucb_estimate = self.prev_ucb[i,j,k]
                        else:
                            self.prev_ucb[i,j,k] = ucb_estimate
                        region_ucbs[index, k] = ucb_estimate
                    else:
                        region_ucbs[index, k] = 1
                index += 1
        if DEBUG: print(f'Max UCB: {np.max(region_ucbs)}, Min UCB: {np.min(region_ucbs)}')

        # Solves for the optimal action based on these reward index values
        action, _ = common.solve_exploit(self.effort_levels, self.N, region_ucbs, self.budget)
        # Action is now returned to be a vector of length at most N of (loc index, effort level) pairs
        if DEBUG: print(f'Chosen action: {action}')

        true_action = []
        for index, effort in action:
            first_loc  = int(np.floor(index / self.n_grid_points)) # gets out the actual locations from the grid
            second_loc = int(np.remainder(index, self.n_grid_points)) # to return
            true_action.append(((self.grid_discretization[first_loc], self.grid_discretization[second_loc]), effort))
        if DEBUG: print(f'Final action: {true_action}')

        return true_action
