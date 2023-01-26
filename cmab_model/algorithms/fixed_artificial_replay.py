import numbers
import numpy as np
from .algorithm import Algorithm
from .utils import common
from .fixed_discretization import FixedDiscretization

class FixedArtificialReplay(FixedDiscretization):

    def __init__(self, env, T, grid_discretization, dataset):
        super().__init__(env, T, grid_discretization)

        self.dataset = dataset

        # process historical dataset into parse-able dictionary
        self.history = {}
        for i in range(self.n_grid_points):
            for j in range(self.n_grid_points):
                for k in range(len(self.effort_levels)):
                    self.history[(i, j, k)] = []

        for h in range(len(self.dataset['points'])):
            loc    = self.dataset['points'][h]
            effort = self.dataset['efforts'][h].item()
            reward = self.dataset['rewards'][h]

            loc_discrete    = self.get_discrete_loc(loc)
            effort_discrete = self.get_discrete_effort(effort)
            dim = (loc_discrete[0], loc_discrete[1], effort_discrete)

            self.history[dim].append(reward)


    def get_dataset_size(self):
        ''' return number of reward observations remaining in historical data '''
        n_obs = 0
        for key in self.history:
            n_obs += len(self.history[key])
        return n_obs


    def dataset_contains(self, loc, effort):
        ''' check whether historical dataset contains samples of a given point

        returns False if does not contain
        otherwise returns the (loc, reward) sample observed in history '''

        loc_discrete    = self.get_discrete_loc(loc)
        effort_discrete = self.get_discrete_effort(effort)
        dim = (loc_discrete[0], loc_discrete[1], effort_discrete)

        if len(self.history[dim]) > 0:
            reward = self.history[dim].pop()
            return (loc_discrete, reward)

        return False


    def one_step(self, t):
        ''' returns flag: (bool) True if we took an online sample '''

        action = self.pick_action(self.regret_iterations) # pick a combinatorial action
        used_historical = False

        obs_reward = np.zeros(len(action))

        # look in historical data for reward samples from subarms
        for i, (loc, effort) in enumerate(action):
            check_data = self.dataset_contains(loc, effort)
            if check_data is not False:
                # update estimates for those entries in the dataset
                obs_reward[i] = check_data[1]
                self.update_single_obs((loc, effort), obs_reward[i])

                used_historical = True

        # take an online action only if no subarm is in historical dataset
        if not used_historical:
            # take online action and update estimates in the dataset
            for i, (loc, effort) in enumerate(action):
                obs_reward[i] = self.env.mean_reward(loc, effort)

            self.regret_iterations += 1
            self.regret += self.env.optimal - np.sum(obs_reward)
            self.update_obs(action, obs_reward)

        if t % 100 == 99:
            print(f'  {t}: historical dataset size {self.get_dataset_size()}, used_historical {used_historical}')

        return np.sum(obs_reward).item(), not used_historical


    def reset(self, dataset, true_means=None):
        ''' reset all quantities '''
        self.regret = 0
        self.regret_iterations = 0
        self.dataset = dataset

        if true_means is not None:
            self.true_means = true_means

    def history_use_percentage(self):
        raise NotImplementedError
