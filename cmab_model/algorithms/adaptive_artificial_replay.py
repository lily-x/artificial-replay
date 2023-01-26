import numbers
import numpy as np
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('tkagg')

from .algorithm import Algorithm
from .utils import common
from .utils import common
from .utils import tree
from .adaptive_discretization import AdaptiveDiscretization, update_single_obs


class AdaptiveArtificialReplay(AdaptiveDiscretization):

    def __init__(self, env, T, epsilon, inherit_flag, dataset):
        super().__init__(env, T, epsilon, inherit_flag)

        self.dataset = dataset
        self.n_historical = len(self.dataset['points'])

        # initialize historical dataset

        # list of discretizations of the location space for each possible effort level
        self.dataset_tree_list = []
        for j in range(len(env.effort_levels)):
            self.dataset_tree_list.append(tree.Tree(env.effort_levels[j], 2, self.max_depth))

        for h in range(self.n_historical):
            point  = dataset['points'][h]
            effort = dataset['efforts'][h]
            reward = dataset['rewards'][h]

            action = (point, effort)
            inherit_flag = False
            update_single_obs(self.dataset_tree_list, action, reward, self.effort_levels, inherit_flag)

            # print(f'  {h:3d}   add ({point.round(2)}, {effort.item()}, {reward.item():.2f})   historical dataset size', self.get_dataset_size(), 'tree size', self.get_dataset_tree_size())
            # print(f'  {i:3d}   add ({point.round(2)}, {effort.item()}, {reward.item():.2f})   historical dataset tree size', self.get_dataset_tree_size())

            # if h % 100 == 99:
            if h % 500 == 499:
                # title = f'{h+1}_of_{self.n_historical}'
                self.visualize_historical_dataset(f'{h+1}')
        sys.exit(0)


    def visualize_historical_dataset(self, title=''):
        plt.figure(1, figsize=(12, 3.7))
        # plt.clf() # clear figure

        # plt.figure(figsize=(12, 3.7))
        fig, axs = plt.subplots(1, len(self.effort_levels), figsize=(12, 3.7), num=1)
        for j, eff in enumerate(self.effort_levels):
            self.dataset_tree_list[j].visualize_split(axs[j], f'effort {eff}')

        # plt.suptitle(title)
        # plt.show()
        plt.savefig(f'historical_dataset_{title}', bbox_inches='tight')

        plt.close()


    def reset(self):
        raise NotImplementedError


    def dataset_effort_empty(self, effort):
        ''' whether tree at a given effort level is empty '''
        j = self.env.effort_to_j(effort)  # convert effort value to indiex

        return self.dataset_tree_list[j].is_empty()


    def dataset_empty(self):
        ''' whether every tree at all effort levels are empty '''
        for j in range(len(effort_levels)):
            if not self.dataset_tree_list[j].is_empty():
                return False
        return True


    def dataset_contains(self, loc, effort):
        ''' check whether historical dataset contains samples of a given point

        returns False if does not contain
        otherwise returns the reward sample observed in history '''

        # tree at given effort level is empty
        if self.dataset_effort_empty(effort):
            return False

        j = self.env.effort_to_j(effort)
        relevant_tree = self.dataset_tree_list[j]

        node, node_val = relevant_tree.get_active_ball(loc)

        # remove and return one historical observation
        return node.get_historical_obs()


    def get_dataset_size(self):
        ''' return the number of historical observations in the current dataset

        returns a dict with # stored observations per effort level and total # '''
        n_obs = self.get_dataset_size_per_eff()
        total_n_obs = 0
        for eff in self.effort_levels:
            total_n_obs += n_obs[eff]
        return total_n_obs


    def get_dataset_size_per_eff(self):
        ''' return the number of historical observations in the current dataset

        returns a dict with # stored observations per effort level and total # '''
        total_n_obs = 0
        n_obs = {}
        for j, eff in enumerate(self.effort_levels):
            tree = self.dataset_tree_list[j]
            n_obs[eff] = tree.get_n_observations()
            total_n_obs += n_obs[eff]

        return n_obs



    def get_dataset_tree_size(self):
        ''' return the number of nodes in the dataset tree (summed across effort levels) '''
        n_nodes = 0
        for j, eff in enumerate(self.effort_levels):
            tree = self.dataset_tree_list[j]
            n_nodes += tree.get_tree_size()
        return n_nodes


    def one_step(self, t):
        '''
        pick and execute an action

        only take online action IF AND ONLY IF no subarm is in historical dataset

        returns flag: (bool) True if we took an online sample '''

        action = self.pick_action(self.regret_iterations) # picks an action
        used_historical = False # check whether to take online action

        obs_reward = np.zeros(len(action))

        for i, (loc, effort) in enumerate(action):
            # check whether we have value in dataset that we can use and feed back to algorithm
            check_data = self.dataset_contains(loc, effort)
            if check_data is not False:
                # update estimates for those entries in the dataset
                obs_reward[i] = check_data[1]
                self.update_single_obs(loc, effort, obs_reward[i])

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
            print(f'  {t}: historical dataset size {self.get_dataset_size_per_eff()}, used_historical {used_historical}')

        # if t % 10 == 9:
        #     self.visualize_historical_dataset(f't = {t}')
        # print('regret iterations', self.regret_iterations)

        return np.sum(obs_reward), not used_historical
