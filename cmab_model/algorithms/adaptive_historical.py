import numpy as np

from .algorithm import Algorithm
from .utils import common
from .utils import tree
from .adaptive_discretization import AdaptiveDiscretization

class AdaptiveHistorical(AdaptiveDiscretization):

    def __init__(self, env, T, epsilon, inherit_flag, dataset):
        super().__init__(env, T, epsilon, inherit_flag)

        self.dataset = dataset
        # Creates a list of discretizations of the location space for each possible effort level
        self.tree_list = []
        for j in range(len(self.effort_levels)):
            self.tree_list.append(tree.Tree(self.effort_levels[j], 2, self.max_depth))

        self.inherit_flag = inherit_flag

        for h in range(len(self.dataset['points'])):
            loc    = self.dataset['points'][h]
            reward = self.dataset['rewards'][h]
            effort = self.dataset['efforts'][h]

            effort_discrete = np.argmin(np.abs(self.effort_levels - np.asarray(effort))) # gets the index of chosen effort level
            current_tree = self.tree_list[effort_discrete] # tree for the specificly chosen effort level
            node, node_val = current_tree.get_active_ball(loc) # node that was selected for having highest reward

            ready_to_split = node.add_obs(loc, reward)
            if ready_to_split: # split a region
                current_tree.tree_split_node(node, inherit_flag) # get tree and split

    def reset(self):
        """ Resets the agent by setting all parameters back to zero """

        self.regret = 0 # resets the estimates back to zero
        self.regret_iterations = 0

        self.tree_list = []
        for j in range(len(self.effort_levels)):
            self.tree_list.append(tree.Tree(self.effort_levels[j], 2, self.max_depth))

        for h in range(len(self.dataset['points'])):
            loc    = self.dataset['points'][h]
            reward = self.dataset['rewards'][h]
            effort = self.dataset['efforts'][h]

            effort_discrete = np.argmin(np.abs(self.effort_levels - np.asarray(effort))) # gets the index of chosen effort level
            current_tree = self.tree_list[effort_discrete] # tree for the specificly chosen effort level
            node, node_val = current_tree.get_active_ball(loc) # node that was selected for having highest reward

            # update mean reward of node and number of visits
            t = node.num_visits
            node.mean_val = (t * node.mean_val + reward) / (t+1)
            node.num_visits += 1

            if node.splitting_condition(): # split a region
                # get tree and split
                current_tree.tree_split_node(node, self.inherit_flag)
