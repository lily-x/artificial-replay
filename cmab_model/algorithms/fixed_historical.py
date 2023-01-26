import numbers
import numpy as np
from .algorithm import Algorithm
from .utils import common
from .fixed_discretization import FixedDiscretization

DEBUG = False

class FixedHistorical(FixedDiscretization):

    def __init__(self, env, T, grid_discretization, dataset):
        super().__init__(env, T, grid_discretization)

        self.dataset = dataset

        for h in range(len(self.dataset['points'])):
            loc    = self.dataset['points'][h]
            reward = self.dataset['rewards'][h]
            effort = self.dataset['efforts'][h]

            loc_discrete    = self.get_discrete_loc(loc)
            effort_discrete = self.get_discrete_effort(effort)

            dim = (loc_discrete[0], loc_discrete[1], effort_discrete) # combine the indexes together to index into matrix

            self.cum_reward[dim] += reward # update reward estimate
            self.num_visits[dim] += 1 # update number of visits


    def reset(self):
        super().reset()

        for i in range(len(self.dataset['points'])):
            loc    = self.dataset['points'][i]
            reward = self.dataset['rewards'][i]
            effort = self.dataset['efforts'][i]

            loc_discrete    = self.get_discrete_loc(loc)
            effort_discrete = self.get_discrete_effort(effort)

            dim = (loc_discrete[0], loc_discrete[1], effort_discrete) # combine the indexes together to index into matrix

            self.cum_reward[dim] += reward
            self.num_visits[dim] += 1
