''' regression-based approach '''

import sys
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .algorithm import Algorithm
from .utils import common


sys.path.append('../process_data')

# from process_data.learn_reward import NN, PAWSdataset
# from process_data.learn_reward import *
from learn_reward import NN, PAWSdataset


class Regression(Algorithm):
    '''
      > build NN (or generic model) to match available historical data; discretize R^2 space
      > solve knapsack with regressed values over grid
      > update NN (another training epoch/s) - how to tradeoff previous data with the new data samples
    '''

    def __init__(self, env, T, grid_discretization, dataset):
        super().__init__(env)

        self.T = T

        self.grid_discretization = grid_discretization
        self.n_grid_points       = len(self.grid_discretization)

        # Tensor of [i,j,effort] estimates for num visits and the mean reward
        self.reward_estimates = np.zeros([self.n_grid_points, self.n_grid_points, len(self.effort_levels)])
        self.num_visits       = np.zeros([self.n_grid_points, self.n_grid_points, len(self.effort_levels)])

        self.dataset_buffer  = []
        self.max_buffer_size = 256 #256 #64  # number of points we accumulate in buffer before retraining

        # get (x, y, effort) tuples of discretized grid, as input to NN regression
        self.discretized_grid_points = np.zeros((len(self.effort_levels), self.n_grid_points ** 2, 3))
        idx = 0
        for i, x in enumerate(self.grid_discretization):
            for j, y in enumerate(self.grid_discretization):
                for k, eff in enumerate(self.effort_levels):
                    self.discretized_grid_points[k, idx, :] = [x, y, eff]
                idx += 1
        self.discretized_grid_points = torch.tensor(self.discretized_grid_points).float()

        # ------------------------------------------
        # get historical data
        # ------------------------------------------

        self.dataset = dataset

        train_points = np.concatenate([dataset['points'], dataset['efforts']], axis=1)
        train_labels = dataset['rewards']

        self.nn_dataset = PAWSdataset(train_points, train_labels)

        # ------------------------------------------
        # set up NN
        # ------------------------------------------

        # NN hyperparameters
        input_size  = 3      # x, y, effort
        hidden_size = 10
        output_size = 1

        n_epochs      = 5
        batch_size    = 16

        learning_rate = 0.001
        momentum      = 0.9

        self.net = NN(input_size, hidden_size, output_size)

        self.criterion = nn.BCELoss()
        self.optimizer = optim.SGD(self.net.parameters(), lr=learning_rate, momentum=momentum)

        print('  setting up data loaders...')
        self.data_loader = torch.utils.data.DataLoader(
                self.nn_dataset,
                batch_size=batch_size, shuffle=True)

        # ------------------------------------------
        # train NN on historical data
        # ------------------------------------------
        # loop over our data iterator, feed the inputs to the network, and optimize

        for epoch in range(n_epochs):  # loop over dataset
            print(f'training, epoch {epoch} -----------------')

            running_loss = 0.0
            # for each minibatch
            for i, (inputs, labels) in enumerate(self.data_loader, 0):
                # forward
                outputs = self.net(inputs) #.squeeze()
                loss = self.criterion(outputs, labels)

                # backprop
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 100 == 99:    # print every 100 mini-batches
                    print(f'[{epoch + 1}, {i + 1:5d}]  loss: {running_loss / 100:.3f}')
                    running_loss = 0.0

        print('\nfinished training\n\n')



    def get_discrete_loc(self, loc):
        ''' gets the indices of location in grid discretization '''
        discrete_loc = [np.argmin(np.abs(self.grid_discretization - np.asarray(loc[0])), axis=0),
                        np.argmin(np.abs(self.grid_discretization - np.asarray(loc[1])), axis=0)]
        return np.array(discrete_loc)

    def get_discrete_effort(self, effort):
        ''' gets the index of chosen effort level '''
        return np.argmin(np.abs(self.effort_levels - np.asarray(effort)))

    def reset(self):
        raise NotImplementedError

    def update_single_obs(self, action, reward, check_retrain=True):
        ''' add observation to a buffer. if buffer is sufficiently long, then retrain NN

        check_retrain will be False if we're adding batch updates from a combinatorial action '''
        loc, effort = action
        self.dataset_buffer.append(((loc[0], loc[1], effort), reward))

        if check_retrain and len(self.dataset_buffer) >= self.max_buffer_size:
            self.retrain_net()


    def update_obs(self, actions, rewards):
        ''' update reward estimates for a combinatorial action '''
        for i, action in enumerate(actions):
            self.update_single_obs(action, rewards[i], check_retrain=False)

        # if buffer is sufficiently long, retrain NN
        if len(self.dataset_buffer) >= self.max_buffer_size:
            self.retrain_net()


    def retrain_net(self):
        ''' add all points from buffer and retrain net '''

        new_points = torch.tensor([tup[0] for tup in self.dataset_buffer])
        new_labels = torch.tensor([tup[1] for tup in self.dataset_buffer])
        self.nn_dataset.add_items(new_points, new_labels)

        print('updating and re-training')

        n_epochs = 5
        for epoch in range(n_epochs):  # loop over dataset
            print(f'   epoch {epoch} -----------------')

            running_loss = 0.0
            # for each minibatch
            for i, (inputs, labels) in enumerate(self.data_loader, 0):
                # forward
                outputs = self.net(inputs) #.squeeze()
                loss = self.criterion(outputs, labels)

                # backprop
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 100 == 99:    # print every 1000 mini-batches
                    print(f'      [{epoch + 1}, {i + 1:5d}]  loss: {running_loss / 100:.3f}')
                    running_loss = 0.0

        self.dataset_buffer = []


    def pick_action(self, t):
        ''' use regressor to estimate reward across discretized grid, then solve knapsack '''
        # get reward estimates from NN
        est_reward = np.zeros((self.n_grid_points**2, len(self.effort_levels)))
        for k, eff in enumerate(self.effort_levels):
            points = self.discretized_grid_points[k, :, :]
            est_reward[:, k] = self.net(points).detach().numpy().flatten()

        action, _ = common.solve_exploit(self.effort_levels, self.N, est_reward, self.budget)

        true_action = []
        for index, effort in action:
            first_loc  = int(np.floor(index / self.n_grid_points)) # gets out the actual locations from the grid
            second_loc = int(np.remainder(index, self.n_grid_points)) # to return
            true_action.append(((self.grid_discretization[first_loc], self.grid_discretization[second_loc]), effort))

        return true_action
