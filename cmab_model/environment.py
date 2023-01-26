''' Implement continuous-armed, combinatorial problem domains '''

import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import torch

from algorithms.utils.common import solve_exploit



def visualize_reward(reward_func, reward_str=''):
    """ visualize reward function """

    # create grid of points
    x = np.arange(0.0, 1.0, 0.05)
    y = np.arange(0.0, 1.0, 0.05)
    X, Y = np.meshgrid(x, y)

    # evaluate reward function across grid

    # for PWL reward
    efforts = np.ones(X.shape) * 0.5
    Z = reward_func(X, Y, efforts)
    Z = Z.reshape(X.shape)

    # visualize as 2D heatmap
    im = plt.imshow(Z, extent=(0, 1, 0, 1), origin='lower') #, cmap=plt.cm.RdBu)
    # add contour lines with labels
    plt.colorbar()
    plt.title(f'reward function: {reward_str}')
    plt.show()

    # visualize in 3D
    from mpl_toolkits.mplot3d import axes3d
    # from mpl_toolkits import mplot3d
    # import matplotlib as mpl
    # mpl.use('tkagg')

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    # fig = plt.figure() #(figsize=(13, 7))
    # ax = plt.axes(projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='coolwarm') # rstride=1, cstride=1, cmap='coolwarm', edgecolor='none')
    # ax.set_zlim(0, 1.0)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('reward')
    ax.set_title('reward function')
    fig.colorbar(surf) # shrink=0.5,, aspect=5  # add color bar indicating the PDF
    # ax.view_init(60, 35)
    plt.show()


class Environment:
    '''
    Environment base class
    Assumed to be two-dimensional
    '''
    def __init__(self, effort_levels, N, budget):
        self.dataset = {'points':  np.array([]).reshape(0, 2),
                        'efforts': np.array([]).reshape(0, 1),
                        'rewards': np.array([]).reshape(0, 1)}

        self.effort_levels = effort_levels
        self.N             = N
        self.budget        = budget

        self.optimal       = None
        # self.mean_reward   = None

        # dictionaries to quickly convert effort value to index
        self.effort_to_j_map = {}
        self.j_to_effort_map = {}
        for j, effort in enumerate(effort_levels):
            self.effort_to_j_map[effort] = j
            self.j_to_effort_map[j] = effort

    def effort_to_j(self, effort):
        return self.effort_to_j_map[effort]

    def j_to_effort(self, j):
        return self.j_to_effort_map[j]

    def get_point(self):
        pass

    def get_random_point(self):
        return np.random.rand(2)

    def get_dataset(self):
        return self.dataset

    def get_optimal(self, N, budget, grid_discretization):
        ''' optimal one-step mean reward '''
        # This is the same regardless of environment instantiation
        # (just depends on average expected reward)

        n_grid_points = len(grid_discretization)

        # Tensor of [i,j,effort] estimates for num visits and the mean reward
        region_ucbs = np.zeros([n_grid_points ** 2, len(self.effort_levels)])
        index = 0
        for i in range(n_grid_points):
            for j in range(n_grid_points):
                for k in range(len(self.effort_levels)):
                    # import pdb; pdb.set_trace()
                    points = np.array([[grid_discretization[i], grid_discretization[j]]])
                    effort = np.array([[self.effort_levels[k]]])
                    region_ucbs[index, k] = self.mean_reward(points, effort)
                index += 1

        # Solves for the optimal action based on these reward index values
        _, opt_val = solve_exploit(self.effort_levels, N, region_ucbs, budget)

        return opt_val


    def check_valid_comb_action(self, points, efforts):
        ''' ensure combinatorial action is valid:
        does not exceed boundary, and within budget '''
        eps_float = 1e-6 # allow for floating point error

        # not < 0, not > 1
        if np.any(points < -1e-6) or np.any(points > 1 + 1e-6):
            raise Exception(f'points invalid value {points}')
        points[np.where(points < 0), 0] = 0
        points[np.where(points > 1), 0] = 1

        # within budget
        assert np.sum(efforts) <= self.budget

        # not too many non-zero actions
        assert np.sum(efforts > 0) <= self.N

        return True


    def check_valid_action(self, point, effort):
        if isinstance(effort, np.ndarray): # if an array and not a single point/scalar
            assert len(point) == len(effort)
            self.check_valid_comb_action(point, effort)

        else:
            eps_float = 1e-6 # allow for floating point error

            # not < 0, not > 1
            if point[0] < -1e-6 or point[0] > 1 + 1e-6:
                raise Exception(f'point[0] invalid value {point[0]}')
            elif point[0] < 0: point = 0
            elif point[0] > 1: point = 1

            if point[1] < -1e-6 or point[1] > 1 + 1e-6:
                raise Exception(f'point[1] invalid value {point[1]}')
            elif point[1] < 0: point = 0
            elif point[1] > 1: point = 1

        return True


    def mean_reward(self, point, effort):
        ''' return mean reward for action
            point: [x, y]
            effort = value

            returns: float
        '''

        pass


    def sample_obs(self, points, efforts):
        ''' return vector of samples from true underlying reward '''
        pass

    def add_to_dataset(self, points, efforts, rewards):
        ''' add a single or multiple points to historical dataset '''
        if isinstance(rewards, np.ndarray): # if an array and not a single point/scalar
            assert len(points) == len(efforts) == len(rewards)

        self.dataset['points'] = np.vstack((self.dataset['points'], points))
        self.dataset['efforts'] = np.vstack((self.dataset['efforts'], efforts))
        self.dataset['rewards'] = np.vstack((self.dataset['rewards'], rewards))

    def get_historical_a_r(self, h):
        ''' h is index into historical dataset
        return (action, reward) tuple '''

        assert h < len(self.dataset['rewards'])
        return (self.dataset['points'][h, :], self.dataset['efforts'][h]), self.dataset['rewards'][h]

    def reset_algo(self):
        pass

    def reset_data(self, RESET_DATA_FLAG):
        pass



class RandomEnvironment(Environment):
    '''
    Dummy class with randomly generated data
    for quick testing
    '''
    def __init__(self, effort_levels, N, budget, n_historical=0):
        super().__init__(effort_levels, N, budget)

        # generate historical data
        self.n_historical = n_historical

        points = np.random.rand(n_historical, 2)
        effort = np.random.choice(effort_levels, size=n_historical, replace=True)
        labels = self.sample_obs(points, efforts)

        for h in range(n_historical):
            self.add_to_dataset(points[h], effort[h], labels[h])

        # randomly generate coefficients for 2D polynomial for expected rewards
        self.reward_coefs = np.random.uniform(-10, 10, 6)
        # np.random.rand(N) # randomly generate expected rewards

        c = self.reward_coefs
        reward_str = (f'${c[0]:.2f}x^2 ' + ('+' if c[1] >= 0 else '') +
                      f'{c[1]:.2f}xy'    + ('+' if c[2] >= 0 else '') +
                      f'{c[2]:.2f}y^2'   + ('+' if c[3] >= 0 else '') +
                      f'{c[3]:.2f}x'     + ('+' if c[4] >= 0 else '') +
                      f'{c[4]:.2f}y'     + ('+' if c[5] >= 0 else '') +
                      f'{c[5]:.2f}$')

        # visualize_reward(self.mean_reward, reward_str)

    def reward_fn(self, x, y, efforts):
        c = self.reward_coefs

        xy_space = c[0]*np.multiply(x,x) + c[1]*np.multiply(x,y) + c[2]*np.multiply(y,y) + c[3]*x + c[4]*y + c[5]
        return xy_space * efforts


    def mean_reward(self, points, efforts):
        ''' return mean reward for action '''
        assert self.check_valid_action(points, efforts)

        if np.isscalar(efforts):
            point, effort = points, efforts
            return self.reward_fn(point[0], point[1], effort).item()

        # array of points
        else:
            rewards = np.zeros(len(efforts))
            for i, point in enumerate(points):
                rewards[i] = self.reward_fn(point[0], point[1], efforts[i])
            return rewards


    def sample_obs(self, points, efforts):
        ''' return vector of samples from true underlying reward '''
        rewards = self.mean_reward(points, efforts)

        obs = np.zeros(len(effort))
        for i in range(len(effort)):
            obs[i] = np.random.choice([0, 1], p=[1-rewards[i], rewards[i]])

        return obs


class QuadraticEnvironment(Environment):
    '''
    Dummy class with randomly generated data
    for quick testing
    '''
    def __init__(self, effort_levels, N, budget, n_historical=0, frac_bad_arms=None):
        super().__init__(effort_levels, N, budget)

        # generate historical data
        self.n_historical = n_historical

        if frac_bad_arms is None:
            # sample historical points uniformly at random
            points = np.random.rand(n_historical, 2)

        else:
            # percentage of data to put in the bottom 20% of reward, between [0.0, 0.1] and [0.9, 1.0]
            assert 0 <= frac_bad_arms <= 1

            # generate fraction of bad points
            n_bad_arms    = n_historical * frac_bad_arms
            points_left   = np.random.rand(math.floor(n_bad_arms / 4), 2)  # left side
            points_right  = np.random.rand(math.floor(n_bad_arms / 4), 2)  # right side
            points_top    = np.random.rand(math.floor(n_bad_arms / 4), 2)  # top strip
            points_bottom = np.random.rand(math.floor(n_bad_arms / 4), 2)  # bottom strip

            points_left[:, 0]    = points_left[:, 0] * 0.1
            points_right[:, 0]   = points_right[:, 0] * 0.1 + 0.9
            points_top[:, 0]     = points_top[:, 0] * 0.1
            points_bottom[:, 1]  = points_bottom[:, 0] * 0.1 + 0.9

            points = np.vstack([points_left, points_right, points_top, points_bottom])

            # generate points for good region
            n_good_arms = n_historical - points.shape[0]  # compute this way in case of any rounding issues
            good_points = (np.random.rand(n_good_arms, 2) * 0.8) + 0.1

            points = np.vstack([points, good_points])


        effort = np.random.choice(effort_levels, size=n_historical, replace=True)
        labels = np.asarray([self.mean_reward(points[h], effort[h]) for h in range(n_historical)])
        for h in range(n_historical):

            self.add_to_dataset(points[h], effort[h], labels[h])

        # reward_str = f'quadratic'
        # visualize_reward(self.reward_fn, reward_str)
        # sys.exit(0)

    def set_optimal(self, grid_points):

        est_reward = np.zeros((self.n_grid_points**2, len(self.effort_levels)))
        for k, eff in enumerate(self.effort_levels):
            points = self.discretized_grid_points[k, :, :]
            est_reward[:, k] = self.mean_reward(points, efforts)

        self.optimal = solve_exploit(effort_levels, N, est_reward, budget)


    def reward_fn(self, x, y, effort):
        reward =  1 - np.square(x - 0.5) - np.square(y - 0.5)
        reward = np.clip(reward, 0, 1)
        return effort * reward


    def mean_reward(self, points, efforts):
        ''' return mean reward for action '''
        assert self.check_valid_action(points, efforts)

        # single point
        if np.isscalar(efforts):
            point, effort = points, efforts
            # prob = 1 - (point[0]-(math.pi/5))**2 - (point[1] - (math.pi/8))**2
            reward = self.reward_fn(point[0], point[1], effort)
            return reward.item()
            # return effort * np.random.binomial(n=1, p=prob)

        # array of points
        else:
            rewards = np.zeros(len(efforts))
            for i, point in enumerate(points):
                # prob = 1 - (point[0]-(math.pi/5))**2 - (point[1] - (math.pi/8))**2
                # rewards[i] = efforts[i] * np.random.binomial(n=1, p=prob)
                rewards[i] = self.reward_fn(point[0], point[1], efforts[i])
            return rewards

        # put reward on non-rational point so no advantage to adaptive discretization algorithm

    def sample_obs(self, points, efforts):
        ''' return vector of samples from true underlying reward '''
        probs = self.mean_reward(points, efforts)

        # single point
        if np.isscalar(efforts):
            return efforts * np.random.binomial(n=1, p=probs)

        # array of points
        else:
            obs = np.zeros(len(efforts))
            for i, prob in enumerate(probs):
                obs[i] = efforts[i] * np.random.binomial(n=1, p=prob)
            return obs


class PWLEnvironment(QuadraticEnvironment):
    '''
    Piecewise linear reward function to evaluate coupling
    '''
    def __init__(self, effort_levels, N, budget, n_historical=0):
        super().__init__(effort_levels, N, budget, n_historical=n_historical)

        # generate historical data
        self.n_historical = n_historical

        points = np.random.rand(n_historical, 2)
        effort = np.random.choice(effort_levels, size=n_historical, replace=True)
        labels = np.asarray([self.mean_reward(points[h], effort[h]) for h in range(n_historical)])

        for h in range(n_historical):
            self.add_to_dataset(points[h], effort[h], labels[h])

        reward_str = f'pwl'
        visualize_reward(self.reward_fn, reward_str)
        sys.exit(0)

    def reward_fn(self, x, y, effort):
        ''' very simple plane '''
        reward = x/2 + y/2
        return reward * effort


class UgandaEnvironment(Environment):
    '''
    Environment class with real data
    from Murchison Falls National Park, Uganda
    '''

    def __init__(self, reward_fn, effort_levels, N, budget, historical_data=None, n_historical=None):
        super().__init__(effort_levels, N, budget)

        self.reward_fn = reward_fn

        # no historical data points provided
        if historical_data is None:
            assert n_historical is not None

            # add random points to dataset uniformly at random
            points, efforts, rewards = self.get_sample_points_randomly(n_historical)
            efforts = efforts.reshape(-1, 1)
            self.add_to_dataset(points, efforts, rewards)

        # historical data points provided
        else:
            print('add historical data')
            self.n_historical = len(historical_data['points'])

            points  = historical_data['points']
            efforts = historical_data['efforts']
            rewards = self.sample_obs(points, efforts)

            efforts = efforts.reshape(-1, 1)

            self.add_to_dataset(points, efforts, rewards)

        # reward_str = 'uganda_reward'
        # visualize_reward(self.reward_fn, reward_str)
        # sys.exit(0)


    def mean_reward(self, points, efforts):
        ''' return mean reward for action '''
        assert self.check_valid_action(points, efforts)

        if torch.is_tensor(points): # isinstance(points, np.ndarray)
            inputs  = torch.cat((points, efforts), axis=1)
            with torch.no_grad():
                rewards = self.reward_fn(inputs).numpy()

        elif np.isscalar(efforts):
            inputs = torch.tensor([[points[0], points[1], efforts]]).float()
            with torch.no_grad():
                rewards = self.reward_fn(inputs).numpy().item()

        else:
            inputs  = torch.tensor(np.concatenate([points, efforts], axis=1)).float()
            with torch.no_grad():
                rewards = self.reward_fn(inputs).numpy()

        return rewards

    def sample_obs(self, points, efforts):
        '''
        Given an input list of point locations, sample a reward observation for each point
        '''
        if isinstance(points, np.ndarray):
            points  = torch.from_numpy(points).float()
            efforts = torch.from_numpy(efforts).float()

        print(points.shape)
        print(efforts.shape)

        inputs = torch.cat((points, efforts), axis=1)
        with torch.no_grad():
            rewards = self.reward_fn(inputs).numpy().squeeze()

        obs = np.zeros((len(efforts), 1))
        for i in range(len(efforts)):
            obs[i] = np.random.choice([0, 1], p=[1-rewards[i], rewards[i]])

        return obs


    def get_sample_points_randomly(self, n_points):
        ''' randomly select sample points '''
        points  = np.random.rand(n_points, 2)   # pick random (x, y) location
        efforts = np.random.rand(n_points, 1)   # pick random effort

        rewards = self.sample_obs(points, efforts)

        return points, efforts, rewards


    def true_reward_from_points(self, points, efforts=None):
        '''
        Given an input list of point locations, get true reward for each point

        If effort is not given, then randomly select effort
        '''
        # if efforts is None: efforts = np.random.rand(n_points)

        if isinstance(points, np.ndarray):
            points  = torch.from_numpy(points).float()
            efforts = torch.from_numpy(efforts).float()

        inputs = torch.cat((points, efforts), axis=1)
        with torch.no_grad():
            rewards = self.reward_fn(inputs).numpy()

        return rewards
