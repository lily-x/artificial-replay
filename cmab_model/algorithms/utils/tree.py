import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from bounds_utils import bounds_contains, split_bounds

class Node():
    '''
    Node representing an l-infinity ball in R^2, that points
    to sub-balls (defined via node children).
    Stores a value for the mean reward, a number of visits.

    This class is used to represent (and store data about)
    a tuple (location, effort).
    Attributes:
        bounds : numpy.ndarray
            Bounds of each dimension [ [x0, y0], [x1, y1], ..., [xd, yd] ],
            representing the cartesian product in R^d:
            [x0, y0] X [x1, y1] X ... X [xd, yd]
        depth: int
            Node depth, root is at depth 0.
        mean_val : double, default: 0
            Initial node mean value estimate
        num_visits : int, default = 0
            Number of visits to the node.
        children: (list)
            List of children for the node
    '''

    def __init__(self, effort, bounds, depth, max_depth, mean_val, num_visits):
        '''
        Initialization for a node.
        Args:
            bounds : numpy.ndarray
                Bounds of each dimension [ [x0, y0], [x1, y1], ..., [xd, yd] ],
                representing the cartesian product in R^d:
                [x0, y0] X [x1, y1] X ... X [xd, yd]
            depth: int
                Node depth, root is at depth 0.
            mean_val : double, default: 0
                Initial node Q value
            num_visits : int, default = 0
                Number of visits to the node.
            effort : float
                Value of effort level
        '''

        self.dim = len(bounds)  # updates dimension of the box

        self.radius = (bounds[:, 1] - bounds[:, 0]).max() / 2.0  # calculates its radius

        assert self.radius > 0.0, "Error: radius of a ball should be strictly positive"
        self.effort     = effort
        self.bounds     = bounds
        self.depth      = depth
        self.max_depth  = max_depth
        self.mean_val   = mean_val
        self.num_visits = num_visits
        self.prev_ucb   = 1.  # need to track previous UCB estimate to ensure monotone decreasing

        self.children = []  # list of children for a box

        # historical observations of (loc, reward) tuples
        # need to store these manually for implementing historical dataset
        self.obs = []

    def get_n_obs(self):
        ''' returns number of historical observations stored within this tree '''
        n_obs = len(self.obs)
        for child in self.children:
            n_obs += child.get_n_obs()
        return n_obs

    def get_tree_size(self):
        ''' returns number of nodes comprising this tree '''
        size = 1
        for child in self.children:
            size += child.get_tree_size()
        return size

    def get_all_bounds(self):
        ''' recursively get list of all bounds of subtree
        used for visualization '''
        if self.is_leaf(): return [self.bounds]

        all_bounds = []
        for child in self.children:
            all_bounds.append(child.get_all_bounds())
        all_bounds = [item for items in all_bounds for item in items]
        return all_bounds

    def get_all_obs(self):
        ''' recursively get list of all observations in subtree
        used for visualization '''
        if self.is_leaf(): return self.obs

        all_obs = []
        for child in self.children:
            all_obs.append(child.get_all_obs())
        all_obs = [item for items in all_obs for item in items]
        return all_obs


    def splitting_condition(self):
        ''' when to split node in tree '''
        return self.num_visits >= 2**(2*self.depth)


    def add_obs(self, loc, reward):
        ''' add an observation to historical observations dataset '''
        # update mean reward of node and number of visits
        t = self.num_visits
        self.mean_val = (t * self.mean_val + reward) / (t+1)
        self.num_visits += 1

        self.obs.append((loc, reward))

        return self.splitting_condition()  # ready to split?

    def is_empty(self):
        ''' whether there are no historical observations within this subtree '''
        if self.is_leaf():
            return len(self.obs) == 0
        else:
            return False

    def point_in_bounds(self, p):
        ''' whether a given point falls within the bounds of this subtree '''
        assert self.dim == 2 # for now, dimensions > 2 not implemented

        bounds = self.bounds
        return bounds[0][0] <= p[0] <= bounds[0][1] and bounds[1][0] <= p[1] <= bounds[1][1]

    def is_leaf(self):
        return len(self.children) == 0

    def contains(self, state):
        return bounds_contains(self.bounds, state)

    def sample_point(self):
        ''' sample a random point within the region '''
        if self.dim != 2: raise NotImplementedError

        bounds = self.bounds
        point = np.random.rand(self.dim)
        point[0] = point[0] * (bounds[0][1] - bounds[0][0]) + bounds[0][0]
        point[1] = point[1] * (bounds[1][1] - bounds[1][0]) + bounds[1][0]
        return point

    def get_historical_obs(self):
        ''' remove and return one historical observation '''
        if self.is_empty():
            return False
        else:
            return self.obs.pop()


    def split_node(self, inherit_flag=True, inherit_value=1):
        '''
        Splits a node across all of the dimensions
        Args:
            inherit_flag:  (bool) boolean of whether to intialize estimates of children to that of parent
                >> in practice, we always want this to be True
            inherit_value: default mean_val to inherit if inherit_flag is false
        '''

        child_bounds = split_bounds(self.bounds)  # splits the bounds of the box

        for bounds in child_bounds:  # adds a child for each of the split bounds, initializing their values appropriately
            if inherit_flag:
                self.children.append(
                    Node(self.effort, bounds, self.depth+1, self.max_depth, self.mean_val, self.num_visits))
            else:
                self.children.append(
                    Node(self.effort, bounds, self.depth+1, self.max_depth, inherit_value, 0))


        # split out observations across children
        for idx, (loc, reward) in enumerate(self.obs):
            flag = False
            for child in self.children:
                if child.point_in_bounds(loc):
                    flag = True
                    child.obs.append((loc, reward))
                    break
            # assert that all observations have been re-allocated to some child
            if not flag:
                for child in self.children:
                    print('  ', child.bounds.flatten().round(2))
                raise Exception(f'should not be here... all points should be allocated to children. location {loc.round(2)}')


        self.obs = []

        return self.children





class Tree():
    '''
    Tree representing a collection of l-infinity balls (boxes) in R^d, that points
    to sub-balls (defined via node children).
    Stores a hierarchical collections of nodes with value for the q_estimate, a number of visits, and
    Attributes:
        dim : int
            Dimension of the space of R^d.
        head: (Node)
            Pointer to the first node in the hierarchical partition
    '''

    # Defines a tree by the number of steps for the initialization
    def __init__(self, effort, dim, max_depth):
        ''' Initializes tree '''

        self.dim = dim
        self.effort = effort
        self.max_depth = max_depth

        bounds = np.asarray([[0.0, 1.0] for _ in range(dim)])
        self.head = Node(effort, bounds, 0, max_depth, 0, 0)
        self.leaves = [self.head]

    def get_n_observations(self):
        ''' returns number of historical observations stored within this tree '''
        return self.head.get_n_obs()

    def get_tree_size(self):
        ''' returns number of historical observations stored within this tree '''
        return self.head.get_tree_size()

    def is_empty(self):
        ''' Returns whether tree is empty '''
        return self.head.is_empty()

    def get_head(self):
        ''' Returns the head of the tree '''
        return self.head

    def get_max(self, node=None, root=True):
        ''' Returns the maximum reward value across all nodes in the tree '''
        if root:
            node = self.head
        if len(node.children) == 0:
            return node.mean_val
        else:
            return np.max([self.get_max(child, False) for child in node.children])

    def get_min(self, node=None, root=True):
        ''' Returns the minimum reward value across all nodes in the tree '''
        if root:
            node = self.head
        if len(node.children) == 0:
            return node.mean_val
        else:
            return np.min([self.get_min(child, False) for child in node.children])


    def visualize_split(self, ax, title=''):

        all_bounds = self.head.get_all_bounds()
        all_obs    = self.head.get_all_obs()


        for loc, reward in all_obs:
            ax.plot(loc[0], loc[1], marker='o',
            markersize=3, markeredgecolor='none', markerfacecolor='royalblue',
            alpha=0.5)

        for bounds in all_bounds:
            x_size = bounds[0][1] - bounds[0][0]
            y_size = bounds[1][1] - bounds[1][0]
            x_loc  = bounds[0][0]
            y_loc  = bounds[1][0]

            # print('add rect', bounds.flatten(), x_size, y_size, x_loc, y_loc)

            ax.add_patch(Rectangle((x_loc, y_loc), x_size, y_size,
                edgecolor='darkred', facecolor='none', lw=.5, zorder=10))

        # ax.set_title(title)

        ax.set_aspect('equal', 'box')
        # fig.tight_layout()

    def get_active_ball(self, state, node=None, root=True):
        '''
            Gets the active ball for a given state, i.e., the node in the tree containing the state with the largest Q Value
            Args:
                state: np.array corresponding to a state
                node: Current node we are searching for max value over children of
                root: indicator that we are at the root, and should start calculating from the head of the tree
            Returns:
                best_node: the node corresponding to the largest q value containing the state
                best_mean_val: the value of the best node
        '''

        if root:
            node = self.head

        if len(node.children) == 0:
            return node, node.mean_val

        else:
            best_mean_val = (-1)*np.inf

            for child in node.children:
                if child.contains(state):
                    nn, nn_mean_val = self.get_active_ball(state, child, False)
                    if nn_mean_val >= best_mean_val:
                        best_node, best_mean_val = nn, nn_mean_val

            return best_node, best_mean_val



    def tree_split_node(self, node, inherit_flag=True, value=1):
        self.leaves.remove(node)
        children = node.split_node(inherit_flag, value)
        self.leaves = self.leaves + children
