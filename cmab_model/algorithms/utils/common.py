import numpy as np
import random as rd
import gurobipy as gp
from gurobipy import GRB


def conf_r(T, t, n_pulls):
    """ compute confidence radius """
    # return np.sqrt(2*np.log(1+T) / n_pulls)
    eps = .1
    return np.sqrt(-np.log(eps) / (2. * max(1, n_pulls)))

def rd_argmax(vector):
    """
    Compute random among eligible maximum indices
    :param vector: np.array
    :return: int, random index among eligible maximum indices
    """
    m = np.amax(vector)
    indices = np.nonzero(vector == m)[0]
    return rd.choice(indices)


def solve_exploit(effort_levels, N, reward_estimates, budget):
    """ given historical data, solve for optimal arm pull

    B = vector of (index, effort) level pairs
    B = [(0,0),(1, .1), (2, .25), (3,.5)] eg.
    N = maximum number of locations that we can visit (potentially fewer than number regions)
    reward_estimates = a matrix of [loc, effort] pairs, so second dimension has length of B - should be optimistic estimates
    budget = budget limit

    Returns:
        a vector of length at most N of (loc index, effort level) pairs
    """

    n_effort = len(effort_levels)
    n_regions = reward_estimates.shape[0] # len(reward_estimates) #reward_estimates.shape[0] # gets out the number of locations
    # print(f'Solving IP for: {n_effort} effort levels and {n_locations} locations')
    model = gp.Model('exploit')

    # copy mu from inputs
    mu = np.copy(reward_estimates)
    # silence output
    model.setParam('OutputFlag', 0)

    # decision variables for each [loc, effort] pair
    x = [[model.addVar(vtype=GRB.BINARY, name='x_{}_{}'.format(i, j))
            for j in range(n_effort)] for i in range(n_regions)]

    model.setObjective(gp.quicksum([x[i][j] * mu[i][j]
                for i in range(n_regions) for j in range(n_effort)]),
                GRB.MAXIMIZE)

    model.addConstrs((gp.quicksum(x[i][j] for j in range(n_effort)) <= 1
                for i in range(n_regions)), 'one_per_target') # at most pull one arm per location

    model.addConstr(gp.quicksum([x[i][j] * effort_levels[j]
                for i in range(n_regions) for j in range(n_effort)]) <= budget, 'budget')  # stay in budget

    model.addConstr(gp.quicksum([x[i][j]
                for i in range(n_regions) for j in range(n_effort)]) <= N, 'capacity')
        # can only pick at most N locations

    model.optimize()

    if model.status != GRB.OPTIMAL:
        raise Exception('Uh oh! Model status is {}'.format(model.status))
    # convert x to beta
    action = []
    for i in range(n_regions):
        for j in range(n_effort):
            if abs(x[i][j].x - 1) < 1e-2:
                action.append((i, effort_levels[j]))
    return action, model.ObjVal
