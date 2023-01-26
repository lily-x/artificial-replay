''' combinatorial metric bandit

run main experiments:
- set up the environment with the reward function
- pick a list of algorithms
- run experiment
- generate plots
'''

import sys, os
import pickle
import time
from datetime import datetime

import math
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append('process_data') # needed for pickle of Uganda data
# from process_data import learn_reward

from environment import RandomEnvironment, UgandaEnvironment, QuadraticEnvironment, PWLEnvironment

sys.path.append('algorithms/utils')
from algorithms.fixed_discretization       import FixedDiscretization
from algorithms.fixed_historical           import FixedHistorical
from algorithms.fixed_artificial_replay    import FixedArtificialReplay
from algorithms.adaptive_discretization    import AdaptiveDiscretization
from algorithms.adaptive_historical        import AdaptiveHistorical
from algorithms.adaptive_artificial_replay import AdaptiveArtificialReplay
from algorithms.regression                 import Regression


def smooth(rewards, weight=0.7):
    """ smoothed exponential moving average """
    prev = rewards[0]
    smoothed = np.zeros(len(rewards))
    for i, val in enumerate(rewards):
        smoothed_val = prev * weight + (1 - weight) * val
        smoothed[i] = smoothed_val
        prev = smoothed_val

    return smoothed



print('############################################')
print('running experiment: combinatorial bandit')
print('############################################')


########################################################################
# experiment parameters
########################################################################

INHERIT_FLAG      = True      # whether estimates are inherited from parent to child in adaptive discretization algorithm

seed              = 294
num_iters         = 1#60 #3 #10 #3 # 10

# ----------------------------
# environment settings
# ----------------------------

env_name = 'quadratic'
# env_name = 'random'
# env_name = 'pwl'
# env_name = 'uganda'

T             = 1000 #5000 #2000 #500
N             = 5
budget        = 3
n_historical  = 300 #1000

effort_levels = [0, 0.5, 1]
epsilon       = 0.03125 # discretization parameter for uniform discretization based algorithm


# only for quadratic setting
# for analysis with spurious data
frac_bad_arms = None # default: 0.2. fraction of historical data to put on the worst arms

# ----------------------------
# track stats
# ----------------------------
full_regret_df = pd.DataFrame({'algo': [], 'iter': [], 't': [], 'reward': [], 'regret': []})

algo_names = ['fixed_ignorant', 'fixed_historical', 'fixed_artificial_replay',
              'adaptive_ignorant', 'adaptive_historical', 'adaptive_artificial_replay',
              'regression', 'optimal']
stats = {}
for stat_name in ['runtime', 'unused_h', 'regret', 'n_regions']:
    for algo_name in algo_names:
        stats[f'{stat_name}-{algo_name}'] = np.zeros(num_iters)


# ----------------------------
# execute
# ----------------------------

np.random.seed(seed)
seeds = np.arange(num_iters)

for iter in tqdm(range(num_iters)):
    np.random.seed(seeds[iter])

    regret_data = [] # dataset of regret information

    # print('---------------------------------------------------------')
    # print(f'iteration {iter} / {num_iters}')
    # print('---------------------------------------------------------')

    ########################################################################
    # set up environment data
    ########################################################################

    if env_name == 'uganda' or env_name == 'uganda_real_h':
        filename_in = './process_data/Uganda.pickle'
        with open(filename_in, 'rb') as f:
            data = pickle.load(f)

        reward_fn = data['net']

        if env_name == 'uganda_real_h':
            print('historical data size', data['points'].shape)
            # import pdb; pdb.set_trace()
            np.random.shuffle(data['points'])
            np.random.shuffle(data['efforts'])
            np.random.shuffle(data['rewards'])
            historical_data = {'points':  data['points'][:n_historical],
                               'efforts': data['efforts'][:n_historical],
                               'rewards': data['rewards'][:n_historical]}
        else:
            historical_data = None

        env = UgandaEnvironment(reward_fn, effort_levels, N, budget, historical_data=historical_data, n_historical=n_historical)

    elif env_name == 'random':
        env = RandomEnvironment(effort_levels, N, budget, n_historical)

    elif env_name == 'quadratic':
        env = QuadraticEnvironment(effort_levels, N, budget, n_historical, frac_bad_arms=frac_bad_arms)

    elif env_name == 'pwl':
        env = PWLEnvironment(effort_levels, N, budget, n_historical)

    elif env_name == 'simple':
        env = SimpleEnvironment(.05, 0, 10)


    # create a grid of [0,1] based on the epsilon parameter
    # we create the grid here to ensure it's standardized across all algorithms and the optimal solution solver
    grid_discretization = np.linspace(0, 1, math.ceil(1/epsilon))

    # calculate the optimal objective value for the environment
    env.optimal = env.get_optimal(N, budget, grid_discretization)

    print(f'  discretization: epsilon {epsilon}, grid {len(grid_discretization)}: {grid_discretization.round(2)}')
    print(f'  optimal value for environment: {env.optimal}')


    ########################################################################
    # run algorithms
    ########################################################################

    algo_list = {
        # 'fixed_ignorant' :            FixedDiscretization(env, T, grid_discretization),
        # 'fixed_historical' :          FixedHistorical(env, T, grid_discretization, env.dataset),
        # 'fixed_artificial_replay' :   FixedArtificialReplay(env, T, grid_discretization, env.dataset),
        'adaptive_ignorant' :         AdaptiveDiscretization(env, T, epsilon, INHERIT_FLAG),
        'adaptive_historical' :       AdaptiveHistorical(env, T, epsilon, INHERIT_FLAG, env.dataset),
        'adaptive_artificial_replay': AdaptiveArtificialReplay(env, T, epsilon, INHERIT_FLAG, env.dataset),
        # 'regression':                 Regression(env, T, grid_discretization, env.dataset)
    }


    for algo_name in algo_list:
        print('-----------------------------')
        print(f'  {algo_name}')
        print('-----------------------------')
        algo = algo_list[algo_name]

        start_time = time.time()
        for t in range(T+n_historical):
            # if the algorithm has finished, stop simulating
            if algo.regret_iterations >= T: break

            # run a one-step update to pick arm, get observation, and calculate regret
            reward, flag = algo.one_step(algo.regret_iterations)

            # 'Average per-episode regret': algorithm.regret / algorithm.regret_iterations}

            # if we took online step, track stats
            if flag:
                run_stats = {'algo': algo_name, 'iter': iter, 't': algo.regret_iterations,
                         'reward': reward, 'regret': algo.regret}

                # print(t, run_stats)

                # track
                regret_data.append(run_stats)
        env.reset_algo()

        # log iteration-wide stats per algo
        stats[f'runtime-{algo_name}'][iter]   = time.time() - start_time
        stats[f'regret-{algo_name}'][iter]    = algo.regret
        stats[f'unused_h-{algo_name}'][iter]  = -1
        stats[f'n_regions-{algo_name}'][iter] = -1

        if algo_name in ['fixed_artificial_replay', 'adaptive_artificial_replay']:
            stats[f'unused_h-{algo_name}'][iter] = algo.get_dataset_size()

        if algo_name in ['adaptive_ignorant', 'adaptive_historical', 'adaptive_artificial_replay']:
            stats[f'n_regions-{algo_name}'][iter] = algo.get_tree_size()



    # add optimal to regret data
    opt_data = {'algo': ['optimal'] * T, 'iter': [iter] * T, 't': np.arange(T),
                'reward': [env.optimal] * T, 'regret': np.zeros(T)}

    regret_df = pd.DataFrame(regret_data)
    opt_df    = pd.DataFrame(opt_data)

    full_regret_df = pd.concat([full_regret_df, regret_df, opt_df], copy=False)
    # print(full_regret_df)



# ----------------------------
# plot
# ----------------------------

palette = [#'darkgreen', 'lime', 'palegreen', 'seagreen',
            'darkgreen', 'royalblue', 'palegreen', #'cornflowerblue',
            'darkred', 'orange', 'yellow', #'indianred',
           # 'midnightblue', 'mediumblue', 'cornflowerblue', 'royalblue',
           # 'darkred', 'indianred', 'salmon',
           'pink', 'gray']  # 'darkolivegreen', 'olivedrab', 'yellowgreen'
# hue_order = ['fixed_ignorant', 'fixed_historical', 'fixed_artificial_replay', #'Pseudo Online Fixed',
#              'adaptive_ignorant', 'adaptive_historical', 'adaptive_artificial_replay', #'Pseudo Online Adaptive',
#              # 'Ignorant Regression', 'Historical Regression', 'Pseudo Online Regression',
#              'regression', 'optimal']

# weird workaround to only list algorithms used
algo_list['optimal'] = None  # needed to include line in plot
palette   = [palette[i] for i, algo_name in enumerate(algo_names) if algo_name in algo_list.keys()]
hue_order = [algo_name  for i, algo_name in enumerate(algo_names) if algo_name in algo_list.keys()]

# smoothing
# full_regret_df['Reward Smooth'] = smooth(full_regret_df['Reward'])

p = sns.lineplot(data=full_regret_df, x='t', y='regret',
                hue='algo', palette=palette, hue_order=hue_order,
                errorbar='se') # ('ci', 95)
title = f'env {env_name}, N {N}, B {budget}, eps {epsilon}, n_historical {n_historical}, n_effort {len(effort_levels)}'
if env_name == 'quadratic':
    title = f'{title}, frac {frac_bad_arms}'
p.set_title(title)


# ----------------------------
# write out
# ----------------------------

out_dir     = f'./{num_iters}_iters'
plots_dir   = f'{out_dir}/plots'
results_dir = f'{out_dir}/results'

# ensure directories exist
if not os.path.exists(plots_dir):   os.makedirs(plots_dir)
if not os.path.exists(results_dir): os.makedirs(results_dir)

str_time = datetime.now().strftime('%d-%m-%Y_%H:%M:%S')
file_out = f'regret_env-{env_name}_n{N}_b{budget}_eps{epsilon}_H{n_historical}_effort{len(effort_levels)}_{str_time}'

if env_name == 'quadratic' and frac_bad_arms is not None:
    file_out = f'frac{frac_bad_arms}_{file_out}'

full_regret_df.to_csv(f'{results_dir}/{file_out}.csv')

stats_df = pd.DataFrame(stats)
stats_df.to_csv(f'{results_dir}/stats_{file_out}.csv')

plt.savefig(f'{plots_dir}/{file_out}.png')
plt.show()
