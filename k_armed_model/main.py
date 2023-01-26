''' finite K-armed  bandit

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

from k_armed import KArmedEnvironment, KArmedRandom, KArmedSolver, KArmedHistorical, KArmedArtificialReplay


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
print('running experiment: K-armed')
print('############################################')


########################################################################
# experiment parameters
########################################################################

# infrastucture setup
seed              = 294
num_iters         = 60 #10 #3
np.random.seed(seed)
seeds = np.arange(num_iters)


# environment setup
T      = 1000 #100 #500 #1000 #2000 #5000
K      = 10 #10
budget = 1
n_historical = 10 #100 # 10, 100, 1000, 10000



# ----------------------------
# track stats
# ----------------------------
full_regret_df = pd.DataFrame({'algo': [], 'iter': [], 't': [], 'reward': [], 'regret': []})

algo_names = ['ignorant', 'historical', 'artificial_replay', 'random']
stats = {}
for stat_name in ['runtime', 'unused_h', 'regret']:
    for algo_name in algo_names:
        stats[f'{stat_name}-{algo_name}'] = np.zeros(num_iters)



for iter in tqdm(range(num_iters)):
    np.random.seed(seeds[iter])

    regret_data = [] # dataset of regret information

    print('---------------------------------------------------------')
    print(f'iteration {iter} / {num_iters}')
    print('---------------------------------------------------------')

    ########################################################################
    # set up environment data
    ########################################################################

    env = KArmedEnvironment(K, budget, n_historical)
    print(f'optimal value for environment: {env.optimal}')


    ########################################################################
    # run algorithms
    ########################################################################

    algo_list = {
        'random':            KArmedRandom(env, T),
        'ignorant':          KArmedSolver(env, T),
        'historical':        KArmedHistorical(env, T, env.dataset),
        'artificial_replay': KArmedArtificialReplay(env, T, env.dataset),
    }

    for algo_name in algo_list:
        algo = algo_list[algo_name]

        start_time = time.time()
        for t in range(T+n_historical): #tqdm(range(T+n_historical)):
            # if the algorithm has finished, stop simulating
            if algo.regret_iterations >= T: break

            # run a one-step update to pick arm, get observation, and calculate regret
            reward, flag = algo.one_step(algo.regret_iterations)

            # if we took online step, track stats
            if flag:
                run_stats = {
                    'algo': algo_name, 'iter': iter, 't': algo.regret_iterations,
                    'reward': reward, 'regret': algo.regret}

                # track number of historical datapoints available
                if   algo_name == 'random':            run_stats['unused_h'] = n_historical
                elif algo_name == 'ignorant':          run_stats['unused_h'] = n_historical
                elif algo_name == 'historical':        run_stats['unused_h'] = 0
                elif algo_name == 'artificial_replay': run_stats['unused_h'] = algo.get_dataset_size()
                else:                                  raise NotImplementedError

                regret_data.append(run_stats)



        # log iteration-wide stats
        if   algo_name == 'random':            unused_h = n_historical
        elif algo_name == 'ignorant':          unused_h = n_historical
        elif algo_name == 'historical':        unused_h = 0
        elif algo_name == 'artificial_replay': unused_h = algo.get_dataset_size()
        else:                                  raise NotImplementedError

        stats[f'runtime-{algo_name}'][iter]  = time.time() - start_time
        stats[f'unused_h-{algo_name}'][iter] = unused_h
        stats[f'regret-{algo_name}'][iter]   = algo.regret

        # env.reset_algo()

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

p = sns.lineplot(data=full_regret_df, x='t', y='regret',
                hue='algo', #palette=palette, hue_order=hue_order,
                errorbar='se') # ('ci', 95)
p.set_title(f'k-armed, K {K}, B {budget}, n_historical {n_historical}')
# p.set_title(f'K = {K}, N = {N}, delta = {delta}, alpha = {alpha}')

out_dir     = './k_armed_b1' #'./k_armed_historical'
plots_dir   = f'{out_dir}/plots'
results_dir = f'{out_dir}/results'



# ensure directories exist
if not os.path.exists(plots_dir):   os.makedirs(plots_dir)
if not os.path.exists(results_dir): os.makedirs(results_dir)

str_time = datetime.now().strftime('%d-%m-%Y_%H:%M:%S')
file_out = f'regret_k_armed_k{K}_b{budget}_H{n_historical}_{str_time}'

full_regret_df.to_csv(f'{results_dir}/regret_{file_out}.csv')

stats_df = pd.DataFrame(stats)
stats_df.to_csv(f'{results_dir}/stats_{file_out}.csv')

plt.savefig(f'{plots_dir}/{file_out}.png')
plt.show()
