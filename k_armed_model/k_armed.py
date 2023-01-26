import numpy as np


def conf_r(T, t, n_pulls):
    """ compute confidence radius """
    return np.sqrt(2*np.log(1+T) / n_pulls)


class KArmedEnvironment:# (Environment):
    ''' K-armed bandit with random rewards
    simple one-dimensional base class '''
    def __init__(self, K, budget, n_historical=0):
        self.dataset = {'arms':    [],
                        'rewards': []}

        self.K           = K
        self.budget      = budget

        self.mean_reward = np.random.rand(K)

        self.optimal_action = self.get_optimal_action()
        self.optimal = np.dot(self.mean_reward, self.optimal_action)  # optimal reward

        # generate historical data
        self.n_historical = n_historical

        historical_arms = np.random.choice(self.K, n_historical, replace=True)

        for h in range(n_historical):
            true_mean  = self.mean_reward[historical_arms[h]]
            obs_reward = np.random.binomial(n=1, p=true_mean)

            self.add_to_dataset(historical_arms[h], obs_reward)


    def effort_to_j(self, effort):
        raise Exception('this function does not apply to K-armed bandit')

    def j_to_effort(self, j):
        raise Exception('this function does not apply to K-armed bandit')

    def get_point(self):
        raise Exception('this function does not apply to K-armed bandit')

    def get_random_arm(self):
        return np.random.choice(self.K)

    def get_dataset(self):
        return self.dataset

    def get_optimal_action(self):
        ''' optimal one-step action '''
        opt_arms = self.mean_reward.argsort()[-self.budget:][::-1]
        opt_action = np.zeros(self.K).astype(int)
        opt_action[opt_arms] = 1
        return opt_action

    def check_valid_action(self, action):
        if (len(action) != self.K or                    # each arm accounted for
            np.sum(action) > self.budget or             # within budget
            np.sum(action) < 1 or                       # must pull at least one arm
            np.all(~(action == 1) | (action == 0))):    # all actions 0 or 1
            raise Exception(f'invalid action {action} with K={self.K} arms and budget {self.budget}')

        return True


    def sample_obs(self, action):
        ''' return vector of samples from true underlying reward '''
        self.check_valid_action(action)

        obs = np.zeros(self.K).astype(int)
        for arm in range(self.K):
            # only get samples for arms we pull
            if action[arm] == 1:
                true_mean = self.mean_reward[arm]
                obs[arm] = np.random.binomial(n=1, p=true_mean)
        return obs


    def add_to_dataset(self, arms, rewards):
        ''' add a single or multiple points to historical dataset '''
        if np.isscalar(arms): # add single observation
            self.dataset['arms'].append(arms)
            self.dataset['rewards'].append(rewards)

        else:
            assert len(arms) == len(rewards)
            for h in range(len(arms)):
                self.dataset['arms'].append(arms[h])
                self.dataset['rewards'].append(rewards[h])

    def get_historical_a_r(self, h):
        ''' h is index into historical dataset
        return (action, reward) tuple '''

        assert h < len(self.dataset['rewards'])
        return self.dataset['arms'][h, :], self.dataset['rewards'][h]

    def reset_algo(self):
        pass

    def reset_data(self, RESET_DATA_FLAG):
        pass


class KArmedSolver: #(Algorithm):
    '''
    UCB-based algorithm for k-armed bandit
    '''
    def __init__(self, env, T):
        self.env    = env
        self.K      = env.K
        self.budget = env.budget
        self.T      = T

        # track number of iterations and cumulative regret
        self.regret = 0
        self.regret_iterations = 0

        self.cum_reward = np.zeros(self.K)
        self.num_visits = np.zeros(self.K)

        # track previous UCB to ensure monotone decreasing
        self.prev_ucb   = np.ones(self.K)


    def reset(self):
        ''' Reset regret and number of iterations back to zero '''
        self.regret = 0
        self.regret_iterations = 0

        self.cum_reward = np.zeros(self.K)
        self.num_visits = np.zeros(self.K)
        self.prev_ucb   = np.ones(self.K)


    def update_combinatorial_action_obs(self, action, rewards):
        ''' update estimates from one full combinatorial action '''
        assert self.env.check_valid_action(action)
        assert len(action) == len(rewards) == self.K
        for arm in range(self.K):
            # only update for arms we actually act on
            if action[arm] == 1:
                self.num_visits[arm] += 1
                self.cum_reward[arm] += rewards[arm]


    def update_single_arm_obs(self, arm, reward):
        ''' update estimates from a single observation '''
        assert np.isscalar(arm)

        self.num_visits[arm] += 1
        self.cum_reward[arm] += reward


    def pick_action(self, t):
        ''' determine action according to UCB estimates '''
        ucb_estimates = np.zeros(self.K)

        for arm in range(self.K):
            if self.num_visits[arm] > 0:
                reward_estimate = self.cum_reward[arm] / self.num_visits[arm]
                ucb_estimates[arm] = reward_estimate + conf_r(self.T, t, self.num_visits[arm])

                # ensure monotone
                if ucb_estimates[arm] > self.prev_ucb[arm]:
                    ucb_estimates[arm] = self.prev_ucb[arm]
                else:
                    self.prev_ucb[arm] = ucb_estimates[arm]

            else:
                ucb_estimates[arm] = 1

        # pick arms with highest UCB estimates
        chosen_arms = ucb_estimates.argsort()[-self.budget:][::-1]
        action = np.zeros(self.K).astype(int)
        action[chosen_arms] = 1

        # print('  ', t, 'true', self.env.mean_reward.round(2), 'ucb', ucb_estimates.round(2), 'action', action)

        return action


    def one_step(self, t):
        '''
        One-Step loop for an online algorithm.
        Starts by picking an action according to the current timestep
        Updates the number of regret iterations, and updates regret based on the selected arm
        Obtains an observation sampled from the true distribution of the selected action
        Updates the observation
        Returns true (indicating that an online sample was taken and regret should be added)
        '''

        action = self.pick_action(self.regret_iterations)
        self.regret_iterations += 1

        obs_reward = self.env.sample_obs(action)
        tot_reward = obs_reward.sum()

        # print(f'  {t} action {action} obs {obs_reward}')

        # check for no funny business
        for arm in range(self.K):
            if action[arm] == 0:
                assert obs_reward[arm] == 0, f'something wrong {obs_reward}, {action}'

        self.regret += self.env.optimal - tot_reward
        self.update_combinatorial_action_obs(action, obs_reward)

        # print(f'return tot {tot_reward}')

        return tot_reward, True


class KArmedRandom(KArmedSolver):
    def __init__(self, env, T):
        super().__init__(env, T)
        self.reset()

    def pick_action(self, t):
        ''' take a random action '''
        arms = np.random.choice(self.K, self.budget, replace=False)
        action = np.zeros(self.K)
        action[arms] = 1
        return action


class KArmedHistorical(KArmedSolver):
    def __init__(self, env, T, dataset):
        super().__init__(env, T)

        self.dataset = dataset
        self.reset()


    def reset(self):
        super().reset()

        for h in range(len(self.dataset['arms'])):
            arm    = self.dataset['arms'][h]
            reward = self.dataset['rewards'][h]

            self.cum_reward[arm] += reward
            self.num_visits[arm] += 1



class KArmedArtificialReplay(KArmedSolver):
    def __init__(self, env, T, dataset):
        super().__init__(env, T)

        self.dataset = dataset

        # process historical dataset into parse-able dictionary
        self.history = {}
        for i in range(self.K):
            self.history[i] = []

        for h in range(len(self.dataset['arms'])):
            arm    = self.dataset['arms'][h]
            reward = self.dataset['rewards'][h] #.item()?

            self.history[arm].append(reward)


    def get_dataset_size(self):
        ''' return number of reward observations remaining in historical data '''
        n_obs = 0
        for arm in range(self.K):
            n_obs += len(self.history[arm])
        return n_obs


    def dataset_contains(self, arm):
        ''' check whether historical dataset contains samples of a given point

        returns False if does not contain
        otherwise returns the (loc, reward) sample observed in history '''

        if len(self.history[arm]) > 0:
            reward = self.history[arm].pop()
            return (arm, reward)

        return False


    def one_step(self, t):
        ''' returns flag: (bool) True if we took an online sample '''

        action = self.pick_action(self.regret_iterations) # pick a combinatorial action
        self.env.check_valid_action(action)

        used_historical = False

        # look in historical data for reward samples from subarms
        obs_reward = np.zeros(self.K)
        for arm in range(self.K):
            if action[arm] != 1: continue  # only look at arms we actually act on

            check_data = self.dataset_contains(arm)
            if check_data is not False:
                # update estimates for those entries in the dataset
                obs_reward[arm] = check_data[1]
                self.update_single_arm_obs(arm, obs_reward[arm])

                used_historical = True

        # take an online action only if no subarm is in historical dataset
        if not used_historical:
            obs_reward = self.env.sample_obs(action)

            assert obs_reward.sum() <= self.budget # ensure no funny business

            self.regret_iterations += 1
            self.regret += self.env.optimal - np.sum(obs_reward)
            self.update_combinatorial_action_obs(action, obs_reward)

        if t % 100 == 99:
            print(f'  {t}: historical dataset size {self.get_dataset_size()}, used_historical {used_historical}')

        return np.sum(obs_reward).item(), not used_historical


    def reset(self):
        ''' reset all quantities '''
        super().reset()

        # reset history
        self.history = {}
        for i in range(self.K):
            self.history[i] = []

        for h in range(len(self.dataset['arms'])):
            arm    = self.dataset['arms'][h]
            reward = self.dataset['rewards'][h]

            self.history[arm].append(reward)


    def used_all_history(self):
        ''' returns whether we've gone through all historical data '''
        return self.get_dataset_size() == 0


    def history_use_percentage(self):
        ''' percentage of the historical data we've used '''
        curr_n_historical     = self.get_dataset_size()
        original_n_historical = self.env.n_historical
        return curr_n_historical / original_n_historical
