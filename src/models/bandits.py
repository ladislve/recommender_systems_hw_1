import numpy as np

class EpsilonGreedy:
    def __init__(self, n_arms, epsilon=0.1):
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.counts = np.zeros(n_arms)
        self.rewards = np.zeros(n_arms)

    def select_arm(self):
        if np.random.random() < self.epsilon or np.min(self.counts) == 0:
            return np.random.randint(self.n_arms)
        return int(np.argmax(self.rewards / np.maximum(self.counts, 1)))

    def update(self, arm, reward):
        self.counts[arm] += 1
        self.rewards[arm] += reward


class UCB1:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.counts = np.zeros(n_arms)
        self.rewards = np.zeros(n_arms)
        self.t = 0

    def select_arm(self):
        self.t += 1
        if np.min(self.counts) == 0:
            return int(np.argmin(self.counts))
        means = self.rewards / self.counts
        bonus = np.sqrt(2 * np.log(self.t) / self.counts)
        return int(np.argmax(means + bonus))

    def update(self, arm, reward):
        self.counts[arm] += 1
        self.rewards[arm] += reward


class ThompsonSampling:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.alpha = np.ones(n_arms)
        self.beta = np.ones(n_arms)

    @property
    def counts(self):
        return (self.alpha - 1) + (self.beta - 1)

    def select_arm(self):
        samples = np.random.beta(self.alpha, self.beta)
        return int(np.argmax(samples))

    def update(self, arm, reward):
        if reward > 0:
            self.alpha[arm] += 1
        else:
            self.beta[arm] += 1


def simulate_bandits(bandits, policies, policy_names, user_order, reward_fn, k=10):
    static_rewards = {name: [] for name in policy_names}
    bandit_rewards = {name: [] for name in bandits}
    bandit_arms = {name: [] for name in bandits}

    for uid in user_order:
        for pi, (pname, policy) in enumerate(zip(policy_names, policies)):
            recs = policy(uid, k=k)
            r = reward_fn(uid, recs)
            static_rewards[pname].append(r)

        for bname, bandit in bandits.items():
            arm = bandit.select_arm()
            recs = policies[arm](uid, k=k)
            r = reward_fn(uid, recs)
            bandit.update(arm, r)
            bandit_rewards[bname].append(r)
            bandit_arms[bname].append(arm)

    return static_rewards, bandit_rewards, bandit_arms