import numpy as np
from scipy import stats as sps


class TSBandit:
    """
    Implements a Thompson-Sampling Multiarmed Bandit algorithm for the item recommendation problem

    See (insert link) for details

    """
    def __init__(self, M, l):
        """
        Creates a new instance of the TS Bandit for item recommendation problem

        Inits Beta params [a, b] to uniformly sampled in [0, 1]

        :param M: number of actions
        :param l: return this number of recommendations
        """
        self.l = l
        self.M = M
        self.params = np.ones(shape=(self.M, 2))

    def predict(self):
        """
        Get the next prediction from the bandit
        :return: an np array with action probabilities, an np array of selected actions
        """
        pr = sps.beta.rvs(a=self.params[:,0], b=self.params[:,1])
        rec = np.argsort(-pr)[:self.l]
        return pr, rec

    def update(self, actions, response):
        """
        Updates the bandit with responses to previously given actions

        :param actions: actions for which updates are
        :param response: the rewards for the actions
        :return: an updated params
        """
        self.params[actions] += np.vstack([response, 1 - response]).T
        return self.params.copy()


class RandomModel:
    """
    """
    def __init__(self, M, l):
        """
        :param M: number of actions
        :param l: return this number of recommendations
        """
        self.l = l
        self.M = M

    def predict(self):
        """
        Get the next prediction
        :return: an np array with action probabilities, an np array of selected actions
        """
        return [], np.random.permutation(self.M)[:self.l]

    def update(self, actions, response):
        return []


class OptimalModel:
    """
    """
    def __init__(self, M, l):
        """
        :param M: number of actions
        :param l: return this number of recommendations
        """
        self.l = l
        self.M = M
        self.interest = None

    def predict(self):
        """
        Get the next prediction
        :return: an np array with action probabilities, an np array of selected actions
        """
        return [], np.argsort(-self.interest)[:self.l]

    def update(self, actions, response):
        return []


class EpsilonGreedyModel:
    """
    """
    def __init__(self, M, l, epsilon):
        """
        :param M: number of actions
        :param l: return this number of recommendations
        """
        self.l = l
        self.M = M
        self.epsilon = epsilon
        self.interest = None

    def predict(self):
        """
        Get the next prediction
        :return: an np array with action probabilities, an np array of selected actions
        """
        cur_max = np.argsort(-self.interest)[:self.l]
        random_choice = np.random.permutation(self.M)[:self.l]
        greedy = sps.bernoulli.rvs(self.epsilon, self.l)
        return [], greedy*random_choice + (1-greedy)*cur_max

    def update(self, actions, response):
        return []


def ts_model(M, l):
    assert l <= M
    return TSBandit(M=int(M), l=int(l))


def random_model(M, l):
    assert l <= M
    return RandomModel(M=int(M), l=int(l))


def optimal_model(M, l):
    assert l <= M
    return OptimalModel(M=int(M), l=int(l))


def epsilon_greedy_model(M, l, epsilon):
    assert l <= M
    assert 0 < epsilon < 1.0
    return EpsilonGreedyModel(M=int(M), l=int(l), epsilon=float(epsilon))