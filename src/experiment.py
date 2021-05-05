import numpy as np
import scipy.stats as sps
from collections import namedtuple

from mathmodel import BanditNoiseLoopModel as Model

import random

from results import MultipleResults


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
        pr = sps.beta(a=self.params[:,0], b=self.params[:,1]).rvs()
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
        greedy = sps.bernoulli(self.epsilon).rvs(self.l)
        return [], greedy*random_choice + (1-greedy)*cur_max

    def update(self, actions, response):
        return []


def get_ts_model(M, l):
    assert l <= M
    return TSBandit(M=int(M), l=int(l))

def get_random_model(M, l):
    assert l <= M
    return RandomModel(M=int(M), l=int(l))

def get_optimal_model(M, l):
    assert l <= M
    return OptimalModel(M=int(M), l=int(l))

def get_epsilon_greedy_model(M, l, epsilon):
    assert l <= M
    assert 0 < epsilon < 1.0
    return EpsilonGreedyModel(M=int(M), l=int(l), epsilon=float(epsilon))

def init_random_state(seed):
    np.random.seed(int(seed))
    random.seed = int(seed)
    return seed

def skip_params(M, l):
    return not (M >= l)

class BanditLoopExperiment:
    """
    The main experiment for hidden loops paper
    See details in the paper.
    """

    default_state = {
        'interest': 'Current interests:{}',
        'probabilities': 'Estimated success probas:{}',
        'recommendations': 'Actions:{}',
        'loop_amp': 'Loop effect amplitude',
        #'bandit_params': 'Reward estimates, TS bandit state'
    }

    default_figures = {
        'Loop effect': ['loop_amp'],
        'Estimated success probas': ['probabilities'],
        'Interests': ['interest'],
        'Actions': {'data':['recommendations'],'plot_fun': MultipleResults.scatterplot}
    }

    def __init__(self, bandit_model, bandit_name):
        self.bandit_name = bandit_name
        self.bandit_model = bandit_model

    def prepare(self, w, Q, p, b, init_interest, use_log=False):
        """
        Initializes the experiment

        :param train_size: size of the sliding window as a portion of the dataset
        :return: None
        """
        self.w = float(w)
        self.p = float(p)
        self.b = float(b)

        self.use_log = bool(use_log)

        self.bandit = self.bandit_model()
        self.init_interest = init_interest()
        if hasattr(self.bandit, 'interest'):
            self.bandit.interest = self.init_interest

        self.win_streak = np.zeros(self.bandit.M)
        self.lose_streak = np.zeros(self.bandit.M)
        self.interest = []
        self.probabilities = []
        self.recommendations = []
        self.response = []
        self.bandit_params = []

        self.loop_amp = []

        self.index = []

    def eval_metrics(self, cur_interest, init_interest):
        cur_interest = np.asarray(cur_interest)
        init_interest = np.asarray(init_interest)
        cur_loop_amp = np.linalg.norm(cur_interest - init_interest)**2
        self.loop_amp += [cur_loop_amp]

    ResultsTuple = namedtuple('ResultsNumpy',
                              ['interest', 'TS_params',
                               'probabilities', 'recommendations',
                               'response'])

    def get_as_np(self):
        return BanditLoopExperiment.ResultsTuple(
            interest=np.array(self.interest),
            TS_params=np.array(self.bandit_params),
            probabilities=np.array(self.probabilities),
            recommendations=np.array(self.recommendations),
            response=np.array(self.response)
        )

    def run_experiment(self, T):
        def save_iter(t, pr, rec, resp, params, interest):
            self.index.append(t)
            self.probabilities.append(pr)
            self.recommendations.append(rec)
            self.response.append(resp)
            self.bandit_params.append(params)
            self.interest.append(interest)

        cur_interest = self.init_interest

        for t in range(T):
            cur_probabilities, cur_actions = self.bandit.predict()
            
            cur_response = Model.make_response_noise(
                cur_interest[cur_actions],
                w=self.w,
                p=self.p
            )
            
            cur_bandit_params = self.bandit.update(cur_actions, cur_response)
            interest_update  = Model.get_interest_update(
                    l=self.bandit.l, M=self.bandit.M, actions=cur_actions, response=cur_response, 
                    win_streak=self.win_streak*(cur_interest > 0),
                    lose_streak=self.lose_streak*(cur_interest < 0),
                    b=self.b)    


            cur_interest = cur_interest + interest_update
            if hasattr(self.bandit, 'interest'):
                self.bandit.interest = cur_interest

            # TODO: create function for update
            self.win_streak[cur_actions] = self.win_streak[cur_actions]*cur_response + cur_response

            self.lose_streak[cur_actions] = self.lose_streak[cur_actions]*(1-cur_response) + \
                    (1-cur_response)

            save_iter(t,
                      pr=cur_probabilities,
                      rec=cur_actions,
                      resp=cur_response,
                      params=cur_bandit_params,
                      interest=cur_interest)

            self.eval_metrics(cur_interest=cur_interest,
                              init_interest=self.init_interest)
