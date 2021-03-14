import numpy as np
import scipy.stats as sps

from mathmodel import BanditNoiseLoopModel as Model

import random


class TSBandit:

    def __init__(self, M, l):
        self.l = l
        self.M = M
        self.params = sps.uniform.rvs(size=(self.M, 2))

    def predict(self):
        pr = sps.beta(a = self.params[:,0], b = self.params[:,1]).rvs()
        rec = np.argsort(-pr)[:self.l]
        return pr, rec

    def update(self, actions, response):
        self.params[actions] += np.vstack([response, 1 - response]).T
        return self.params


def get_ts_model(M, l):
    return TSBandit(M=M, l=l)


def init_random_state(seed):
    np.random.seed(int(seed))
    random.seed = int(seed)
    return seed


class BanditLoopExperiment:
    """
    The main experiment for hidden loops paper
    See details in the paper.

    In short.

    Creates a feedback loop on a regression problem (e.g. Boston housing).
    Some of the model predictions are adhered to by users and fed back into the model as training data.
    Users add a normally distributed noise to the log of the target variable (price).
    Uses a sliding window to retrain the model on new data.

    """

    default_state = {
        'interest': 'Current interests',
        'probabilities': 'Action probabilities',
        'loop_amp': 'Loop effect amplitude',
    }

    default_figures = {
        'Loop effect': ['loop_amp'],
        'Interests': ['interest']
    }

    def __init__(self, bandit, bandit_name):
        self.bandit_name = bandit_name
        self.bandit = bandit

    def prepare(self, M, l, w, Q, p, use_log=False):
        """
        Initializes the experiment

        :param train_size: size of the sliding window as a portion of the dataset
        :return: None
        """
        self.w = w
        self.Q = Q
        self.p = p

        self.use_log = bool(use_log)

        self.init_interest = Model.interest_init(self.bandit.M)
        self.interest = [self.init_interest]
        self.probabilities = []
        self.recommendations = []
        self.response = []
        self.bandit_params = []

        self.loop_amp = []

        self.index = []

    def eval_metrics(self, cur_interest, init_interest):
        cur_loop_amp = np.linalg.norm(cur_interest - init_interest, axis=1)**2
        self.loop_amp += [cur_loop_amp]

    def get_as_np(self):
        result = object()
        result.interest = np.array(self.interest)
        result.TS_params = np.array(self.bandit_params)
        result.probabilities = np.array(self.probabilities)
        result.recommendations = np.array(self.recommendations)
        result.response = np.array(self.response)

    def run_experiment(self, T):
        def save_iter(t, pr, rec, resp, params, interest):
            self.index.append(t)
            self.probabilities.append(pr)
            self.recommendations.append(rec)
            self.response.append(resp)
            self.bandit_params.append(params)
            self.interest.append(interest)

        cur_interest = self.interest[0]

        for t in range(T):
            cur_probabilities, cur_actions = self.bandit.predict()
            
            cur_response = Model.make_response(
                cur_interest[cur_actions]
            )
            
            cur_bandit_params = self.bandit.update(cur_actions, cur_response)
            
            cur_interest = Model.get_interest_update(l=self.bandit.l, actions=cur_actions, response=cur_response)
            save_iter(t, cur_probabilities, cur_actions, cur_response, cur_bandit_params, cur_interest)

            self.eval_metrics(cur_interest, self.init_interest)