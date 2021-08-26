import numpy as np
from collections import namedtuple

from mathmodel import BanditNoiseLoopModel as Model

import random

from results import MultipleResults


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

    def __init__(self, bandit_model, experiment_name):
        self.experiment_name = experiment_name
        self.bandit_model = bandit_model

    def prepare(self, w, init_interest, use_log=False):
        """
        Initializes the experiment

        :param train_size: size of the sliding window as a portion of the dataset
        :return: None
        """
        self.w = float(w)

        self.use_log = bool(use_log)

        self.bandit = self.bandit_model()
        self.init_interest = init_interest()
        if hasattr(self.bandit, 'interest'):
            self.bandit.interest = self.init_interest

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

    def update_state(self, interest, actions, response):
        pass

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

    def get_updated_interest(self, actions, responses, interest):
        return Model.get_updated_interest(
            l=self.bandit.l, M=self.bandit.M,
            interest=interest,
            actions=actions,
            response=responses
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
                w=self.w
            )
            
            cur_bandit_params = self.bandit.update(cur_actions, cur_response)

            cur_interest = self.get_updated_interest(
                actions=cur_actions,
                responses=cur_response,
                interest=cur_interest
            )

            if hasattr(self.bandit, 'interest'):
                self.bandit.interest = cur_interest
                
            self.update_state(cur_interest, cur_actions, cur_response)

            save_iter(t,
                      pr=cur_probabilities,
                      rec=cur_actions,
                      resp=cur_response,
                      params=cur_bandit_params,
                      interest=cur_interest)

            self.eval_metrics(cur_interest=cur_interest,
                              init_interest=self.init_interest)


class WinStreakLoopExperiment(BanditLoopExperiment):

    def prepare(self, w, init_interest, b=0.0, use_log=False):
        super().prepare(w, init_interest, use_log=use_log)

        self.b = float(b)
        self.win_streak = np.zeros(self.bandit.M)
        self.lose_streak = np.zeros(self.bandit.M)

    def get_updated_interest(self, actions, responses, interest):
        return Model.get_updated_interest_winstreak(
                    l=self.bandit.l, M=self.bandit.M,
                    interest=interest,
                    actions=actions, response=responses,
                    win_streak=self.win_streak*(interest > 0),
                    lose_streak=self.lose_streak*(interest < 0),
                    b=self.b)

    def update_state(self, cur_interest, cur_actions, cur_response):
        self.win_streak[cur_actions] = self.win_streak[cur_actions] * cur_response + cur_response

        self.lose_streak[cur_actions] = self.lose_streak[cur_actions] * (1 - cur_response) + \
                                        (1 - cur_response)


class RestartsLoopExperiment(BanditLoopExperiment):

    def prepare(self, w, init_interest, r0=0.0, s=1.0, use_log=False):
        super().prepare(w, init_interest, use_log=use_log)

        self.r0 = r0
        self.s = s

    def get_updated_interest(self, actions, responses, interest):
        return Model.get_updated_interest_restarts(
                    l=self.bandit.l, M=self.bandit.M,
                    interest=interest,
                    actions=actions, response=responses,
                    r0=self.r0,
                    s=self.s)