import numpy as np
from matplotlib import pyplot as plt
import seaborn as sb
import scipy.stats as sps


import random


def init_random_state(seed):
    np.random.seed(int(seed))
    random.seed = int(seed)
    return seed


def init_data(M):
    return [sps.uniform(-1, 2).rvs(M)], [sps.uniform.rvs(size=(M,2))] 

def sigmoid(x):
    return 1. / (1 + np.exp(-x))

def make_response_noise(interest, w, Q, p):
    n = len(interest)
    noise_interest = interest + w*(sps.beta(Q, Q).rvs(n) - 0.5)
    return sps.bernoulli(p = sigmoid(noise_interest)).rvs()

def make_response(interest):
    return sps.bernoulli(p = sigmoid(interest)).rvs()

def single_model_experiment(T, M, l, w, Q, p):
    interest, TS_params = init_data(M) 
    probalities = []
    recomendations = []
    response = []

    for t in range(T):
        probalities.append(
            sps.beta(a = TS_params[t][:,0], b = TS_params[t][:,1]).rvs()
        )
        
        recomendations.append(np.argsort(-probalities[t])[:l])
        
        response.append(
            make_response_noise(interest[t][recomendations[t]], w, Q, p)
        )
        
        new_params = TS_params[t].copy()
        new_params[recomendations[t]] += np.vstack([response[t], 1-response[t]]).T
        TS_params.append(new_params)
        
        bias = sps.uniform(0, 0.01).rvs(l)
        new_interest = interest[t].copy()
        new_interest[recomendations[t]] += response[t]*bias - bias*(1-response[t]) 
        interest.append(new_interest)

    interest = np.array(interest)
    TS_params = np.array(TS_params)
    probalities = np.array(probalities)
    recomendations = np.array(recomendations)
    response = np.array(response)

    return interest, TS_params, probalities, recomendations, response

class HiddenLoopExperiment:
    """
    The main experiment for hidden loops paper
    See details in the paper.

    In short.

    Creates a feedback loop on a regression problem (e.g. Boston housing).
    Some of the model predictions are adhered to by users and fed back into the model as training data.
    Users add a normally distributed noise to the log of the target variable (price).
    Uses a sliding window to retrain the model on new data.

    """
    def __init__(self, T, M, l, w, Q, p, seed):
        self.T = T
        self.M = M
        self.l = l
        self.w = w 
        self.Q = Q
        self.p = p
        self.seed = seed

    def _prepare(self, use_log=False):
        """
        Initializes the experiment

        :param train_size: size of the sliding window as a portion of the dataset
        :return: None
        """
        init_random_state(self.seed)
        self.use_log = bool(use_log)

        self.interest, self.TS_params = init_data(self.M) 
        self.probalities = []
        self.recomendations = []
        self.response = []

        self.index = []

    def _predict_TS(self, params):
        pr = sps.beta(a = params[:,0], b = params[:,1]).rvs() 
        rec = np.argsort(-pr)[:self.l] 
        return pr, rec

    def _make_response(self, interest):
        n = len(interest)
        noise_interest = interest + self.w*(sps.beta(self.Q, self.Q).rvs(n) - 0.5)
        return sps.bernoulli(p = sigmoid(noise_interest)).rvs()

    def _update_params(self, params, rec, response):
        new_params = params.copy()
        new_params[rec] += np.vstack([response, 1-response]).T
        return new_params

    def _update_interest(self, interest, rec, response):
        bias = sps.uniform(0, 0.01).rvs(self.l)
        new_interest = interest.copy()
        new_interest[rec] += response*bias - bias*(1-response) 
        return new_interest

    def _convert_data_np(self):
        self.interest = np.array(self.interest)
        self.TS_params = np.array(self.TS_params)
        self.probalities = np.array(self.probalities)
        self.recomendations = np.array(self.recomendations)
        self.response = np.array(self.response)

    def run_experiment(self):
        self._prepare()

        def save_iter(t, pr, rec, resp, params, interest):
            self.index.append(t)
            self.probalities.append(pr)
            self.recomendations.append(rec)
            self.response.append(resp)
            self.TS_params.append(params)
            self.interest.append(interest)


        cur_interest = self.interest[0]
        cur_TS_params = self.TS_params[0]
        cur_probalities = None
        cur_recomendations = None
        cur_response = None

        for t in range(self.T):
            cur_probalities, cur_recomendations = self._predict_TS(cur_TS_params) 
            
            cur_response = self._make_response(
                cur_interest[cur_recomendations]
            )
            
            cur_TS_params = self._update_params(cur_TS_params, cur_recomendations, cur_response)
            
            cur_interest = self._update_interest(cur_interest, cur_recomendations, cur_response)
            save_iter(t, cur_probalities, cur_recomendations, cur_response, cur_TS_params, cur_interest)
            

        self._convert_data_np()
