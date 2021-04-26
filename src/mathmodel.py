import scipy.stats as sps
import scipy.special as special
import numpy as np

class BanditNoiseLoopModel:

    @staticmethod
    def interest_init(M):
        assert M >= 0

        return sps.uniform(-0.5, 1).rvs(M)

    @staticmethod
    def make_response_noise(interest, w, p):
        n = len(interest)

        assert n > 0
        assert w >= 0
        noise = sps.bernoulli(p).rvs(n)
        noise_interest = interest*noise - interest*(1-noise) + sps.uniform(-w, 2*w).rvs(n)
        return sps.bernoulli(p=special.expit(noise_interest)).rvs()

    @staticmethod
    def make_response(interest):
        n = len(interest)

        assert n > 0

        return sps.bernoulli(p=special.expit(interest)).rvs()


    @staticmethod
    def get_interest_update(l, M, actions, response, win_streak, lose_streak, b):
        assert M >= l > 0

        bias = sps.uniform(0, 0.01).rvs(l)
        new_interest = np.zeros(M)
        tmp_lose_streak = lose_streak[actions] - (lose_streak[actions] > 0 )
        tmp_win_streak = win_streak[actions] - (win_streak[actions] > 0 )
        new_interest[actions] += (response * bias * (1 + b*tmp_lose_streak) - 
                bias * (1 - response) * (1 + b*tmp_win_streak)) 
        return new_interest
