import scipy.stats as sps
import scipy.special as special
import numpy as np

class BanditNoiseLoopModel:

    @staticmethod
    def interest_init(M):
        assert M >= 0

        return sps.uniform(-0.5, 1).rvs(M)

    @staticmethod
    def make_response_noise(interest, w):
        n = len(interest)

        assert n > 0
        assert w >= 0
        noise_interest = interest + sps.uniform(-w, 2*w).rvs(n)
        return sps.bernoulli(p=special.expit(noise_interest)).rvs()

    @staticmethod
    def make_response(interest):
        n = len(interest)

        assert n > 0

        return sps.bernoulli(p=special.expit(interest)).rvs()


    @staticmethod
    def get_updated_interest(l, M, interest, actions, response):
        assert M >= l > 0

        bias = sps.uniform(0, 0.01).rvs(l)
        new_interest = np.zeros(M)
        new_interest[actions] += (response * bias -
                bias * (1 - response))
        return interest + new_interest

    @staticmethod
    def get_updated_interest_winstreak(l, M, interest, actions, response, win_streak, lose_streak, b):
        assert M >= l > 0

        bias = sps.uniform(0, 0.01).rvs(l)
        new_interest = np.zeros(M)
        tmp_lose_streak = lose_streak[actions] - (lose_streak[actions] > 0 )
        tmp_win_streak = win_streak[actions] - (win_streak[actions] > 0 )
        new_interest[actions] += (response * bias * (1 + b*tmp_lose_streak) -
                bias * (1 - response) * (1 + b*tmp_win_streak))
        return interest + new_interest

    @staticmethod
    def get_updated_interest_restarts(l, M, interest, actions, response, s, r0):
        """
        Calculates new interests when a scaled restart is possible

        That is, for selected elements:

        new_interest = scale * (old_interest + bias * update) + (1 - scale) * init_interest

        :param l:
        :param M:
        :param interest:
        :param actions:
        :param response:
        :param s: scale
        :param r0: probability to restart
        :return:
        """
        assert M >= l > 0

        mask = actions[np.random.uniform(size=l) < r0]

        restart = BanditNoiseLoopModel.interest_init(M)

        scale = np.ones(M)
        scale[mask] = s

        updated_interest = BanditNoiseLoopModel.get_updated_interest(
            l=l, M=M, interest=interest, actions=actions, response=response)

        # todo redo mask, apply?
        index_mask = np.zeros(M)
        index_mask[mask] = 1
        new_interest = updated_interest * (1 - index_mask) + \
                       ((1 - scale) * restart + scale * updated_interest) * index_mask
        return new_interest
