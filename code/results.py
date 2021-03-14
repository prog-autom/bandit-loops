import seaborn as sb
from matplotlib import pyplot as plt
import pandas as pd


class MultipleResults:
    index_keys = ['round', 'trial']

    def __init__(self, model_name, **initial_state):
        self.model_name = model_name
        self.state_vars = initial_state
        for k, v in initial_state.items():
            vars(self)[k] = list()

    def add_state(self, trial=None, **update_state):
        for k in self.state_vars.keys():
            vars(self)[k].extend([{'round':i, 'trial':trial, k:j} for i, j in enumerate(update_state[k])])

    def add_results(self, **results):
        for k in self.state_vars.keys():
            vars(self)[k].extend(results[k])

    @property
    def get_state(self):
        return {k: vars(self)[k] for k in self.state_vars}

    @staticmethod
    def lineplot(data, x, y, **kwargs):
        return sb.lineplot(data=data, x=x, y=y, legend='brief', **kwargs)

    @staticmethod
    def histplot(data, x, y, **kwargs):
        return sb.histplot(data=data, x=y, stat="probability", legend=True, **kwargs)

    def save_state(self, path):
        data_keys = self.get_state.keys()
        data = pd.DataFrame(columns=MultipleResults.index_keys)
        for k in data_keys:
            data_k = pd.DataFrame(data=vars(self)[k])
            data = data.merge(data_k, how="outer", on=MultipleResults.index_keys)

        data.to_csv(f"{path}/{self.model_name}-state.csv", index_label='row')

    def plot_multiple_results(self, path, plot_fun=sb.lineplot, **figures):
        for fig in figures.keys():
            plt.figure()
            data = pd.DataFrame(columns=MultipleResults.index_keys)
            data_keys = figures[fig]['data'] if isinstance(figures[fig], dict) \
                        else figures[fig]
            plot_fun = figures[fig].get('plot_fun', plot_fun) \
                        if isinstance(figures[fig], dict) \
                        else plot_fun
            for k in data_keys:
                data_k = pd.DataFrame(data=vars(self)[k])
                plot_fun(data=data_k, x="round", y=k, label=self.state_vars[k])
                data = data.merge(data_k, how="outer", on=MultipleResults.index_keys)                # todo rounds are not preserved
            plt.title(fig)
            plt.legend()
            plt.savefig(f"{path}/{self.model_name}-{fig}.png", dpi=300)
            data.to_csv(f"{path}/{self.model_name}-{fig}.csv", index_label='row')