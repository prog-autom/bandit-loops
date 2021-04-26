import seaborn as sb
from matplotlib import pyplot as plt
import pandas as pd

# TODO must also collect errors

class MultipleResults:
    index_keys = ['round', 'trial']

    def __init__(self, model_name, params={}, **initial_state):
        self.model_name = model_name
        self.params = params
        self.param_names = set(params.keys())
        self.state_vars = initial_state
        for k, v in initial_state.items():
            vars(self)[k] = list()

    def add_state(self, trial=None, **update_state):
        for k in self.state_vars.keys():
            vars(self)[k].extend([{'round':i, 'trial':trial, **self.params, k:j} for i, j in enumerate(update_state[k])])

    def add_results(self, param_names=[], **results):
        for k in self.state_vars.keys():
            vars(self)[k].extend(results[k])
            self.param_names.update(param_names)

    @property
    def get_state(self):
        return {k: vars(self)[k] for k in self.state_vars}

    @staticmethod
    def lineplot(data, x, y, **kwargs):
        return sb.lineplot(data=data, x=x, y=y, legend='brief', **kwargs)

    @staticmethod
    def scatterplot(data, x, y, **kwargs):
        return sb.scatterplot(data=data, x=x, y=y, legend='brief', **kwargs)

    @staticmethod
    def histplot(data, x, y, **kwargs):
        return sb.histplot(data=data, x=y, stat="probability", legend=True, **kwargs)

    def save_state(self, path):
        data_keys = self.get_state.keys()
        index_keys = set(MultipleResults.index_keys + list(self.param_names))
        data = pd.DataFrame(columns=index_keys)
        for k in data_keys:
            data_k = pd.DataFrame(data=vars(self)[k])
            data = data.merge(data_k, how="outer", on=list(index_keys))

        data.to_csv(f"{path}/{self.model_name}-state.csv", index_label='row')
        data.to_json(f"{path}/{self.model_name}-state.json", orient="records", lines=True)
        data.to_parquet(f"{path}/{self.model_name}-state.parquet", index=True)

    def plot_multiple_results(self, path, plot_fun=sb.lineplot, **figures):
        for fig in figures.keys():
            plt.figure()
            index_keys = set(MultipleResults.index_keys + list(self.param_names))
            data = pd.DataFrame(columns=index_keys)
            data_keys = figures[fig]['data'] if isinstance(figures[fig], dict) \
                        else figures[fig]
            plot_fun = figures[fig].get('plot_fun', plot_fun) \
                        if isinstance(figures[fig], dict) \
                        else plot_fun
            for k in data_keys:
                data_k = pd.DataFrame(data=vars(self)[k])
                if hasattr(data_k[k][0], '__len__') and len(data_k[k][0]) > 0:
                    for i in range(len(data_k[k][0])):
                        name = f"{k}-{i}"
                        data_k[name] = list(x[k][i] for x in vars(self)[k])
                        plot_fun(data=data_k[['round', name]], x="round", y=name, label=str.format(self.state_vars[k], i))
                else:
                    plot_fun(data=data_k, x="round", y=k, label=self.state_vars[k])
                data = data.merge(data_k, how="outer", on=list(index_keys))
            plt.title(fig)
            plt.legend()
            plt.savefig(f"{path}/{self.model_name}-{fig}.png", dpi=300)
            data.to_csv(f"{path}/{self.model_name}-{fig}.csv", index_label='row')