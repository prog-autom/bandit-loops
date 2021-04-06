import inspect
from functools import partial

from joblib import Parallel, delayed
from mldev.experiment import experiment_tag
import os
from sklearn.model_selection import ParameterGrid
from results import MultipleResults


@experiment_tag()
class ExploreParams(object):

    def __init__(self, *args, **kwargs):
        self.grid = kwargs.get('grid')
        self.pipeline = kwargs.get('pipeline')
        self.run_times = kwargs.get('run_times', 1)
        self.results = kwargs.get('results', [])
        self.folder = kwargs.get('folder', None)
        self.random_seed = int(kwargs.get('random_seed', 0))

    @staticmethod
    def init_pipeline_grid(pipeline):
        # todo this should be generic enough for different setups
        states = dict()
        bandit_model = pipeline['bandit_model']
        bandit_name = pipeline['bandit_name']
        experiment = pipeline['experiment']
        init_random_state = pipeline['init_random_state']
        states['bandit_model'] = [[partial(ExploreParams.fetch_model, model_gen=bandit_model)]]
        states['bandit_name'] = [[bandit_name]]
        states['experiment'] = [[partial(ExploreParams.fetch_invoke, f=experiment)]]
        states['init_random_state'] = [[partial(ExploreParams.fetch_invoke, f=init_random_state)]]

        return ParameterGrid(states)

    @staticmethod
    def get_relevant_params(func, params):
        sig = inspect.signature(func, follow_wrapped=False)
        return {k: params.get(k, sig.parameters[k].default) for k in sig.parameters}

    @staticmethod
    def invoke_if_defined(obj, func_name, params):
        if hasattr(obj, func_name):
            f = getattr(obj, func_name)
            if callable(f):
                f(**ExploreParams.get_relevant_params(f, params))

    @staticmethod
    def fetch_invoke(f, state):
        #f_ = f() # get rid of partial when state is known
        f_ = f
        return f_(**ExploreParams.get_relevant_params(f_, state))

    @staticmethod
    def fetch_model(model_gen, state):
        return ExploreParams.fetch_invoke(model_gen, state)

    @staticmethod
    def process_trial(state, param_names, results, random_seed):
        state['init_random_state'](state={'seed':random_seed})

        model = state['bandit_model']
        state['bandit_model'] = lambda: model(state=state)
        mye = state['experiment'](state=state)

        ExploreParams.invoke_if_defined(mye, 'prepare', state)

        ExploreParams.invoke_if_defined(mye, 'run_experiment', state)

        params = {k: state[k] for k in param_names}

        mye_local = MultipleResults(state['bandit_name'], params=params, **results)
        mye_local.add_state(trial=state['trial'], **vars(mye))

        return mye_local

    def __call__(self, *args, **kwargs):

        pipeline_grid = ExploreParams.init_pipeline_grid(pipeline=self.pipeline)

        for code in pipeline_grid:
            states = dict(self.grid)
            states.update(code)
            states['trial'] = list(range(self.run_times))

            param_grid = ParameterGrid(states)
            results = Parallel(n_jobs=-1, verbose=10)(
                delayed(ExploreParams.process_trial)
                    (
                        state=state,
                        param_names=list(self.grid.keys()),
                        results=dict(self.results),
                        random_seed=seed
                    )
                   for state, seed in list(zip(param_grid,
                                               range(self.random_seed,
                                                     self.random_seed + len(param_grid))))
            )

            if self.folder is not None:
                mye_results = MultipleResults(code['bandit_name'][0],
                                              params=self.grid, # rewrite MultipleResults
                                              **self.results)

                for mye_result in results:
                    mye_results.add_results(param_names=mye_result.param_names,
                                            **mye_result.get_state)

                target_folder = f"{self.folder}"
                os.makedirs(target_folder, exist_ok=True)

                mye_results.save_state(f"{target_folder}")