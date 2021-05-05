import inspect
from functools import partial
from itertools import filterfalse

from joblib import Parallel, delayed
from mldev.experiment import experiment_tag
import os
from sklearn.model_selection import ParameterGrid
from results import MultipleResults


@experiment_tag()
class ExploreParams(object):
    """
    This class runs the specified pipeline in a BanditLoopExperiment over the parameters grid.

    Example usage in yaml:
    .. code-block:: yaml

        grid_analysis: &grid_analysis !ExploreParams
          grid:
            p: [1.0]
            w: [0, 1, 2]
          results:
            interest: "User interests"
            loop_amp: '\abs(\mu^t - \mu^0)'
          pipeline:
            bandit_model: !function src/experiment.get_ts_model
            bandit_name: 'ts_model'
            experiment: !function src/experiment.BanditLoopExperiment
            init_random_state: !function src/experiment.init_random_state
            grid_filter: !function src/experiment.skip_params
          folder: "${env.TARGETFOLDER}"
          run_times: 1
          random_seed: "${env.RANDOMSEED}"

    """
    # todo consider using python-constraint to filter out irrelevant parameter combinations

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
        grid_filter = pipeline.get('grid_filter', bool)
        init_interest = pipeline.get('init_interest')

        states['bandit_model'] = [[partial(ExploreParams.fetch_model, model_gen=bandit_model)]]
        states['bandit_name'] = [[bandit_name]]
        states['experiment'] = [[partial(ExploreParams.fetch_invoke, f=experiment)]]
        states['init_random_state'] = [[partial(ExploreParams.fetch_invoke, f=init_random_state)]]
        states['grid_filter'] = [[partial(ExploreParams.fetch_invoke, grid_filter)]]
        states['init_interest'] = [[partial(ExploreParams.fetch_invoke, f=init_interest)]]

        return ParameterGrid(states)

    @staticmethod
    def get_relevant_params(func, params):
        """
        Returns a dict of params that have the corresponding arguments in the function func

        :param func:
        :param params:
        :return:
        """
        sig = inspect.signature(func, follow_wrapped=False)
        return {k: params.get(k, sig.parameters[k].default) for k in sig.parameters}

    @staticmethod
    def invoke_if_defined(obj, func_name, params):
        """
        Invoke the specificed function func_name if defined in the object obj
        and provide it arguments params

        :param obj:
        :param func_name:
        :param params:
        :return:
        """
        if hasattr(obj, func_name):
            f = getattr(obj, func_name)
            if callable(f):
                f(**ExploreParams.get_relevant_params(f, params))

    @staticmethod
    def fetch_invoke(f, state):
        """
        Calls f and passes acceptable parameters from state as named arguments

        :param f:
        :param state:
        :return:
        """

        #f_ = f() # get rid of partial when state is known
        f_ = f
        return f_(**ExploreParams.get_relevant_params(f_, state))

    @staticmethod
    def fetch_model(model_gen, state):
        return ExploreParams.fetch_invoke(model_gen, state)

    @staticmethod
    def process_trial(state, param_names, results, random_seed):
        """
        Runs a single trial of the experiment

        :param state: current parameters and pipeline combination
        :type state: dict
        :param param_names: names of the parameters (but not pipelines)
        :type param_names: dict
        :param results: a dict/set with attribute names that should be added to the trial results
        :param random_seed: the random seed to set
        :return: gathered results for the trial
        :rtype: MultipleResults
        """
        params = {k: state[k] for k in param_names}
        mye_local = MultipleResults(state['bandit_name'], params=params, **results)

        state['init_random_state'](state={'seed':random_seed})

        model = state['bandit_model']
        init_interest = state['init_interest']
        state['bandit_model'] = lambda: model(state=state)
        state['init_interest'] = lambda: init_interest(state=state)

        mye = state['experiment'](state=state)

        ExploreParams.invoke_if_defined(mye, 'prepare', state)

        ExploreParams.invoke_if_defined(mye, 'run_experiment', state)

        mye_local.add_state(trial=state['trial'], **vars(mye))

        return mye_local


    def __call__(self, *args, **kwargs):
        """
        Invoked by mldev. Parameters are not used

        Runs the experiment on parameter and pipeline grid.
        Skips parameters if grid_filter is True.

        Then saves results to the specified folder.

        :return: None
        """

        # create a dummy pipeline grid
        # theoretically there can be many combinations of pipelines
        pipeline_grid = ExploreParams.init_pipeline_grid(pipeline=self.pipeline)

        for code in pipeline_grid:

            # prepare possible states for a fixed pipeline
            states = dict(self.grid)
            states.update(code)
            states['trial'] = list(range(self.run_times))

            # grid_filter is needed to check for consistency in params
            # alternative could be to throw a ValueError if params are incompatible
            grid_filter = code['grid_filter'][0]

            # gives a parameter grid to explore in parallel
            # note that pipeline is fixed already
            param_grid = ParameterGrid(states)

            # uses param_grid as a queue to fetch jobs from
            # and executes the jobs using many workers (n_jobs=-1 - means =n_cpu)
            results = Parallel(n_jobs=-1, verbose=10)(
                delayed(ExploreParams.process_trial)
                    (
                        state=state,
                        param_names=list(self.grid.keys()),
                        results=dict(self.results),
                        random_seed=seed
                    )
                   # give each trial its own random seed
                   # see also https://joblib.readthedocs.io/en/latest/auto_examples/parallel_random_state.html
                   # and filter param_grid using the grid_filter function
                   for state, seed in list(zip(filterfalse(grid_filter, param_grid),
                                               range(self.random_seed,
                                                     self.random_seed + len(param_grid))))
            )

            if self.folder is not None:
                # save collected results into the folder
                mye_results = MultipleResults(code['bandit_name'][0],
                                              params=self.grid, # rewrite MultipleResults
                                              **self.results)

                for mye_result in results:
                    mye_results.add_results(param_names=mye_result.param_names,
                                            **mye_result.get_state)

                target_folder = f"{self.folder}"
                os.makedirs(target_folder, exist_ok=True)

                mye_results.save_state(f"{target_folder}")