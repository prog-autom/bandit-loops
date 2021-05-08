import argparse
import json
import os

from experiment import *
from results import MultipleResults

from joblib.parallel import Parallel, delayed

parser = None
folder = None
run_times = None
random_seed = None


def create_model(model_params):
    if 'ts_model' in model_params:
        model_name = 'ts_model'
        model = lambda : get_ts_model(**model_params['ts_model'])
        print(f"Using model: {model_name}")
        yield model, model_name
    elif 'random_model' in model_params:
        model_name = 'random_model'
        model = lambda : get_random_model(**model_params['random_model'])
        print(f"Using model: {model_name}")
        yield model, model_name
    elif 'optimal_model' in model_params:
        model_name = 'optimal'
        model = lambda : get_optimal_model(**model_params['random_model'])
        print(f"Using model: {model_name}")
        yield model, model_name
    elif 'epsilon_greedy_model' in model_params:
        model_name = 'epsilon_greedy_model'
        model = lambda : get_epsilonn_greedy_model(**model_params['random_model'])
        print(f"Using model: {model_name}")
        yield model, model_name

def bandit_loop(model_params, params):
    print(f"Running bandit-loop experiment")
    for model, model_name in create_model(model_params):
        ble_results = MultipleResults(model_name, **BanditLoopExperiment.default_state)

        def process_trial(trial):
            ble_local = MultipleResults(model_name, **BanditLoopExperiment.default_state)
            ble = BanditLoopExperiment(model, model_name)

            prepare_params = {k: params[k] for k in params.keys() & {'w', 'Q', 'p', 'b'}}
            ble.prepare(**prepare_params)

            loop_params = {k: params[k] for k in params.keys() & {'T'}}
            ble.run_experiment(**loop_params)

            ble_local.add_state(trial=trial, **vars(ble))

            return ble_local

        results = Parallel(n_jobs=-1, verbose=10)(delayed(process_trial)(trial) for trial in range(0, run_times))
        for ble_result in results:
            ble_results.add_results(**ble_result.get_state)

        target_folder = f"{folder}"
        os.makedirs(target_folder, exist_ok=True)
        ble_results.plot_multiple_results(target_folder, **BanditLoopExperiment.default_figures)
        ble_results.save_state(f"{target_folder}")


def init_random(random_seed):
    return init_random_state(random_seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", type=str, help="Kind of experiment: bandit-loop")
    parser.add_argument("--params", type=str, help="A json string with experiment parameters")
    parser.add_argument("--model_params", type=str, help="A json string with model name and parameters")
    parser.add_argument("--folder", type=str, help="Save results to this folder", default="./results")
    parser.add_argument("--random_seed", type=int, help="Use the provided value to init the random state", default=42)
    parser.add_argument("--run_times", type=int, help="How many time to repeat the trial", default=1)
    args = parser.parse_args()
    model_str = args.model_params
    params_str = args.params
    kind = args.kind

    folder = args.folder
    random_seed = args.random_seed
    run_times = args.run_times
    os.makedirs(folder, exist_ok=True)

    model_dict = json.loads(model_str)
    params_dict = json.loads(params_str)

    init_random_state(random_seed)

    if kind == "bandit-loop":
        bandit_loop(model_dict, params_dict)
    else:
        parser.error("Unknown experiment kind: " + kind)
