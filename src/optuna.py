from copy import deepcopy
from datetime import datetime
from typing import Callable, Union

from src.train import run_downstream, run_pretext


def get_param_val(kwarg, val, trial) -> Union[float, int]:

    # type has to contain either "int" or "float" and can optionally contain "log" for logarithmic sampling
    t, lo, hi = val

    if "float" in t:
        return trial.suggest_float(kwarg, lo, hi, log=("log" in t))
    elif "int" in t:
        suggested_int = trial.suggest_int(kwarg, lo, hi, log=False)
        return suggested_int if "log" not in t else 2 ** suggested_int
    else:
        raise ValueError(f"The type of parameter to search over should be 'int' or 'float'. Type provided: {t}")


def create_optuna_objective(
        run_params: dict,
        save_models: bool = False,
) -> Callable:
    """
    Create the objective function used for the optuna hyperparameter search.
    """

    experiment_id = run_params["experiment_id"]
    current_run_params = deepcopy(run_params)

    def objective(trial):

        # fill current_run_params with sampled parameter values (bayesian optimization)
        for kwarg, val in run_params.items():
            if not (isinstance(val, tuple) or kwarg == "optimizer_kwargs"):
                continue
            if kwarg == "optimizer_kwargs":
                for optim_kwarg, optim_val in run_params["optimizer_kwargs"].items():
                    if isinstance(optim_val, tuple):
                        current_run_params["optimizer_kwargs"][optim_kwarg] = get_param_val(optim_kwarg, optim_val, trial)
            else:
                current_run_params[kwarg] = get_param_val(kwarg, val, trial)

        # modify experiment id to have unique id for each experiment run during optuna search
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        current_run_params["experiment_id"] = experiment_id + "_" + current_time

        # run trial
        if "pretext_model" in current_run_params:
            best_acc = run_downstream(**current_run_params, save_models=save_models)
        else:
            best_acc = run_pretext(**current_run_params, save_models=save_models)

        # return the accuracy this parameter combination achieved
        return best_acc

    return objective
