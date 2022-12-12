from copy import deepcopy
from datetime import datetime
from typing import Callable

from src.train import run_downstream, run_pretext


def create_optuna_objective(
        RUN_PARAMS: dict,
) -> Callable:
    """
    Create the objective function used for the optuna hyperparameter search.
    """

    experiment_id = RUN_PARAMS["experiment_id"]
    run_params_copy = deepcopy(RUN_PARAMS)

    def objective(trial):

        # add trial kwargs to RUN_DOWNSTREAM_PARAMS
        for kwarg, val in RUN_PARAMS.items():
            if not (isinstance(val, tuple) or kwarg == "optimizer_kwargs"):
                continue
            if kwarg == "optimizer_kwargs":
                for optim_kwarg, optim_val in RUN_PARAMS["optimizer_kwargs"].items():
                    if isinstance(optim_val, tuple):
                        t, lo, hi = optim_val
                        # optimizer_kwargs are always float
                        run_params_copy["optimizer_kwargs"][optim_kwarg] = trial.suggest_float(optim_kwarg, lo, hi, log=('log' in t))

            else:
                # type, lower_bound, upper_bound
                t, lo, hi = val
                if 'int' in t:
                    if 'log' in t:
                        run_params_copy[kwarg] = 2 ** trial.suggest_int(kwarg, lo, hi, log=False)
                    else:
                        run_params_copy[kwarg] = trial.suggest_int(kwarg, lo, hi, log=False)
                elif 'float' in t:
                    run_params_copy[kwarg] = trial.suggest_float(kwarg, lo, hi, log=('log' in t))
                else:
                    raise ValueError(f'Type of keyword argument `{kwarg}` must be either "int" or "float".')

        # modify experiment id to have unique id for each experiment run during optuna search
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_params_copy["experiment_id"] = experiment_id + "_" + current_time

        # run trial
        if "pretext_model" in run_params_copy:
            best_acc = run_downstream(**run_params_copy)
        else:
            best_acc = run_pretext(**run_params_copy)

        return best_acc

    return objective