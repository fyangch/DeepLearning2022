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

    def objective(trial):

        # add trial kwargs to RUN_DOWNSTREAM_PARAMS
        for kwarg, val in RUN_PARAMS.items():
            if not (isinstance(val, tuple) or kwarg == "optimizer_kwargs"):
                continue
            print(kwarg)
            if kwarg == "optimizer_kwargs":
                print("optimkwarg")
                for optim_kwarg, optim_val in RUN_PARAMS["optimizer_kwargs"].items():
                    if isinstance(optim_val, tuple):
                        t, lo, hi = optim_val
                        # optimizer_kwargs are always float
                        RUN_PARAMS["optimizer_kwargs"][optim_kwarg] = trial.suggest_float(optim_kwarg, lo, hi, log=('log' in t))

            else:
                # type, lower_bound, upper_bound
                t, lo, hi = val

                if 'int' in t:
                    if 'log' in t:
                        RUN_PARAMS[kwarg] = 2 ** trial.suggest_int(kwarg, lo, hi, log=False)
                    else:
                        RUN_PARAMS[kwarg] = trial.suggest_int(kwarg, lo, hi, log=False)
                elif 'float' in t:
                    RUN_PARAMS[kwarg] = trial.suggest_float(kwarg, lo, hi, log=('log' in t))
                else:
                    raise ValueError(f'Type of keyword argument `{kwarg}` must be either "int" or "float".')

        # modify experiment id to have unique id for each experiment run during optuna search
        current_time = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
        RUN_PARAMS["experiment_id"] = experiment_id + "_" + current_time

        # run trial
        if "pretext_model" in RUN_PARAMS:
            best_acc = run_downstream(**RUN_PARAMS)
        else:
            best_acc = run_pretext(**RUN_PARAMS)

        return best_acc

    return objective