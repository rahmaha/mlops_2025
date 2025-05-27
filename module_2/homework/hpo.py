import os
import pickle
import click
import mlflow
import numpy as np
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll import scope
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from mlflow import log_metric, log_param, log_artifacts

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("random-forest-hyperopt")


def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


@click.command()
@click.option(
    "--data_path",
    default="D:\github\mlops_2025\module_2\data\output",
    help="Location where the processed NYC taxi trip data was saved"
)
@click.option(
    "--num_trials",
    default=15,
    help="The number of parameter evaluations for the optimizer to explore"
)
def run_optimization(data_path: str, num_trials: int):
    
    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

    def objective(params):
        with mlflow.start_run(nested=True):
            log_param('data_path', data_path)

            rf = RandomForestRegressor(**params)
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            
            log_metric('rmse', rmse)
            log_param('max_depth', params['max_depth'])
            log_param('n_estimators', params['n_estimators'])
            log_param('min_samples_split', params['min_samples_split'])
            log_param('min_samples_leaf', params['min_samples_leaf'])
            log_param('random_state', params['random_state'])

            return {'loss': rmse, 'status': STATUS_OK}
    

    search_space = {
        'max_depth': scope.int(hp.quniform('max_depth', 1, 20, 1)),
        'n_estimators': scope.int(hp.quniform('n_estimators', 10, 50, 1)),
        'min_samples_split': scope.int(hp.quniform('min_samples_split', 2, 10, 1)),
        'min_samples_leaf': scope.int(hp.quniform('min_samples_leaf', 1, 4, 1)),
        'random_state': 42
    }

    rstate = np.random.default_rng(42)  # for reproducible results
    with mlflow.start_run(run_name='random-forest-hyperopt'):
        fmin(
            fn=objective,
            space=search_space,
            algo=tpe.suggest,
            max_evals=num_trials,
            trials=Trials(),
            rstate=rstate
        )


if __name__ == '__main__':
    run_optimization()
