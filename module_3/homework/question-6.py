import pandas as pd
import mlflow
import joblib
import os
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from prefect import flow, task, get_run_logger


@task
def read_data(filename: str) -> pd.DataFrame:
    """Read data into DataFrame"""
    df = pd.read_parquet(filename)

    df.tpep_dropoff_datetime = pd.to_datetime(df.tpep_dropoff_datetime)
    df.tpep_pickup_datetime = pd.to_datetime(df.tpep_pickup_datetime)

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ["PULocationID", "DOLocationID"]
    df[categorical] = df[categorical].astype(str)

    return df


mlflow.set_tracking_uri("sqlite:///mlruns.db")
mlflow.set_experiment("Taxi Trip Model")

@task
def train_model(df: pd.DataFrame):
    logger = get_run_logger()
    feature_dicts = df[['PULocationID', 'DOLocationID']].to_dict(orient='records')

    vec = DictVectorizer(sparse=True)
    X = vec.fit_transform(feature_dicts)
    y_train = df['duration'].values

    model = LinearRegression()
    model.fit(X, y_train)

    with mlflow.start_run() as run:
        # Save model locally
        model_path = "model.joblib"
        joblib.dump(model, model_path)

        # Log manually
        mlflow.log_artifact(model_path, artifact_path="model")

        # Log metadata
        mlflow.log_param("intercept", model.intercept_)
        mlflow.log_param("features", list(vec.get_feature_names_out()))

        # Log model size
        model_size = os.path.getsize(model_path) 
        mlflow.log_metric("model_size", model_size)

        logger.info(f"Model logged with run_id: {run.info.run_id}")
        logger.info(f"Model size: {model_size:.2f} ")

    return model.intercept_

@flow
def main_flow():
    file_path = r'D:\github\mlops_2025\module_3\homework\data\yellow_tripdata_2023-03.parquet'

    # Read data
    data = read_data(file_path)

    # Train model using both pickup and dropoff locations
    intercept = train_model(data)

    print(f"Final Model Intercept: {intercept}")


if __name__ == "__main__":
    main_flow()
