import pandas as pd
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

@task
def train_model(df: pd.DataFrame) -> float:
    logger = get_run_logger()

    # Convert both categorical columns into feature dictionaries
    feature_dicts = df[['PULocationID', 'DOLocationID']].to_dict(orient='records')

    # Fit a DictVectorizer
    vec = DictVectorizer(sparse=True)
    X = vec.fit_transform(feature_dicts)
    y_train = df['duration'].values

    # Train a linear regression model
    model = LinearRegression()
    model.fit(X, y_train)

    # Log and return the intercept
    logger.info(f"Intercept of the model: {model.intercept_}")
    return vec, model

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