import batch
import pandas as pd
from datetime import datetime

# This function is to created a datetime object for testing
def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)

# Test the prepare_data function
def test_prepare_data():
    # create a DataFrame with sample data (fake data)
    data = [
        (None, None, dt(1, 1), dt(1, 10)),      # no pickup and dropoff location, 9 minutes duration
        (1, 1, dt(1, 2), dt(1, 10)),            # 8 minutes duration (all valid)
        (1, None, dt(1, 2, 0), dt(1, 2, 59)),   # no dropoff location, 59 seconds -> too short -> drop
        (3, 4, dt(1, 2, 0), dt(2, 2, 1)),]      # 60 minutes and 1 second -> too long  -> drop

    columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
    df = pd.DataFrame(data, columns=columns)

    # what we expect after prepare_data() called
    expected_data= pd.DataFrame({
        'PULocationID': ['-1', '1'],
        'DOLocationID': ['-1', '1'],
        'duration': [9.0, 8.0],
    }) # expect of the output DataFrame, result after trasformation

    # first run the prepare_data function 
    # only PULocationID and DOLocationID are categorical
    categorical = ['PULocationID', 'DOLocationID']
    actual = batch.prepare_data(df, categorical)
    # duration is calculated as total seconds

    # reset index to compare with expectation
    actual = actual[['PULocationID', 'DOLocationID', 'duration']].reset_index(drop=True)
    expected_data = expected_data.reset_index(drop=True)

    # compare actual and expected DataFrame
    assert actual.equals(expected_data), f'DataFrame mismatch: \n{actual}\n!=\n{expected_data}'

