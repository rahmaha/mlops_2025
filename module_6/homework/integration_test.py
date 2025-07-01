import batch
import pandas as pd
import os
from datetime import datetime


def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)


s3_endpoint_url = os.getenv('S3_ENDPOINT_URL')
options = {
    'client_kwargs': {
        'endpoint_url': s3_endpoint_url
    }
}

data = [
    # no pickup and dropoff location, 9 minutes duration
    (None, None, dt(1, 1), dt(1, 10)),
    (1, 1, dt(1, 2), dt(1, 10)),            # 8 minutes duration (all valid)
    # no dropoff location, 59 seconds -> too short -> drop
    (1, None, dt(1, 2, 0), dt(1, 2, 59)),
    # 60 minutes and 1 second -> too long  -> drop
    (3, 4, dt(1, 2, 0), dt(2, 2, 1)),
]

columns = ['PULocationID', 'DOLocationID',
           'tpep_pickup_datetime', 'tpep_dropoff_datetime']
df_input = pd.DataFrame(data, columns=columns)

input_file = batch.get_input_path(2023, 1)
output_file = batch.get_output_path(2023, 1)

df_input.to_parquet(
    input_file,
    engine='pyarrow',
    compression=None,
    index=False,
    storage_options=options
)

os.system('python batch.py 2023 1')


df_actual = pd.read_parquet(output_file, storage_options=options)
print('Predicted duration sum: ', df_actual['predicted_duration'].sum())
