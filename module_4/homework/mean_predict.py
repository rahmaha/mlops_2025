#!/usr/bin/env python
# coding: utf-8

import pickle
import pandas as pd
import sys

with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)


categorical = ['PULocationID', 'DOLocationID']


def read_data(filename):
    df = pd.read_parquet(filename)

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')

    return df


def get_file(year, month):
    # year = int(sys.argv[1])  # 2023
    # month = int(sys.argv[2])  # 4
    # template url
    url_template = 'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    # format the url
    input_file = url_template.format(year=year, month=month)

    return input_file


def apply_model(input_file):
    df = read_data(input_file)
    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)
    mean_pred = y_pred.mean()

    print('Mean predicted duration: ', mean_pred)

def run():
    year = int(sys.argv[1])  # 2023
    month = int(sys.argv[2])  # 4

    input_file = get_file(year, month)
    apply_model(input_file)
    

if __name__ == '__main__':
    run()

