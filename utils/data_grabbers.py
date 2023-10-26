import json
import os
import pandas as pd

def get_feature_values():
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')

    features = {}

    with open(os.path.join(data_dir, "features.json")) as f:
        features = json.load(f)
    
    return features

def get_raw_train_dataframe():
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    return pd.read_csv(os.path.join(data_dir, "raw", "train_final.csv"))

def get_raw_test_dataframe():
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    return pd.read_csv(os.path.join(data_dir, "raw", "test_final.csv"))

def get_raw_dataframes():
    return get_raw_train_dataframe(), get_raw_test_dataframe()