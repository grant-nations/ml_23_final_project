from utils.data_grabbers import get_feature_values
import pandas as pd
import numpy as np

def encode_categorical(df):
    features = get_feature_values()
    
    for feature_name, feature_values in features.items():
        if isinstance(feature_values, list):
            encoding_vec = list(range(len(feature_values)))
            df[feature_name].replace(feature_values, encoding_vec, inplace=True)

    # sanity check:
    print("Categorical feature encoding sanity check:")
    print(df.head())

    return df

def fill_missing_values(df):
    df = df.replace('?', np.nan)
    df.fillna(df.mean(), inplace=True)

    # sanity check:
    print("Empty value filling sanity check:")
    print(df.head())

    return df

def normalize(df):
    X = df.values # shape (25000, 14)
    X -= np.mean(X, axis = 0) # zero-center the data
    X /= np.std(X, axis = 0) # normalize the data
    return pd.DataFrame(X)

