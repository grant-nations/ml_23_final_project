from processing.preprocessing import encode_categorical, normalize, fill_missing_values
import os
from utils.general import generate_unique_filename
from utils.data_grabbers import get_raw_dataframes

data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed")

train_df, test_df = get_raw_dataframes()

for (df, data_type) in zip([train_df, test_df], ["train", "test"]):
    print(f"preprocessing {data_type} data...\n")

    # output the labels to a separate file
    if data_type == "train":
        labels = df.iloc[:, -1]

        processed_y_path = os.path.join(data_dir, f"train_labels.csv")
        processed_y_path = generate_unique_filename(processed_y_path)

        labels.to_csv(processed_y_path, header=None, index=False) # add header=None to get rid of headers in csv

    if data_type == "train":
        df = df.iloc[:, :-1] # remove labels from training data

    df = encode_categorical(df)
    df = fill_missing_values(df)
    df = normalize(df)

    print("sanity check:")
    print(df.head())

    processed_x_path = os.path.join(data_dir, f"{data_type}_input.csv")
    processed_x_path = generate_unique_filename(processed_x_path)

    df.to_csv(processed_x_path, header=None, index=False)  # add header=None to get rid of headers in csv
