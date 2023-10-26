from preprocessing.preprocess import encode_categorical, normalize, fill_missing_values, split
import os
from utils.general import generate_unique_filename
from utils.data_grabbers import get_raw_dataframes

data_dir = os.path.join(os.path.dirname(
    os.path.dirname(__file__)), "data")

train_df, test_df = get_raw_dataframes()

for (df, data_type) in zip([train_df, test_df], ["train", "test"]):
    print(f"preprocessing {data_type} data...\n")

    # output the labels to a separate file
    if data_type == "train":
        labels = df.iloc[:, -1]

        processed_y_path = os.path.join(
            data_dir, "processed", f"train_labels.csv")
        processed_y_path = generate_unique_filename(processed_y_path)

        # add header=None to get rid of headers in csv
        labels.to_csv(processed_y_path, header=None, index=False)

        train_split_y_path = os.path.join(data_dir, "splits", "train_split_labels.csv")
        train_split_y_path = generate_unique_filename(train_split_y_path)

        validate_split_y_path = os.path.join(data_dir, "splits", "validate_split_labels.csv")
        validate_split_y_path = generate_unique_filename(validate_split_y_path)

        test_split_y_path = os.path.join(data_dir, "splits", "test_split_labels.csv")
        test_split_y_path = generate_unique_filename(test_split_y_path)

        train_split_y, validate_split_y, test_split_y = split(
            labels, train_pct=0.6, validate_pct=0.2)  # test_pct = 0.2
        
        train_split_y.to_csv(train_split_y_path, header=None, index=False)
        validate_split_y.to_csv(validate_split_y_path, header=None, index=False)
        test_split_y.to_csv(test_split_y_path, header=None, index=False)

        df = df.iloc[:, :-1]  # remove labels from training data

    if data_type == "test":
        # remove the id column from the test data
        df = df.iloc[:, 1:]

    df = encode_categorical(df)
    df = fill_missing_values(df)
    df = normalize(df)

    print("sanity check:")
    print(df.head())

    processed_x_path = os.path.join(
        data_dir, "processed", f"{data_type}_input.csv")
    processed_x_path = generate_unique_filename(processed_x_path)

    # add header=None to get rid of headers in csv
    df.to_csv(processed_x_path, header=None, index=False)

    if data_type == "train":
        # split training data

        train_split_x_path = os.path.join(data_dir, "splits", "train_split_input.csv")
        train_split_x_path = generate_unique_filename(train_split_x_path)

        validate_split_x_path = os.path.join(data_dir, "splits", "validate_split_input.csv")
        validate_split_x_path = generate_unique_filename(validate_split_x_path)

        test_split_x_path = os.path.join(data_dir, "splits", "test_split_input.csv")
        test_split_x_path = generate_unique_filename(test_split_x_path)

        train_split_x, validate_split_x, test_split_x = split(
            df, train_pct=0.6, validate_pct=0.2)  # test_pct = 0.2
        
        train_split_x.to_csv(train_split_x_path, header=None, index=False)
        validate_split_x.to_csv(validate_split_x_path, header=None, index=False)
        test_split_x.to_csv(test_split_x_path, header=None, index=False)
