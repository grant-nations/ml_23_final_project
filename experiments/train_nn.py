from classifiers.neural_net.nn import BinClassificationNN
from classifiers.neural_net.dataset import IncomeDataset
from torch.utils.data import DataLoader
import os
import torch
from classifiers.neural_net.train import train, validate
from utils.general import generate_unique_filename
import pandas as pd
import json

MODEL_SAVE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "classifiers", "neural_net", "saved_models")
MODEL_NAME = "nn_final"

json_save_path = os.path.join(MODEL_SAVE_DIR, f"{MODEL_NAME}_hyperparameters.json")
json_save_path = generate_unique_filename(json_save_path)

############# HYPERPARAMETERS #############
K_FOLDS = 5
NUM_EPOCHS = 100

BATCH_SIZES = [32, 64, 128]
LEARNING_RATES = [1e-3, 1e-4, 1e-5]
WEIGHT_DECAYS = [0.0, 1e-3, 1e-4]
DROPOUT_PROBS = [0.1, 0.2, 0.3]
HIDDEN_DIMS = [32, 64, 128]
HIDDEN_LAYERS = [1, 2]
BATCH_NORM = [True, False]

# for testing:
# BATCH_SIZES = [32]
# LEARNING_RATES = [1e-3]
# WEIGHT_DECAYS = [0.0]
# DROPOUT_PROBS = [0.1]
# HIDDEN_DIMS = [32]
# HIDDEN_LAYERS = [1]
# BATCH_NORM = [True]
############################################

# use GPU if available
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

split_data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed")

# get training data
input_dataframe = pd.read_csv(os.path.join(split_data_dir, "train_input.csv"), header=None)
labels_dataframe = pd.read_csv(os.path.join(split_data_dir, "train_labels.csv"), header=None)

training_data = IncomeDataset(input_dataframe=input_dataframe,
                              labels_dataframe=labels_dataframe)

# generate k folds
k_folds = training_data.generate_k_folds(K_FOLDS)

# perform search for best hyperparameters

best_hyperparameters = {
    "batch_size": None,
    "lr": None,
    "w_decay": None,
    "dropout_probs": None,
    "hidden_dims": None,
    "batch_norm": None,
    "hidden_layers": None,
    "avg_epochs_elapsed": None
}

best_accuracy = 0.0

for batch_size in BATCH_SIZES:
    print(
        f"\n=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=\nbatch_size: {batch_size}\n=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=\n")
    for lr in LEARNING_RATES:
        print(f"\n=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=\nlr: {lr}\n=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=\n")
        for w_decay in WEIGHT_DECAYS:
            print(
                f"\n=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=\nw_decay: {w_decay}\n=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=\n")
            for dropout_p in DROPOUT_PROBS:
                print(
                    f"\n=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=\ndropout_p: {dropout_p}\n=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=\n")
                for hidden_dim in HIDDEN_DIMS:
                    print(
                        f"\n=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=\nhidden_dim: {hidden_dim}\n=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=\n")
                    for batch_norm in BATCH_NORM:
                        print(
                            f"\n=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=\nbatch_norm: {batch_norm}\n=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=\n")
                        for hidden_layers in HIDDEN_LAYERS:
                            print(
                                f"\n=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=\nhidden_layers: {hidden_layers}\n=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=\n")

                            dropout_probs = [dropout_p] * hidden_layers
                            hidden_dims = [hidden_dim] * hidden_layers

                            print(f"\n=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=\n"
                                  f"batch_size: {batch_size}\n"
                                  f"lr: {lr}\n"
                                  f"w_decay: {w_decay}\n"
                                  f"dropout_probs: {dropout_probs}\n"
                                  f"hidden_dims: {hidden_dims}\n"
                                  f"batch_norm: {batch_norm}\n"
                                  f"hidden_layers: {hidden_layers}\n"
                                  f"=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=\n")

                            total_correct = 0
                            total_samples = 0

                            avg_epochs_elapsed = 0

                            for k in range(K_FOLDS):
                                # if k < 4:
                                #     continue

                                print(
                                    f"\n=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=\nFold {k}\n=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=\n")

                                # select kth fold as validation set
                                val_X, val_y = k_folds[k]

                                # select remaining folds as training set
                                train_X = pd.DataFrame()
                                train_y = pd.DataFrame()
                                for i in range(K_FOLDS):
                                    if i != k:
                                        train_X = pd.concat([train_X, k_folds[i][0]])
                                        train_y = pd.concat([train_y, k_folds[i][1]])

                                # create dataloaders
                                train_dataloader = DataLoader(
                                    IncomeDataset(input_dataframe=train_X, labels_dataframe=train_y),
                                    batch_size=batch_size, shuffle=True)

                                val_dataloader = DataLoader(IncomeDataset(
                                    input_dataframe=val_X, labels_dataframe=val_y))

                                # create model
                                model = BinClassificationNN(
                                    dropout_probs=dropout_probs,
                                    hidden_dims=hidden_dims,
                                    batch_norm=batch_norm
                                ).to(device)

                                print(model)
                                loss_fn = torch.nn.BCELoss()  # binary cross entropy loss
                                optimizer = torch.optim.Adam(
                                    model.parameters(),
                                    lr=lr, weight_decay=w_decay)

                                # train model
                                avg_epochs_elapsed += train(
                                    train_dataloader, val_dataloader, model, loss_fn, optimizer, device,
                                    num_epochs=NUM_EPOCHS, patience=2, min_delta=0.001, print_every=1)

                                # get validation accuracy
                                val_loss, correct_pct = validate(val_dataloader, model, loss_fn, device)

                                total_correct += correct_pct * len(val_X)
                                total_samples += len(val_X)

                                print(f"Validation loss for fold {k}: {val_loss:.3f}")
                                print(f"Validation accuracy for fold {k}: {correct_pct:.3f}")

                            avg_val_accuracy = total_correct / total_samples
                            print(f"Average validation accuracy: {avg_val_accuracy:.3f}")

                            avg_epochs_elapsed //= K_FOLDS

                            if avg_val_accuracy > best_accuracy:
                                best_accuracy = avg_val_accuracy
                                best_hyperparameters["batch_size"] = batch_size
                                best_hyperparameters["lr"] = lr
                                best_hyperparameters["w_decay"] = w_decay
                                best_hyperparameters["dropout_probs"] = dropout_probs
                                best_hyperparameters["hidden_dims"] = hidden_dims
                                best_hyperparameters["batch_norm"] = batch_norm
                                best_hyperparameters["hidden_layers"] = hidden_layers
                                best_hyperparameters["avg_epochs_elapsed"] = avg_epochs_elapsed

                                with open(json_save_path, "w") as f:
                                    json.dump(best_hyperparameters, f)

# train on entire training set
train_dataloader = DataLoader(training_data, batch_size=best_hyperparameters["batch_size"], shuffle=True)

model = BinClassificationNN(
    dropout_probs=best_hyperparameters["dropout_probs"],
    hidden_dims=best_hyperparameters["hidden_dims"],
    batch_norm=best_hyperparameters["batch_norm"]
).to(device)

print(model)

loss_fn = torch.nn.BCELoss()  # binary cross entropy loss
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=best_hyperparameters["lr"],
    weight_decay=best_hyperparameters["w_decay"])

train(train_dataloader, None, model, loss_fn, optimizer, device,
      num_epochs=best_hyperparameters["avg_epochs_elapsed"], print_every=1)

# save model
model_save_path = os.path.join(MODEL_SAVE_DIR, MODEL_NAME)
model_save_path = generate_unique_filename(model_save_path)
torch.save(model.state_dict(), model_save_path)
print(f"Saved model to {model_save_path}")
