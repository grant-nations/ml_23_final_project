from classifiers.neural_net.nn import BinClassificationNN
from classifiers.neural_net.dataset import IncomeDataset
from torch.utils.data import DataLoader
import os
import torch
from classifiers.neural_net.train import train, validate
from utils.general import generate_unique_filename
import pandas as pd

MODEL_SAVE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "classifiers", "neural_net", "saved_models")
MODEL_NAME = "nn_final"

############# HYPERPARAMETERS #############
K_FOLDS = 5
BATCH_SIZE = 64
NUM_EPOCHS = 100
LEARNING_RATE = 3.5e-3
WEIGHT_DECAY = 1e-5
DROPOUT_P = 0.25
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
input_dataframe = pd.read_csv(os.path.join(split_data_dir, "train_input.csv"))
labels_dataframe = pd.read_csv(os.path.join(split_data_dir, "train_labels.csv"))

training_data = IncomeDataset(input_dataframe=input_dataframe,
                              labels_dataframe=labels_dataframe)

# generate k folds
k_folds = training_data.generate_k_folds(K_FOLDS)

total_correct = 0
total_samples = 0

avg_epochs_elapsed = 0

for k in range(K_FOLDS):

    print(f"\n=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=\nFold {k}\n=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=\n")

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
    train_dataloader = DataLoader(IncomeDataset(input_dataframe=train_X, labels_dataframe=train_y),
                                  batch_size=BATCH_SIZE,
                                  shuffle=True)

    val_dataloader = DataLoader(IncomeDataset(input_dataframe=val_X, labels_dataframe=val_y))

    # create model
    model = BinClassificationNN(dropout_p=DROPOUT_P).to(device)
    print(model)
    loss_fn = torch.nn.BCELoss()  # binary cross entropy loss
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # train model
    avg_epochs_elapsed += train(train_dataloader, val_dataloader, model, loss_fn, optimizer, device,
                                num_epochs=NUM_EPOCHS, patience=5, min_delta=0.001, print_every=1)

    # get validation accuracy
    val_loss, correct_pct = validate(val_dataloader, model, loss_fn, device)

    total_correct += correct_pct * len(val_X)
    total_samples += len(val_X)

    print(f"Validation loss for fold {k}: {val_loss:.3f}")
    print(f"Validation accuracy for fold {k}: {correct_pct:.3f}")

print(f"Average validation accuracy: {total_correct / total_samples:.3f}")

# train on entire training set
train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True)

model = BinClassificationNN(dropout_p=DROPOUT_P).to(device)
print(model)

loss_fn = torch.nn.BCELoss()  # binary cross entropy loss
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

avg_epochs_elapsed //= K_FOLDS
print(f"Training on entire training set for {avg_epochs_elapsed} epochs")

train(train_dataloader, None, model, loss_fn, optimizer, device, num_epochs=avg_epochs_elapsed)

# save model
model_save_path = os.path.join(MODEL_SAVE_DIR, MODEL_NAME)
model_save_path = generate_unique_filename(model_save_path)
torch.save(model.state_dict(), model_save_path)
print(f"Saved model to {model_save_path}")
