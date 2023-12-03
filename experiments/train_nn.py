from classifiers.neural_net.nn import BinClassificationNN
from classifiers.neural_net.dataset import IncomeDataset
from torch.utils.data import DataLoader
import os
import torch
from classifiers.neural_net.train import train, validate
from utils.general import generate_unique_filename

# TODO: use k-fold cross validation

MODEL_SAVE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "classifiers", "neural_net", "saved_models")
MODEL_NAME = "nn_final"

############# HYPERPARAMETERS #############
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

split_data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "splits")

# get training data
training_data = IncomeDataset(input_filepath=os.path.join(split_data_dir, "train_split_input.csv"),
                              labels_filepath=os.path.join(split_data_dir, "train_split_labels.csv"))
train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True)

# get validation data
validation_data = IncomeDataset(input_filepath=os.path.join(split_data_dir, "validate_split_input.csv"),
                                labels_filepath=os.path.join(split_data_dir, "validate_split_labels.csv"))
val_dataloader = DataLoader(validation_data)

# create model
model = BinClassificationNN(dropout_p=DROPOUT_P).to(device)
print(model)
loss_fn = torch.nn.BCELoss()  # binary cross entropy loss
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# train model
train(train_dataloader, val_dataloader, model, loss_fn, optimizer, device, num_epochs=NUM_EPOCHS, patience=5, min_delta=0.001, print_every=1)

# print training accuracy
# validate(train_dataloader, model, loss_fn, device, dataset_name="Train")

# print validation accuracy
# pred_acc = validate(val_dataloader, model, loss_fn, device, dataset_name="Validation")

# train on entire training set

# model_state = model.state_dict()

# model_save_path = os.path.join(MODEL_SAVE_DIR, MODEL_NAME)
# model_save_path = generate_unique_filename(model_save_path)
# torch.save(model_state, os.path.join(MODEL_SAVE_DIR, MODEL_NAME))

# split_data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "splits")

# test_data = IncomeDataset(input_filepath=os.path.join(split_data_dir, "test_split_input.csv"),
#                           labels_filepath=os.path.join(split_data_dir, "test_split_labels.csv"))

# test_dataloader = DataLoader(test_data)


# validate(test_dataloader, model, loss_fn, device, "Test")
