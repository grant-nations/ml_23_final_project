from classifiers.neural_net.nn import BinClassificationNN
from classifiers.neural_net.dataset import IncomeDataset
from torch.utils.data import DataLoader
import os
from utils.cade_gpu import define_gpu_to_use
import torch
from classifiers.neural_net.train import train, validate, test

# TODO: use multifold cross validation to get a better estimate of the model's performance

CADE = False
MODEL_SAVE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "classifiers", "neural_net", "saved_models")
MODEL_NAME = "nn_1"

############# HYPERPARAMETERS #############
BATCH_SIZE = 64
NUM_EPOCHS = 10
LEARNING_RATE = 1e-3
############################################


split_data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "splits")

training_data = IncomeDataset(input_filepath=os.path.join(split_data_dir, "train_split_input.csv"),
                              labels_filepath=os.path.join(split_data_dir, "train_split_labels.csv"))

validation_data = IncomeDataset(input_filepath=os.path.join(split_data_dir, "validate_split_input.csv"),
                                labels_filepath=os.path.join(split_data_dir, "validate_split_labels.csv"))

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
val_dataloader = DataLoader(validation_data)

if CADE:
    define_gpu_to_use(4000) # 4GB of memory

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print(f"Using {device} device")

model = BinClassificationNN().to(device)
print(model)

loss_fn = torch.nn.BCELoss() # binary cross entropy loss
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

best_val_acc = 0
best_model_state = None
for t in range(NUM_EPOCHS):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer, device)
    pred_acc = validate(val_dataloader, model, loss_fn, device)

    if pred_acc > best_val_acc:
        best_val_acc = pred_acc
        best_model_state = model.state_dict()

torch.save(best_model_state, os.path.join(MODEL_SAVE_DIR, MODEL_NAME))

split_data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "splits")

test_data = IncomeDataset(input_filepath=os.path.join(split_data_dir, "test_split_input.csv"),
                          labels_filepath=os.path.join(split_data_dir, "test_split_labels.csv"))

test_dataloader = DataLoader(test_data)

print("Test set results:")

test(test_dataloader, model, loss_fn, device)