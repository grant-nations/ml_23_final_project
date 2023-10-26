import os
from classifiers.neural_net.dataset import IncomeDataset
from classifiers.neural_net.nn import BinClassificationNN
from torch.utils.data import DataLoader
from utils.cade_gpu import define_gpu_to_use
import torch
from classifiers.neural_net.train import evaluate
from utils.general import generate_unique_filename
import pandas as pd

CADE = False
MODEL_SAVE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "classifiers", "neural_net", "saved_models")
MODEL_NAME = "nn_1"
PREDICTIONS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "predictions")
PREDICTIONS_FILENAME = "predictions.csv"

if CADE:
    define_gpu_to_use(4000)  # 4GB of memory

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print(f"Using {device} device")

proc_data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed")

comp_test_data = IncomeDataset(input_filepath=os.path.join(proc_data_dir, "test_input.csv"))

test_dataloader = DataLoader(comp_test_data)

model = BinClassificationNN().to(device)
model.load_state_dict(torch.load(os.path.join(MODEL_SAVE_DIR, MODEL_NAME)))

predictions = evaluate(test_dataloader, model, device)

predictions_filepath = os.path.join(PREDICTIONS_DIR, PREDICTIONS_FILENAME)
predictions_filepath = generate_unique_filename(predictions_filepath)

df = pd.DataFrame(predictions, columns=["Prediction"])

ids = list(range(1, len(predictions) + 1))
df.insert(0, "ID", ids, False)

df.to_csv(predictions_filepath, index=False)