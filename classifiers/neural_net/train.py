import torch
import numpy as np


def train(train_dataloader,
          validate_dataloader,
          model,
          loss_fn,
          optimizer,
          device,
          num_epochs,
          print_every=10,
          patience=1,
          min_delta=0.0):

    early_stopper = EarlyStopper(patience=patience, min_delta=min_delta)
    for t in range(num_epochs):
        print(f"\nEpoch {t+1}\n-------------------------------")
        train_loss = train_epoch(train_dataloader, model, loss_fn, optimizer, device)

        if validate_dataloader is not None:
            validation_loss, acc = validate(validate_dataloader, model, loss_fn, device)
            if early_stopper.early_stop(validation_loss):
                print(f"validation loss increased by {min_delta} or more for {patience} epochs")
                return t

        if t % print_every == 0:
            print(f"train loss: {train_loss:.3f}, validation loss: {validation_loss:.3f}")
            print(f"accuracy: {acc:.3f}")


    return num_epochs


def train_epoch(dataloader, model, loss_fn, optimizer, device):
    model.train()
    loss = 0

    for (X, y) in dataloader:
        X, y = X.to(device), y.to(device)

        # compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return loss.item()


def validate(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    validation_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            validation_loss += loss_fn(pred, y).item()
            pred = torch.round(pred)
            correct += (pred == y).type(torch.float).sum().item()

    validation_loss /= num_batches

    correct /= size
    return validation_loss, correct


def predict(dataloader, model, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for X in dataloader:
            X = X.to(device)
            pred = model(X)
            pred = torch.round(pred)
            predictions.append(pred.numpy().item())

    return np.array(predictions)


# below is taken from https://stackoverflow.com/a/73704579/14111683
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
