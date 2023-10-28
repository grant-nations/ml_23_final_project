import torch
import numpy as np

def train(dataloader, model, loss_fn, optimizer, device, print_every=100):

    iterations_per_epoch = len(dataloader)
    print(f"Iterations per epoch: {iterations_per_epoch}")

    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % print_every == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def validate(dataloader, model, loss_fn, device, dataset_name="Validation"):
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
    print(f"{dataset_name} Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {validation_loss:>8f} \n")
    
    return correct

def evaluate(dataloader, model, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for X in dataloader:
            X = X.to(device)
            pred = model(X)
            pred = torch.round(pred)
            predictions.append(pred.numpy().item())
    
    return np.array(predictions)