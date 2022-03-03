# This is a script to build and train a VIT model.

# Imports

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets.mnist import MNIST
from torchvision.transforms import ToTensor
from tqdm import tqdm

from MyViT import MyViT

# Set the seed

np.random.seed(0)
torch.manual_seed(0)

# CONSTANTS

BATCH_SIZE = 16
N_EPOCHS = 5
LR = 0.01

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    # Loading data
    transform = ToTensor()

    train_set = MNIST(root="./../datasets", train=True, download=True, transform=transform)
    test_set = MNIST(root="./../datasets", train=True, download=True, transform=transform)

    train_loader = DataLoader(train_set, shuffle=True, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_set, shuffle=True, batch_size=BATCH_SIZE)

    # Defining the model
    model = MyViT(
        input_shape=(1, 28, 28)
    )
    model.to(device)

    # Training loop
    optimizer = Adam(model.parameters(), lr=LR)
    criterion = CrossEntropyLoss().to(device)
    for epoch in range(N_EPOCHS):
        train_loss = 0.0
        for batch in tqdm(train_loader):
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            y_hat = model(x)
            loss = criterion(y_hat, y) / len(x)

            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{N_EPOCHS} loss: {train_loss:.2f}")

    # Test loop
    correct, total = 0, 0
    test_loss = 0.0
    for batch in test_loader:
        x, y = batch
        x = x.to(device)
        y = y.to(device)
        y_hat = model(x)
        loss = criterion(y_hat, y)
        test_loss += loss / len(x)

        correct = torch.sum(torch.argmax(y_hat, dim=1) == y).item()
        total += len(x)

    print(f"Test loss: {test_loss:.2f}")
    print(f"Test accuracy: {correct / total * 100:.2f}%")


if __name__ == '__main__':
    main()