from walk_with_constraints import *
from gurobi_MNIST import *

import numpy as np
from matplotlib import pyplot as plt

import torch
import torchvision
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import random
from datetime import datetime

seed = 50
pick_bias = 0.05
timelimit = 3600
delta = 20

np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)

current_time = datetime.now()
time_string = current_time.strftime("%Y-%m-%d_%H-%M-%S")


# Get MNIST digit recognition data set
mnist_train = torchvision.datasets.MNIST(root="./MNIST", train=True, download=True)
mnist_test = torchvision.datasets.MNIST(root="./MNIST", train=False, download=True)

x_train = torch.flatten(mnist_train.data.type(torch.FloatTensor), start_dim=1)
y_train = mnist_train.targets
x_test = torch.flatten(mnist_test.data.type(torch.FloatTensor), start_dim=1)
y_test = mnist_test.targets


x_train /= 255.0  # scaling
x_test /= 255.0  # scaling

width = 500

nn_model = torch.nn.Sequential(
    torch.nn.Linear(28 * 28, width),
    torch.nn.ReLU(),
    torch.nn.Linear(width, width),
    torch.nn.ReLU(),
    torch.nn.Linear(width, 10),
    # torch.nn.Softmax(1),
)

# Training settings
optimizer = torch.optim.SGD(nn_model.parameters(), lr=0.18)
n_epochs = 1500

# Lists to track loss and accuracy
training_losses = []
test_losses = []
test_accuracies = []

# Modify the training loop to track training loss
for epoch in range(n_epochs):
    # optimizer.zero_grad()
    # output = nn_model(x_train)
    # loss = F.cross_entropy(output, y_train)
    # loss.backward()
    # optimizer.step()

    nn_model.train()  # Set the model to training mode
    optimizer.zero_grad()
    output = nn_model(x_train)
    loss = F.cross_entropy(output, y_train)
    loss.backward()
    optimizer.step()


    training_loss = loss.item()
    training_losses.append(training_loss)
    print(f"Epoch {epoch+1}, Training Loss: {training_loss}")

    # Evaluate on test data
    # def test(model, x_test, y_test):
    #     model.eval()
    #     test_loss = 0
    #     correct = 0
    #     with torch.no_grad():
    #         output = model(x_test)
    #         test_loss = F.cross_entropy(output, y_test, reduction='sum').item()
    #         pred = output.argmax(dim=1, keepdim=True)
    #         correct = pred.eq(y_test.view_as(pred)).sum().item()
    #
    #     test_loss /= len(x_test)
    #     test_accuracy = 100. * correct / len(x_test)
    #     return test_loss, test_accuracy

    def test(model, x_test, y_test):
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            output = model(x_test)  # Again, these are raw logits
            test_loss = F.cross_entropy(output, y_test, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)  # Calculate the predicted class using argmax
            correct += pred.eq(y_test.view_as(pred)).sum().item()
        test_loss /= len(x_test)
        test_accuracy = 100. * correct / len(x_test)
        return test_loss, test_accuracy

    test_loss, test_accuracy = test(nn_model, x_test, y_test)
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)
    print(f'Test Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.2f}%')

# Plotting
plt.figure(figsize=(12, 5))

# Plot training loss
plt.subplot(1, 2, 1)
plt.plot(training_losses, label='Training Loss')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot test loss and accuracy
plt.subplot(1, 2, 2)
plt.plot(test_losses, label='Test Loss')
plt.plot(test_accuracies, label='Test Accuracy', linestyle='--')
plt.title('Test Loss and Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Loss/Accuracy')
plt.legend()

plt.tight_layout()
plt.show()