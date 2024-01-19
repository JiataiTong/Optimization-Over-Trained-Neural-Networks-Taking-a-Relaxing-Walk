from walk_with_constraints import *
from gurobi_MNIST_BisonNet import *
from store_network import *
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
timelimit = 60
width = 500
deltas = [5, 10]
n_epochs = 1500
lr = 0.18
rounds = 50

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



nn_model = torch.nn.Sequential(
    torch.nn.Linear(28 * 28, width),
    torch.nn.ReLU(),
    torch.nn.Linear(width, width),
    torch.nn.ReLU(),
    torch.nn.Linear(width, 10),
    # torch.nn.Softmax(1),
)

# Training settings
optimizer = torch.optim.SGD(nn_model.parameters(), lr=lr)


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

save_weights_biases_to_csv(nn_model, f'nn_{width}_{time_string}.csv')

nn_regression = nn_model

for delta in deltas:
    for num in range(rounds):
        random_image_idx = np.random.randint(0, len(mnist_train)+1)
        print(f'---random_image: {random_image_idx}---')
        ex_prob = nn_regression.forward(x_train[random_image_idx, :])
        sorted_labels = torch.argsort(ex_prob)
        right_label = sorted_labels[-1].item()
        wrong_label = sorted_labels[-2].item()
        image = x_train[random_image_idx, :].numpy()
        print('*Random Walk Start*')
        _, max_, _, _, _, _, _ = relaxation_walk_nmist(nn_regression, image, random_image_idx, wrong_label, right_label, delta, seed, pick_bias, timelimit, [test_accuracy, time_string])
        print(f'Random Walk: {max_}')
        print('!Gurobi Start!')
        _, max_, time_count = solve_with_gurobi_and_record(nn_regression, seed, image, random_image_idx, delta, ex_prob, wrong_label, right_label, timelimit, [test_accuracy, time_string])
        print(f'Gurobi: {max_} in {time_count} sec')

