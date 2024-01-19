from walk_with_constraints import *
from gurobi_MNIST import *

import numpy as np
from matplotlib import pyplot as plt

import torch
import torchvision
from skorch import NeuralNetClassifier

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

nn_model = torch.nn.Sequential(
    torch.nn.Linear(28 * 28, 500),
    torch.nn.ReLU(),
    torch.nn.Linear(500, 500),
    torch.nn.ReLU(),
    torch.nn.Linear(500, 10),
    torch.nn.Softmax(1),
)
clf = NeuralNetClassifier(
    nn_model,
    max_epochs=35,
    lr=0.1,
    iterator_train__shuffle=True,
)

clf.fit(X=x_train, y=y_train)
print(f'Network Size: {50} * 2')
training_score = clf.score(x_train, y_train)
validation_score = clf.score(x_test, y_test)
print(f"Training score: {training_score:.4}")
print(f"Validation set score: {validation_score:.4}")

nn_regression = torch.nn.Sequential(*nn_model[:-1])

for num in range(50):
    random_image_idx = np.random.randint(0, len(mnist_train)+1)
    print(f'---random_image: {random_image_idx}---')
    ex_prob = nn_regression.forward(x_train[random_image_idx, :])
    sorted_labels = torch.argsort(ex_prob)
    right_label = sorted_labels[-1].item()
    wrong_label = sorted_labels[-2].item()
    image = x_train[random_image_idx, :].numpy()
    print('*Random Walk Start*')
    _, max_, _, _, _, _, _ = relaxation_walk_nmist(nn_regression, image, random_image_idx, wrong_label, right_label, delta, seed, pick_bias, timelimit, [training_score, validation_score, time_string])
    print(f'Random Walk: {max_}')
    print('!Gurobi Start!')
    _, max_, time_count = solve_with_gurobi_and_record(nn_regression, seed, image, random_image_idx, delta, ex_prob, wrong_label, right_label, timelimit, [training_score, validation_score, time_string])
    print(f'Gurobi: {max_} in {time_count} sec')

