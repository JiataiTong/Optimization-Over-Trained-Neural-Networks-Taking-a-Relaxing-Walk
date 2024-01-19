import numpy as np
from matplotlib import pyplot as plt

import torch
import torchvision
from skorch import NeuralNetClassifier

# Get MNIST digit recognition data set
mnist_train = torchvision.datasets.MNIST(root="./MNIST", train=True, download=True)
mnist_test = torchvision.datasets.MNIST(root="./MNIST", train=False, download=True)

x_train = torch.flatten(mnist_train.data.type(torch.FloatTensor), start_dim=1)
y_train = mnist_train.targets
x_test = torch.flatten(mnist_test.data.type(torch.FloatTensor), start_dim=1)
y_test = mnist_test.targets

x_train /= 255.0  # scaling
x_test /= 255.0  # scaling

for i in [1000]:
    nn_model = torch.nn.Sequential(
        torch.nn.Linear(28 * 28, i),
        torch.nn.ReLU(),
        torch.nn.Linear(i, i),
        torch.nn.ReLU(),
        torch.nn.Linear(i, 10),
        torch.nn.Softmax(1),
    )
    clf = NeuralNetClassifier(
        nn_model,
        max_epochs=40,
        lr=0.1,
        iterator_train__shuffle=True,
    )

    clf.fit(X=x_train, y=y_train)
    print(f'Network Size: {i} * 2')
    print(f"Training score: {clf.score(x_train, y_train):.4}")
    print(f"Validation set score: {clf.score(x_test, y_test):.4}")