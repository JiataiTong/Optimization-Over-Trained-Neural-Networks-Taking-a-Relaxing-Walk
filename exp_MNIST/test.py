import torch
import torchvision

def get_model_info(model):
    input_size = None
    layer_dims = []

    for layer in model:
        if isinstance(layer, torch.nn.Linear):
            if input_size is None:
                input_size = layer.in_features  # Get input size from the first Linear layer
            layer_dims.append(layer.out_features)  # Get output size for each Linear layer

    return input_size, layer_dims

# # Example usage
# model_nn = torch.nn.Sequential(
#     torch.nn.Linear(28 * 28, 50),
#     torch.nn.ReLU(),
#     torch.nn.Linear(50, 50),
#     torch.nn.ReLU(),
#     torch.nn.Linear(50, 10),
#     torch.nn.Softmax(1),
# )
#
# input_size, layer_dims = get_model_info(model_nn)
# print("Input Size:", input_size)
# print("Layer Dimensions:", layer_dims)


mnist_train = torchvision.datasets.MNIST(root="./MNIST", train=True, download=True)
print(len(mnist_train))
# Get the first image and its label
image, label = mnist_train[0]

# Print the shape of the image
print("Shape of a single image:", image.size)
print("label:", label)
