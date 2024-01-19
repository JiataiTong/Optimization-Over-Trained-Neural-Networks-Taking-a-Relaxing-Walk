import torch
import csv


def save_weights_biases_to_csv(model, filename):
    """
    Save weights and biases from a torch.nn.Sequential model to a CSV file.

    Args:
    model (torch.nn.Sequential): The model from which to extract weights and biases.
    filename (str): The name of the CSV file where the data will be stored.
    """
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Layer Name', 'Layer Type', 'Weights', 'Biases'])

        for name, layer in model.named_children():
            layer_type = str(type(layer)).split('.')[-1][:-2]
            weights = getattr(layer, 'weight', None)
            biases = getattr(layer, 'bias', None)

            if weights is not None:
                # Flatten and convert weights to a list
                weights = weights.detach().cpu().numpy().flatten().tolist()
            else:
                weights = []

            if biases is not None:
                # Flatten and convert biases to a list
                biases = biases.detach().cpu().numpy().flatten().tolist()
            else:
                biases = []

            writer.writerow([name, layer_type, weights, biases])