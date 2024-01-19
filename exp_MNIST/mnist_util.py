import torch
import numpy as np

def get_model_info(model):
    input_size = None
    layer_dims = []

    for layer in model:
        if isinstance(layer, torch.nn.Linear):
            if input_size is None:
                input_size = layer.in_features  # Get input size from the first Linear layer
            layer_dims.append(layer.out_features)  # Get output size for each Linear layer

    return input_size, layer_dims


def get_weight_matrix(seq_model):
    weight_matrices = {}
    count = 0
    for i, layer in enumerate(seq_model):
        if isinstance(layer, torch.nn.Linear):
            weight_matrices[count] = layer.weight.data
            count += 1
    return weight_matrices


def get_bias_matrix(seq_model):
    bias_vectors = {}
    count = 0
    for i, layer in enumerate(seq_model):
        if isinstance(layer, torch.nn.Linear):
            bias_vectors[count] = layer.bias.data
            count += 1
    return bias_vectors


def gbdict2lst(dic, layer_dims):
    lst = []
    count = 0
    if isinstance(layer_dims, list) == False:
        for i in range(layer_dims):
            # if dic[i].getAttr('X') is None:
            #     return None
            lst.append(dic[i].x)
        return lst
    else:
        for dim in layer_dims:
            temp = []
            for i in range(dim):
                temp.append(dic[count][i].x)
            count += 1
            lst.append(temp)
        return lst


def gbdict2lst_z(dic, layer_dims):
    # print(layer_dims)
    # print(dic)
    lst = []
    count = 0
    if isinstance(layer_dims, list) == False:
        for i in range(layer_dims):
            # if dic[i].getAttr('X') is None:
            #     return None
            lst.append(dic[i])
        return lst
    else:
        for dim in layer_dims:
            temp = []
            for i in range(dim):
                temp.append(dic[count][i])
            count += 1
            lst.append(temp)
        return lst


def get_binary_activations(model_nn, x):
    x = list(x.values())

    weights = get_weight_matrix(model_nn).values()
    biases = get_bias_matrix(model_nn).values()

    h = torch.tensor(x)
    binary_activations = []

    for W, b in zip(weights, biases):
        # print(b)
        g = torch.matmul(h, W.T) + b
        h = torch.relu(g)  # ReLU activation
        binary_activation = (h > 0).int()  # Convert to binary activation
        # print(binary_activation)
        binary_activations.append(binary_activation.tolist())
        h = h * binary_activation  # Apply binary activation to next layer input

    return binary_activations




def get_prob_list_with_bias(int_z, frac_z, bias):
    prob_list = []
    for zi, zf in zip(int_z, frac_z):
        diff = abs(zi - zf)
        if len(prob_list) == 0:
            prob_list.append(diff + bias)
        else:
            prob_list.append(diff + prob_list[-1] + bias)
    return prob_list


def update_ap_random(g_vals, ap):
    for i in range(len(g_vals)):
        for j in range(len(g_vals[i])):
            if g_vals[i][j] == 0:
                ap[i][j] = 1 - np.random.choice([0, 1])
    return ap


def get_index_from_prob_list(prob_list, num):
    for i in range(len(prob_list)):
        if num <= prob_list[i]:
            return i