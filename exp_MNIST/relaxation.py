import gurobipy as gb
from gurobipy import GRB
import numpy as np
from mnist_util import *
import sys


#########################################################
def get_linear_relaxation(modelnn, image, wrong_label, right_label, delta, time_limit):
    w = get_weight_matrix(modelnn)
    b = get_bias_matrix(modelnn)
    in_size, layer_dims = get_model_info(modelnn)

    model = gb.Model()
    global x_vals
    global z_vals
    x_vals = []
    z_vals = []

    def get_relaxation(model, where):
        if where == GRB.Callback.MIPNODE:
            global x_vals
            global z_vals
            status = model.cbGet(GRB.Callback.MIPNODE_STATUS)
            if status == GRB.INTEGER:
                x_vals = model.cbGetSolution(x)
                z_vals = []
                for i in range(len(layer_dims)):
                    z_vals.append(model.cbGetNodeRel(z[i]))
                z_vals = model.cbGetSolution(z)
                model.terminate()
            if status == GRB.OPTIMAL:
                x_vals = model.cbGetNodeRel(x)
                z_vals = []
                for i in range(len(layer_dims)):
                    z_vals.append(model.cbGetNodeRel(z[i]))
                model.terminate()

    model.setParam('OutputFlag', 0)
    # model.setParam('OutputFlag', 1)
    model.setParam('TimeLimit', time_limit)
    x = model.addVars(in_size, ub=1, lb=0, name="input")
    abs_diff = model.addVars(in_size, ub=1, lb=0, name="abs_diff")
    z = {}
    h = {}
    g = {}
    hbar = {}

    count = 0
    for dim in layer_dims:
        z[count] = model.addVars(dim, vtype=GRB.BINARY, name="neuron_layer_" + str(count))
        h[count] = model.addVars(dim)
        hbar[count] = model.addVars(dim)
        g[count] = model.addVars(dim, lb=-GRB.INFINITY)
        count += 1

    for i in range(layer_dims[0]):
        model.addConstr(gb.quicksum(w[0][i][j] * x[j] for j in range(in_size)) + b[0][i] == g[0][i])
        model.addConstr((z[0][i] == 0) >> (h[0][i] == 0))
        model.addConstr((z[0][i] == 1) >> (h[0][i] == g[0][i]))
        model.addConstr((z[0][i] == 0) >> (g[0][i] <= 0))

    count = 1
    for input_dim, output_dim in zip(layer_dims, layer_dims[1:]):
        for i in range(output_dim):
            model.addConstr(
                gb.quicksum(w[count][i][j] * h[count - 1][j] for j in range(input_dim)) + b[count][i] == g[count][i])
            model.addConstr((z[count][i] == 0) >> (h[count][i] == 0))
            model.addConstr((z[count][i] == 1) >> (h[count][i] == g[count][i]))
            model.addConstr((z[count][i] == 0) >> (g[count][i] <= 0))
        count += 1

    # Bound on the distance to example in norm-1
    # model.addConstr(abs_diff >= x - image)
    # model.addConstr(abs_diff >= -x + image)
    # model.addConstr(abs_diff.sum() <= delta)
    for i in range(in_size):
        model.addConstr(abs_diff[i] >= x[i] - image[i])
        model.addConstr(abs_diff[i] >= -x[i] + image[i])
    model.addConstr(abs_diff.sum() <= delta)


    model.setObjective(g[count - 1][wrong_label] - g[count - 1][right_label], GRB.MAXIMIZE)

    model.optimize(get_relaxation)

    # print(model.status)

    if model.status == 9:
        return None, None

    z_vals = gbdict2lst_z(z_vals, layer_dims)
    return x_vals, z_vals
#########################################################


def get_linear_relaxation_with_restriction(modelnn, image, wrong_label, right_label, delta, time_limit,
                                           activation_pattern, restriction):
    w = get_weight_matrix(modelnn)
    b = get_bias_matrix(modelnn)
    in_size, layer_dims = get_model_info(modelnn)

    model = gb.Model()
    global x_vals
    global z_vals
    x_vals = []
    z_vals = []

    def get_relaxation(model, where):
        if where == GRB.Callback.MIPNODE:
            # print('---- In Callback ----')
            global x_vals
            global z_vals
            status = model.cbGet(GRB.Callback.MIPNODE_STATUS)
            # print(f'Call back status: {status}')
            if status == GRB.INTEGER:
                # print('integer')
                x_vals = model.cbGetSolution(x)
                z_vals = []
                for i in range(len(layer_dims)):
                    z_vals.append(model.cbGetNodeRel(z[i]))
                z_vals = model.cbGetSolution(z)
                # print('--- Callback ends ---')
                model.terminate()
            if status == GRB.OPTIMAL:
                # print('optimal')
                x_vals = model.cbGetNodeRel(x)
                z_vals = []
                for i in range(len(layer_dims)):
                    # print(model.cbGetNodeRel(z[i]))
                    z_vals.append(model.cbGetNodeRel(z[i]))
                # print('--- Callback ends ---')
                model.terminate()

    model.setParam('OutputFlag', 0)
    model.setParam('TimeLimit', time_limit)
    x = model.addVars(in_size, ub=1, lb=0, name="input")
    abs_diff = model.addVars(in_size, ub=1, lb=0, name="abs_diff")
    z = {}
    h = {}
    g = {}
    hbar = {}


    count = 0
    for dim in layer_dims:
        z[count] = model.addVars(dim, vtype=GRB.BINARY, name="neuron_layer_" + str(count))
        h[count] = model.addVars(dim)
        hbar[count] = model.addVars(dim)
        g[count] = model.addVars(dim, lb=-GRB.INFINITY)
        count += 1

    for i in range(layer_dims[0]):
        model.addConstr(gb.quicksum(w[0][i][j] * x[j] for j in range(in_size)) + b[0][i] == g[0][i])
        model.addConstr((z[0][i] == 0) >> (h[0][i] == 0))
        model.addConstr((z[0][i] == 1) >> (h[0][i] == g[0][i]))
        model.addConstr((z[0][i] == 0) >> (g[0][i] <= 0))
        if [0, i] in restriction:
            # print(f'set restriction [0, {i}]')
            model.addConstr(z[0][i] == (1 - activation_pattern[0][i]))

    count = 1
    for input_dim, output_dim in zip(layer_dims, layer_dims[1:]):
        for i in range(output_dim):
            model.addConstr(
                gb.quicksum(w[count][i][j] * h[count - 1][j] for j in range(input_dim)) + b[count][i] == g[count][i])
            model.addConstr((z[count][i] == 0) >> (h[count][i] == 0))
            model.addConstr((z[count][i] == 1) >> (h[count][i] == g[count][i]))
            model.addConstr((z[count][i] == 0) >> (g[count][i] <= 0))
            if [count, i] in restriction:
                # print(f'set restriction [{count}, {i}]')
                model.addConstr(z[count][i] == (1 - activation_pattern[count][i]))
        count += 1

    # Bound on the distance to example in norm-1
    # model.addConstr(abs_diff >= x - image)
    # model.addConstr(abs_diff >= -x + image)
    # model.addConstr(abs_diff.sum() <= delta)
    for i in range(in_size):
        model.addConstr(abs_diff[i] >= x[i] - image[i])
        model.addConstr(abs_diff[i] >= -x[i] + image[i])
    model.addConstr(abs_diff.sum() <= delta)

    model.setObjective(g[count - 1][wrong_label] - g[count - 1][right_label], GRB.MAXIMIZE)

    model.optimize(get_relaxation)

    status = model.status
    # print(status)
    # sys.stdout.flush()
    if status == GRB.Status.INF_OR_UNBD or status == GRB.Status.INFEASIBLE or len(z_vals) == 0:
        # print('model is infeasible')
        return None, None
    # print(z_vals)
    # sys.stdout.flush()

    z_vals = gbdict2lst_z(z_vals, layer_dims)
    return x_vals, z_vals
