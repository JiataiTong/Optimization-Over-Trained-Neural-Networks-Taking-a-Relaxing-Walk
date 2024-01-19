import gurobipy as gb
from gurobipy import GRB
import numpy as np
import random
from Network import *
from util import *
from io_csv import *


def solve_model(modelnn, time_limit):
    start = time.time()
    w = modelnn.get_weight_matrix()
    b = modelnn.get_bias_matrix()
    layer_dims = modelnn.layer_dims
    in_size = modelnn.in_size

    model = gb.Model()

    model.setParam('TimeLimit', time_limit)
    model.setParam('OutputFlag', 1)
    x = model.addVars(in_size, ub=1, lb=0, name="input")
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

    model.setObjective(g[count - 1][0], GRB.MAXIMIZE)

    model.optimize()

    time_count = time.time() - start
    if model.status == 4:
        print(f'{in_size} x {layer_dims} is infeasible or unbounded')
        return None, None, time_count
    elif model.status == GRB.OPTIMAL:
        values = [x[i].x for i in range(len(x))]
        return values, g[count - 1][0].x, time_count
    elif model.SolCount > 0:
        model.setParam(GRB.Param.SolutionNumber, 0)
        output = model.PoolObjVal
        values = [x[i].Xn for i in range(len(x))]
        return values, output, time_count
    else:
        print(f'end with status: {model.status}')
        return None, None, time_count


def solve_with_gurobi(input_size, layer_num, layer_size, random_seed, time_limit):
    layer_dims = layer_num * [layer_size] + [1]
    model_nn = Network(in_size=input_size, layer_dims=layer_dims, seed=random_seed)

    x_max, max_, time_count = solve_model(model_nn, time_limit)
    # store_data(
    #     [['GUROBI', [input_size] + layer_dims, None, random_seed, x_max, max_, None, time_count,
    #       None, None, None]], f'Gurobi_Benchmark.csv')
    if max_ is None:
        store_data(
            [['GUROBI', [input_size] + layer_dims, None, random_seed, x_max, None,
              None, time_count,
              None, None, None]], f'Gurobi_Benchmark_{time_limit}.csv')
    else:
        store_data(
            [['GUROBI', [input_size] + layer_dims, None, random_seed, x_max, model_nn(torch.FloatTensor(x_max)).item(), None, time_count,
              None, None, None]], f'Gurobi_Benchmark_{time_limit}.csv')

    return x_max, max_, time_count

