import pandas as pd
from gurobi_benchmark import *
from relaxation_walk_debug import *
from multi_layer_relaxation_walk import *

seed_list = [53]
input_size_list = [10]
layer_num_list = [2]
layer_size_list = [500]
# timelimit = 3600 * 2
timelimit = 1200

walk_eps = 0.01
pick_bias = 0.05
gap = 2/3

rw_df_exist = True






if os.path.exists(f'RW_experiment_result_{timelimit}.csv'):
    rw_df = pd.read_csv(f'RW_experiment_result_{timelimit}.csv')
else:
    rw_df_exist = False
    rw_df = pd.DataFrame()


for input_size in input_size_list:
    for layer_num in layer_num_list:
        for layer_size in layer_size_list:
            for seed in seed_list:
                tag = str([input_size] + layer_num * [layer_size] + [1])

                print(f'[{input_size}, {layer_num} x {layer_size}, 1] with seed {seed} relaxation waking start')
                inHistory = False
                if rw_df_exist and rw_df['model_size'].isin([tag]).any():
                    results = rw_df.index[rw_df['model_size'] == tag].tolist()
                    for result in results:
                        if rw_df.loc[result, 'seed'] == seed and rw_df.loc[result, 'parameters'] == str([walk_eps, pick_bias]):
                            result = rw_df.loc[result, 'max_']
                            print(f'max: {result}')
                            print(f'-----------------------------------------')
                            inHistory = True
                            break
                    if not inHistory:
                        result = relaxation_walk(input_size, layer_num, layer_size, seed, walk_eps, pick_bias,
                                                 timelimit)
                        print(f'max: {result[1]}')
                        print(f'-----------------------------------------')
                else:
                    result = relaxation_walk(input_size, layer_num, layer_size, seed, walk_eps, pick_bias, timelimit)
                    print(f'max: {result[1]}')
                    print(f'-----------------------------------------')


