import pandas as pd
from gurobi_benchmark import *
from main import *
from multi_layer_relaxation_walk import *
from alter_walk import *

seed_list = [50, 51, 52, 53, 54]
# seed_list = [50]
input_size_list = [10, 100, 1000]
layer_num_list = [1, 2, 3]
layer_size_list = [100, 500]
timelimit = 3600

walk_eps = 0.01
pick_bias = 0.05
gap = 2/3

rw_df_exist = True
sw_df_exist = True
# rwd_df_exist = True
rrw_df_exist = True



if os.path.exists(f'RW_experiment_result_{timelimit}.csv'):
    rw_df = pd.read_csv(f'RW_experiment_result_{timelimit}.csv')
else:
    rw_df_exist = False
    rw_df = pd.DataFrame()
# print(rw_df.to_string())
if os.path.exists(f'SW_experiment_result_{timelimit}.csv'):
    sw_df = pd.read_csv(f'SW_experiment_result_{timelimit}.csv')
else:
    sw_df_exist = False
    sw_df = pd.DataFrame()

if os.path.exists(f'RRW_experiment_result_{timelimit}.csv'):
    rrw_df = pd.read_csv(f'RRW_experiment_result_{timelimit}.csv')
else:
    rrw_df_exist = False
    rrw_df = pd.DataFrame()

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
                        if rw_df.loc[result, 'seed'] == seed and rw_df.loc[result, 'parameters'] == str(
                                [walk_eps, pick_bias]):
                            result = rw_df.loc[result, 'max_']
                            print(f'max: {result}')
                            print(f'-----------------------------------------')
                            inHistory = True
                            break
                    if not inHistory:
                        result = relaxation_walk_deep(input_size, layer_num, layer_size, seed, walk_eps, pick_bias,
                                                 timelimit)
                        print(f'max: {result[1]}')
                        print(f'-----------------------------------------')
                else:
                    result = relaxation_walk_deep(input_size, layer_num, layer_size, seed, walk_eps, pick_bias, timelimit)
                    print(f'max: {result[1]}')
                    print(f'-----------------------------------------')

                # print(tag)
                print(
                    f'[{input_size}, {layer_num} x {layer_size}, 1] with seed {seed} relaxation random walking start')
                inHistory = False
                if rrw_df_exist and rrw_df['model_size'].isin([tag]).any():
                    results = rrw_df.index[rrw_df['model_size'] == tag].tolist()
                    for result in results:
                        if rrw_df.loc[result, 'seed'] == seed and rrw_df.loc[result, 'parameters'] == str([pick_bias]):
                            result = rrw_df.loc[result, 'max_']
                            print(f'max: {result}')
                            print(f'-----------------------------------------')
                            inHistory = True
                            break
                    if not inHistory:
                        result = relaxation_random_walk(input_size, layer_num, layer_size, seed, pick_bias, timelimit)
                        print(f'max: {result[1]}')
                        print(f'-----------------------------------------')
                else:
                    result = relaxation_random_walk(input_size, layer_num, layer_size, seed, pick_bias, timelimit)
                    print(f'max: {result[1]}')
                    print(f'-----------------------------------------')

                print(f'[{input_size}, {layer_num} x {layer_size}, 1] with seed {seed} sampling walking start')
                inHistory = False
                if sw_df_exist and sw_df['model_size'].isin([tag]).any():
                    results = sw_df.index[sw_df['model_size'] == tag].tolist()
                    for result in results:
                        # print(sm_df.loc[result, 'parameters'])
                        if sw_df.loc[result, 'seed'] == seed and sw_df.loc[result, 'parameters'] == str([walk_eps, pick_bias]):
                            result = sw_df.loc[result, 'max_']
                            print(f'max: {result}')
                            print(f'-----------------------------------------')
                            inHistory = True
                            break
                    if not inHistory:
                        result = sampling_walk(input_size, layer_num, layer_size, seed, walk_eps, pick_bias, timelimit)
                        print(f'max: {result[1]}')
                        print(f'-----------------------------------------')
                else:
                    result = sampling_walk(input_size, layer_num, layer_size, seed, walk_eps, pick_bias, timelimit)
                    print(f'max: {result[1]}')
                    print(f'-----------------------------------------')
