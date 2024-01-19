from util import *
from ap_record import *


def single_walk(model_nn, x, eps):
    # print(x)
    start = time.time()
    max_ = -1000
    x_max = None
    step_count = 0
    ap = get_binary_activations(model_nn, x)
    max_lp, x_new = solve_lp(model_nn, ap)
    while max_lp > max_:
        # print(max_lp)
        step_count += 1
        max_ = max_lp
        x_max = x_new
        x = update_x(model_nn, x, x_new, eps)
        ap = get_binary_activations(model_nn, x)
        max_lp, x_new = solve_lp(model_nn, ap)
    time_consuming = time.time() - start
    return x_max, max_, step_count, time_consuming


def single_walk_with_timelimit(model_nn, x, eps, timelimit):
    # print(x)
    start = time.time()
    max_ = -1000
    x_max = None
    step_count = 0
    ap = get_binary_activations(model_nn, x)
    max_lp, x_new = solve_lp_pre_calc(model_nn, ap)
    while max_lp > max_:
        # print(max_lp)
        step_count += 1
        max_ = max_lp
        x_max = x_new
        now = time.time()
        if now - start >= timelimit:
            break
        x = update_x(model_nn, x, x_new, eps)
        ap = get_binary_activations(model_nn, x)
        max_lp, x_new = solve_lp_pre_calc(model_nn, ap)
    time_consuming = time.time() - start
    return x_max, max_, step_count, time_consuming


def single_walk_with_timelimit_and_record(model_nn, x, eps, timelimit, record_ap):
    # print(x)
    start = time.time()
    max_ = -1000
    x_max = None
    step_count = 0
    ap = get_binary_activations(model_nn, x)
    if record_ap.check_and_record(ap):
        time_consuming = time.time() - start
        print(f'walk break, takes {time_consuming}')
        return x_max, max_, step_count, time_consuming, record_ap
    record_ap.check_and_record(ap)
    max_lp, x_new = solve_lp(model_nn, ap)
    while max_lp > max_:
        # print(max_lp)
        step_count += 1
        max_ = max_lp
        x_max = x_new
        now = time.time()
        if now - start >= timelimit:
            break
        x = update_x(model_nn, x, x_new, eps)
        ap = get_binary_activations(model_nn, x)
        if record_ap.check_and_record(ap):
            time_consuming = time.time() - start
            print(f'walk break, takes {time_consuming} - {record_ap.record_num}')
            return x_max, max_, step_count, time_consuming, record_ap
        max_lp, x_new = solve_lp(model_nn, ap)
    time_consuming = time.time() - start
    print(f'walk complete, takes {time_consuming} - {record_ap.record_num}')
    return x_max, max_, step_count, time_consuming, record_ap


def single_random_walk_with_timelimit(model_nn, x, timelimit):
    # print(x)
    start = time.time()
    max_ = -1000
    x_max = None
    step_count = 0
    ap = get_binary_activations(model_nn, x)
    max_lp, x_new, g = solve_lp_pre_calc_with_g(model_nn, ap)
    while max_lp > max_:
        # print(max_lp)
        step_count += 1
        max_ = max_lp
        x_max = x_new
        now = time.time()
        if now - start >= timelimit:
            break
        ap = update_ap_random(g, ap)
        max_lp, x_new, g = solve_lp_pre_calc_with_g(model_nn, ap)
    time_consuming = time.time() - start
    return x_max, max_, step_count, time_consuming
