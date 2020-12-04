### Wiktor Jóźwik 304008
from GetData import take_mpg_data_set, take_cycle_power_data_set
from model import linear_regression_model
from ShowData import graph_metrics, print_best_r2


def main():
    # set default parameters
    k = 10
    max_iteration = 1000
    alpha = 0.1
    diff_ratio = 0.001

    # prepare data from first set
    x, y = take_mpg_data_set()
    y_name = y.name
    # train model
    acc, errors = linear_regression_model(x, y, max_iteration, alpha, diff_ratio, k)
    # graph metrics and show which portion of data is the best for training set
    graph_metrics(acc, y_name)
    print_best_r2(errors, y_name, k)

    # prepare data from second set
    x2, y2 = take_cycle_power_data_set()
    y2_name = y2.name
    # train model, k=5 because this set is much more bigger
    acc2, errors2 = linear_regression_model(x2, y2, max_iteration, alpha, diff_ratio, k=5)
    # graph metrics and show which portion of data is the best for training set
    graph_metrics(acc2, y2_name)
    print_best_r2(errors2, y2_name, k)


if __name__ == "__main__":
    main()
