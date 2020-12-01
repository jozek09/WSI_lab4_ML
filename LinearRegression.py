### Wiktor Jóźwik 304008
from GetData import take_mpg_data_set, take_cycle_power_data_set
from model import linear_regression_model


def main():

    x1_train, x1_test, y1_train, y1_test = take_mpg_data_set()
    linear_regression_model(x1_train, x1_test, y1_train, y1_test, max_iteration=100, alpha=0.01, diff_ratio=0.0005)
    print()
    x2_train, x2_test, y2_train, y2_test = take_cycle_power_data_set()
    linear_regression_model(x2_train, x2_test, y2_train, y2_test, max_iteration=100, alpha=0.01, diff_ratio=0.0005)


if __name__ == "__main__":
    main()
