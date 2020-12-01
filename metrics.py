import numpy as np


def mean_squared_error(y, y_calc):
    n = len(y)
    sum_of_squares = np.sum((y - y_calc) ** 2)
    return sum_of_squares / n


def mean_absolute_error(y, y_calc):
    n = len(y)
    sum_of_abs = np.sum(np.abs(y - y_calc))
    return sum_of_abs / n


def r_square(y, y_calc):
    ss_total = np.sum((y - y.mean()) ** 2)
    ss_regression = np.sum((y - y_calc) ** 2)

    r2 = 1 - (ss_regression / ss_total)
    return r2
