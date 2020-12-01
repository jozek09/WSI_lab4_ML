import numpy as np
from metrics import mean_squared_error, mean_absolute_error, r_square

def check_difference(diff_w, diff_b, diff_ratio):  # Function to check difference between j+1 and j set of parameters
    diff_sum = 0
    for d in diff_w:
        diff_sum += d
    diff_sum += diff_b

    return diff_sum < diff_ratio


def calculate_coefficients_with_gradient(x_train, y_train, max_iteration, alpha, diff_ratio):
    w = np.zeros(x_train.shape[1])  # Initializing array of parameter w
    b = 0  # Parameter b
    cost_array = np.zeros(max_iteration)

    x = np.array(x_train)
    y = np.array(y_train)

    m = len(y)

    for iteration in range(max_iteration):

        w_gradient = np.zeros(x.shape[1])
        b_gradient = 0

        c = 0
        for i in range(m):
            y_cal = np.dot(w, x[i]) + b
            cost = (y[i] - y_cal) ** 2
            w_gradient = w_gradient + (-2) * x[i] * (y[i] - y_cal)
            b_gradient = b_gradient + (-2) * (y[i] - y_cal)

            c += cost

        # Calculating new w and difference between current w and new w
        old_w = w
        old_b = b
        w = w - alpha * w_gradient / m
        b = b - alpha * b_gradient / m
        diff_w = np.abs(old_w - w)
        diff_b = np.abs(old_b - b)

        if check_difference(diff_w, diff_b, diff_ratio):
            print(f"Last iteration of {y_train.name}: {iteration}")
            break

        cost_array[iteration] = c / (2 * m)

    return w, b, cost_array


def linear_regression_model(x_train, x_test, y_train, y_test, max_iteration, alpha, diff_ratio):
    w, b, cost = calculate_coefficients_with_gradient(x_train, y_train, max_iteration, alpha, diff_ratio)

    calculate_metrics_of_model(x_test, y_test, w, b)


def calculate_metrics_of_model(x_test, y_test, w, b):
    y_calc = np.dot(x_test, w) + b

    mse = mean_squared_error(y_test, y_calc)
    mae = mean_absolute_error(y_test, y_calc)
    r2 = r_square(y_test, y_calc)

    print(f"Model for {y_test.name} prediction")
    print(f"Mean squared error(MSE): {mse}")
    print(f"Mean absolute error(MAE): {mae}")
    print(f"R square: {r2}")
