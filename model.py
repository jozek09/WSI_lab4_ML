import numpy as np
from metrics import mean_squared_error, mean_absolute_error, r_square
from sklearn.preprocessing import StandardScaler
from ShowData import print_metrics


def check_difference(diff_w, diff_b, diff_ratio):  # Function to check difference between j+1 and j set of parameters
    diff_sum = 0
    for d in diff_w:
        diff_sum += d
    diff_sum += diff_b

    return diff_sum < diff_ratio


def calculate_coefficients_with_gradient(x, y, max_iteration, alpha, diff_ratio):
    w = np.zeros(x.shape[1])  # Initializing array of parameter w
    b = 0  # Parameter b
    cost_array = np.zeros(max_iteration)

    m = len(y)

    for iteration in range(max_iteration):
        w_gradient = np.zeros(x.shape[1])
        b_gradient = 0

        c = 0
        for i in range(m):
            y_cal = np.dot(w, x[i]) + b
            # calculate cost
            cost = (y[i] - y_cal) ** 2
            # calculate gradients
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

        # if difference is small enough we don't need to iterate to the end
        if check_difference(diff_w, diff_b, diff_ratio):
            break

        cost_array[iteration] = c / (2 * m)

    return w, b, cost_array


def get_train_and_test_sets(x, y, position, k):
    # scale X values for the next of the calculation
    sc = StandardScaler()
    x = sc.fit_transform(x)

    # get start, end positions and portion depending o k size
    portion = round(100 / k)
    start_position = position * portion
    end_position = start_position + portion

    # get train and test sets
    y_train = np.concatenate((np.array(y[:start_position]), np.array(y[end_position:])), axis=0)
    x_train = np.concatenate((np.array(x[:start_position]), np.array(x[end_position:])), axis=0)

    y_test = y[start_position:end_position]
    x_test = x[start_position:end_position]

    return x_train, x_test, y_train, y_test


def linear_regression_model(x, y, max_iteration, alpha, diff_ratio, k):
    r2_best = -np.inf
    # declare arrays for storing data
    mse_array = []
    mae_array = []
    r2_array = []
    # in k range, where k is cluster size
    for i in range(k):
        # get x, y train and test data
        x_train, x_test, y_train, y_test = get_train_and_test_sets(x, y, i, k)

        # calculate coefficients, store cost if needed
        w, b, cost = calculate_coefficients_with_gradient(x_train, y_train, max_iteration, alpha, diff_ratio)

        # calculate squares based on that particular coefficients
        mse, mae, r2 = calculate_metrics_of_model(x_test, y_test, w, b)

        # if current r2 is better than best r2, replace it and store mse, mae and i which is number of cluster
        if r2 > r2_best:
            best_k = i
            r2_best = r2
            mse_store = mse
            mae_store = mae

        mse_array.append(mse)
        mae_array.append(mae)
        r2_array.append(r2)

        # printing metrics(optional)
        # print_metrics(mse, mae, r2, y_test.name)

    # prepare data for return
    accuracy_array = [mse_array, mae_array, r2_array]
    best_errors = [mse_store, mae_store, r2_best, best_k]

    return accuracy_array, best_errors


def calculate_metrics_of_model(x_test, y_test, w, b):
    # calculate y using coefficients w and b
    y_calc = np.dot(x_test, w) + b

    # calculate mse, mae and r2
    mse = mean_squared_error(y_test, y_calc)
    mae = mean_absolute_error(y_test, y_calc)
    r2 = r_square(y_test, y_calc)

    return mse, mae, r2
