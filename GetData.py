import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def get_train_and_test_sets(x, y, size):
    # scale X values for the next of the calculation
    sc = StandardScaler()
    x = sc.fit_transform(x)

    train_size = round(len(x) * size)

    # get train and test sets
    y_train = y[:train_size]
    y_test = y[train_size:]
    x_train = x[:train_size]
    x_test = x[train_size:]

    return x_train, x_test, y_train, y_test


def take_mpg_data_set():
    train_size = 0.75
    mpg = pd.read_csv("auto-mpg.data", delim_whitespace=True)
    mpg.columns = ["Miles Per Gallon", "Cylinders", "Displacement", "Horsepower", "Weight",
                   "Acceleration", "Model Year", "Origin", "Car Name"]
    data = mpg[["Miles Per Gallon", "Cylinders", "Displacement", "Horsepower", "Weight",
                "Acceleration", "Model Year", "Origin"]]
    # shuffle data
    data = data.sample(frac=1)

    # get rid of NaN values(or in this example ? values)
    replaced = data.replace('?', np.nan)
    dropped_nan = replaced.dropna()

    # get x and y
    y = dropped_nan.iloc[:, 0]
    x = dropped_nan.iloc[:, 1:]

    return get_train_and_test_sets(x, y, train_size)


def take_cycle_power_data_set():
    train_size = 0.75

    power = pd.read_excel("Folds5x2_pp.xlsx")
    power.columns = ["Temperature", "Exhaust Vacuum", "Ambient Pressure", "Relative Humidity", "Electrical Energy"]
    # shuffle data
    power = power.sample(frac=1)

    # get x and y
    y = power.iloc[:, -1]
    x = power.iloc[:, :-1]

    return get_train_and_test_sets(x, y, train_size)
