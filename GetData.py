import numpy as np
import pandas as pd


def take_mpg_data_set():
    # read data using pandas library
    mpg = pd.read_csv("auto-mpg.data", delim_whitespace=True)
    mpg.columns = ["Miles Per Gallon", "Cylinders", "Displacement", "Horsepower", "Weight",
                   "Acceleration", "Model Year", "Origin", "Car Name"]
    # get data to work on
    data = mpg[["Miles Per Gallon", "Cylinders", "Displacement", "Horsepower", "Weight",
                "Acceleration", "Model Year"]]

    # get rid of NaN values(or in this example ? values)
    replaced = data.replace('?', np.nan)
    dropped_nan = replaced.dropna()

    # get x and y
    y = dropped_nan.iloc[:, 0]
    x = dropped_nan.iloc[:, 1:]

    return x, y


def take_cycle_power_data_set():
    # read data using pandas library
    power = pd.read_excel("Folds5x2_pp.xlsx")
    power.columns = ["Temperature", "Exhaust Vacuum", "Ambient Pressure", "Relative Humidity", "Electrical Energy"]

    # get x and y
    y = power.iloc[:, -1]
    x = power.iloc[:, :-1]

    return x, y
