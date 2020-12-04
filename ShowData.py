import numpy as np
import matplotlib.pyplot as plt


def graph_metrics(accuracy_array, name):
    mse = accuracy_array[0]
    mae = accuracy_array[1]
    r2 = accuracy_array[2]
    plt.plot(np.arange(len(mse)), mse)
    plt.xlabel('K-th cluster')
    plt.ylabel('Value of mean square error')
    plt.title(f'Mean square error for {name}')
    plt.savefig(f'{name}-MSE.png')
    plt.show()

    plt.plot(np.arange(len(mae)), mae)
    plt.xlabel('K-th cluster')
    plt.ylabel('Value of mean absolute error')
    plt.title(f'Mean absolute error for {name}')
    plt.savefig(f'{name}-MAE.png')
    plt.show()

    plt.plot(np.arange(len(r2)), r2)
    plt.xlabel('K-th cluster')
    plt.ylabel('Value of R square error')
    plt.title(f'R square for {name}')
    plt.savefig(f'{name}-R2.png')
    plt.show()


def print_best_r2(best_errors, name, k):
    print(f'K-fold cross validation, k = {k}')
    print(f"The best R2 error for {name} is {best_errors[2]}")
    print(f"It's for k = {best_errors[3]}")
    print(f"Then MSE is {best_errors[0]}")
    print(f"MAE is {best_errors[1]}")


def print_metrics(mse, mae, r2, name):
    print(f"Model for {name} prediction")
    print(f"Mean squared error(MSE): {mse}")
    print(f"Mean absolute error(MAE): {mae}")
    print(f"R square: {r2}")
    print()
