"""
EECS 445 - Introduction to Maching Learning
HW2 Q2 Linear Regression Optimization Methods)
~~~~~~
Follow the instructions in the homework to complete the assignment.
"""

from ctypes import sizeof
import math
import numpy as np
import matplotlib.pyplot as plt
from helper import load_data
import timeit


def calculate_empirical_risk(X, y, theta):
    # TODO: Implement this function
    loss = 0.0
    for i in range(y.size):
        loss = loss + ((y[i] - np.dot(theta, X[i]))**2) / 2
    loss = loss / y.size
    return loss


def generate_polynomial_features(X, M):
    """
    Create a polynomial feature mapping from input examples. Each element x
    in X is mapped to an (M+1)-dimensional polynomial feature vector
    i.e. [1, x, x^2, ...,x^M].

    Args:
        X: np.array, shape (n, 1). Each row is one instance.
        M: a non-negative integer

    Returns:
        Phi: np.array, shape (n, M+1)
    """
    # TODO: Implement this function
    n = X.shape[0]
    Phi = np.zeros((n, M+1))
    for i in range(len(X)):
        for j in range(M+1):
            Phi[i][j] = pow(X[i], j)
    return Phi


def calculate_RMS_Error(X, y, theta):
    """
    Args:
        X: np.array, shape (n, d)
        y: np.array, shape (n,)
        theta: np.array, shape (d,). Specifies an (d-1)^th degree polynomial

    Returns:
        E_rms: float. The root mean square error as defined in the assignment.
    """
    # TODO: Implement this function
    E_rms = 0.0
    for i in range(len(y)):
        E_rms += (np.dot(theta, X[i]) - y[i]) ** 2
    E_rms = math.sqrt(E_rms / len(y))
    return E_rms


def ls_gradient_descent(X, y, learning_rate=0):
    """
    Implements the Gradient Descent (GD) algorithm for least squares regression.
    Note:
        - Please use the stopping criteria: number of iterations >= 1e6 or |new_loss - prev_loss| <= 1e-10

    Args:
        X: np.array, shape (n, d)
        y: np.array, shape (n,)

    Returns:
        theta: np.array, shape (d,)
    """
    d = X.shape[1]
    theta = np.zeros((d,))
    k = 0
    new_loss = calculate_empirical_risk(X, y, theta)
    prev_loss = 0

    while k < 1e6 and abs(new_loss - prev_loss) > 1e-10:
        prev_loss = new_loss
        update = np.zeros(theta.shape)
        for i in range(len(y)):
            update += (y[i] - np.dot(theta, X[i])) * X[i]
        theta += learning_rate * update / y.size

        new_loss = calculate_empirical_risk(X, y, theta)
        k += 1

    print("gradient_descent k: ", k)
    return theta


def ls_stochastic_gradient_descent(X, y, learning_rate=0):
    """
    Implements the Stochastic Gradient Descent (SGD) algorithm for least squares regression.
    Note:
        - Please do not shuffle your data points.
        - Please use the stopping criteria: number of iterations >= 1e6 or |new_loss - prev_loss| <= 1e-10

    Args:
        X: np.array, shape (n, d)
        y: np.array, shape (n,)

    Returns:
        theta: np.array, shape (d,)
    """
    d = X.shape[1]
    theta = np.zeros((d,))
    k = 0

    new_loss = calculate_empirical_risk(X, y, theta)
    prev_loss = 0

    while k < 1e6 and abs(new_loss - prev_loss) > 1e-10:
        prev_loss = new_loss
        for i in range(len(y)):
            theta += learning_rate * ((y[i] - np.dot(theta, X[i])) * X[i])
            k += 1
        new_loss = calculate_empirical_risk(X, y, theta)
    print("ls_stochastic_gradient_descent k: ", k)
    return theta


def closed_form_optimization(X, y, reg_param=0):
    """
    Implements the closed form solution for least squares regression.

    Args:
        X: np.array, shape (n, d)
        y: np.array, shape (n,)
        `reg_param`: float, an optional regularization parameter

    Returns:
        theta: np.array, shape (d,)
    """
    theta = np.zeros(X.shape[1])
    identity = np.identity(len(X[0]))
    theta = np.matmul(np.linalg.pinv(
        np.matmul(np.transpose(X), X)), np.matmul(np.transpose(X), y))
    return theta


def closed_form_optimization_modified(X, y, reg_param=0):
    """
    Implements the closed form solution for least squares regression.

    Args:
        X: np.array, shape (n, d)
        y: np.array, shape (n,)
        `reg_param`: float, an optional regularization parameter

    Returns:
        theta: np.array, shape (d,)
    """
    theta = np.zeros(X.shape[1])
    identity = np.identity(len(X[0]))
    theta = np.matmul(np.linalg.pinv((reg_param * identity) +
                      np.matmul(np.transpose(X), X)), np.matmul(np.transpose(X), y))
    return theta


def part_2_1(fname_train):
    # TODO: This function should contain all the code you implement to complete 2.1. Please feel free to add more plot commands.
    print("========== Part 2.1 ==========")

    X_train, y_train = load_data(fname_train)

    value = {10**-4, 10**-3, 10**-2, 10**-1}

    for v in value:
        start = timeit.default_timer()
        X = generate_polynomial_features(X_train, 1)
        # print(v, " ", ls_gradient_descent(X, y_train, v))
        # print(v, " ", ls_stochastic_gradient_descent(X, y_train, v))
        stop = timeit.default_timer()
        print('Time: ', stop - start)

    X = generate_polynomial_features(X_train, 1)
    start = timeit.default_timer()
    print(" ", closed_form_optimization(X, y_train))
    stop = timeit.default_timer()
    print('Time: ', stop - start)

    print("Done!")
    plt.plot(X_train, y_train, 'ro')
    plt.legend()
    plt.savefig('q2_1.png', dpi=200)
    plt.close()


def part_2_2(fname_train, fname_validation):
    # TODO: This function should contain all the code you implement to complete 2.2
    print("=========== Part 2.2 ==========")

    X_train, y_train = load_data(fname_train)
    X_validation, y_validation = load_data(fname_validation)

    # (a) OVERFITTING

    # errors_train = np.zeros((11,))
    # errors_validation = np.zeros((11,))
    # # Add your code here
    # for i in range(11):
    #     X = generate_polynomial_features(X_train, i)
    #     theta1 = closed_form_optimization(X, y_train)
    #     errors_train[i] = calculate_RMS_Error(X, y_train, theta1)

    #     X_2 = generate_polynomial_features(X_validation, i)
    #     # theta2 = closed_form_optimization(X_2, y_validation)
    #     errors_validation[i] = calculate_RMS_Error(X_2, y_validation, theta1)

    #     # print(i, " errors_train: ", errors_train)
    #     # print(i, " errors_validation: ", errors_validation)

    # plt.plot(errors_train, '-or', label='Train')
    # plt.plot(errors_validation, '-ob', label='Validation')
    # plt.xlabel('M')
    # plt.ylabel('$E_{RMS}$')
    # plt.title('Part 2.2.a')
    # plt.legend(loc=1)
    # plt.xticks(np.arange(0, 11, 1))
    # plt.savefig('q2_2_a.png', dpi=200)
    # plt.close()

    # (b) REGULARIZATION

    errors_train = np.zeros((10,))
    errors_validation = np.zeros((10,))
    L = np.append([0], 10.0 ** np.arange(-8, 1))
    # Add your code here
    # lamda = {0, 10**-8, 10**-7, 10**-6, 10**-5,
    #          10**-4, 10**-3, 10**-2, 10**-1, 10**-0}
    i = 0
    for lm in L:
        X = generate_polynomial_features(X_train, 10)
        theta3 = closed_form_optimization_modified(X, y_train, reg_param=lm)
        errors_train[i] = calculate_RMS_Error(X, y_train, theta3)

        X_2 = generate_polynomial_features(X_validation, 10)
        # theta2 = closed_form_optimization_modified(
        #     X_2, y_validation, reg_param=lm)
        errors_validation[i] = calculate_RMS_Error(
            X_2, y_validation, theta3)
        i += 1

    plt.figure()
    plt.plot(L, errors_train, '-or', label='Train')
    plt.plot(L, errors_validation, '-ob', label='Validation')
    plt.xscale('symlog', linthresh=1e-8)
    plt.xlabel('$\lambda$')
    plt.ylabel('$E_{RMS}$')
    plt.title('Part 2.2.b')
    plt.legend(loc=2)
    plt.savefig('q2_2_b.png', dpi=200)
    plt.close()

    print("Done!")


def main(fname_train, fname_validation):
    part_2_1(fname_train)
    # part_2_2(fname_train, fname_validation)


if __name__ == '__main__':
    main("dataset/q2_train.csv", "dataset/q2_validation.csv")
