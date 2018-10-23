"""
This file implements the binary logistic regression classifier
@author nbhale
"""

import sys
import math
import numpy as np
import matplotlib.pyplot as plt


def sparse_dot(vec_a, sparse_vec_b):
    """
    Performs dot product on sparse vectors. Assumes both vectors have same dimensions.
    :param vec_a: A normal vector (list/array)
    :param sparse_vec_b: A sparse vector in the form of dictionary
    :return: result of the dot product
    """
    result = 0.0
    for i in sparse_vec_b:
        result += vec_a[i] * sparse_vec_b[i]
    return result


# basic test for sparse_dot
assert sparse_dot([1, 4, 0, 1], {1: 1, 3: 2}) == 6


def nll_i(params, x_i, y_i):
    a_i = sparse_dot(params, x_i)
    return - y_i * a_i + math.log(1 + math.exp(a_i))


def avg_nll(params, X, y):
    """
    Calculates the total loss
    """
    result = 0.0
    for i in range(len(X)):
        result += nll_i(params, X[i], y[i])
    return result / len(X)


def sgd_update(params, x_i, y_i, learning_rate):
    """
    Performs SGD update opn single example
    :param params: Array of dim 1+M
    :param x_i: Sparse array of dim 1+M
    :param y_i: Outut class 0/1
    :param learning_rate: :)
    :return: array of new parameters of dim M+1
    """
    a_i = sparse_dot(params, x_i)
    grad = np.zeros(len(params))
    for j in x_i:  # since x_i is sparse, only calculate where needed.
        grad[j] = - x_i[j] * (y_i - math.exp(a_i) / (1 + math.exp(a_i)))
    return params - learning_rate * grad


def train(X, y, M, epochs, learning_rate=0.1, track_loss=False, X_valid=None, y_valid=None):
    """
    Training process which uses SGD optimizer
    :param X: Array (Nx1) of sparse vectors of size (1+M)
    :param y: Array of outputs
    :param M: Number of features
    :param epochs: Number of epochs
    :param learning_rate: :)
    :param track_loss: If true, will calculate loss on validation and training set
    :return: The learned parameters
    """
    params = np.zeros(M + 1)
    tracked_train_loss, tracked_valid_loss = [], []
    print("params initialized with shape:", params.shape)
    for e in range(epochs):
        for i in range(len(X)):
            params = sgd_update(params, X[i], y[i], learning_rate)
        train_loss = avg_nll(params, X, y)
        print("Training set avg loss after epoch {}: {}".format(e, train_loss))
        if track_loss:
            valid_loss = avg_nll(params, X_valid, y_valid)
            tracked_train_loss.append(train_loss)
            tracked_valid_loss.append(valid_loss)
    if track_loss:
        return params, tracked_train_loss, tracked_valid_loss
    else:
        return params


def predict(params, X, y, write_output=False, output_file=None):
    """
    Calculate the misclassification rate and outputs labels to output_file
    :param params: Array of dim 1+M
    :param X: Array (Nx1) of sparse vectors of size (1+M)
    :param y: Array (Nx1) of output classes
    :return: error rate value
    """
    errors = 0
    predictions = []
    for i in range(len(X)):
        x_i = X[i]
        a_i = sparse_dot(params, x_i)
        if math.exp(a_i) / (1 + math.exp(a_i)) > 0.5:
            predicted_y_i = 1
        else:
            predicted_y_i = 0
        predictions.append(predicted_y_i)
        if predicted_y_i != y[i]:
            errors += 1
    if write_output:
        with open(output_file, 'w') as out_f:
            for y_i in predictions:
                out_f.write(str(y_i) + '\n')
    return errors / len(X)


def read_data(in_file):
    """
    Reads data from file
    :param in_file: data file
    :return: X (a list of dicts), y (a list of labels)
    """
    X, y = [], []
    with open(in_file) as inf:
        for line in inf:
            data = line.strip().split('\t')
            y.append(int(data[0]))
            x_i = [feature.split(':') for feature in data[1:]]
            x_i = {int(k) + 1: int(v) for k, v in x_i}  # adding 1 to accommodate x_i[0] = 1
            x_i[0] = 1
            X.append(x_i)
    return X, y


if __name__ == "__main__":
    _, train_input_file, valid_input_file, test_input_file, \
    dict_input_file, train_output_file, test_output_file, metrics_output_file, num_epochs = sys.argv
    print(sys.argv)
    with open(dict_input_file) as dict_file:
        word_to_index = {k.strip(): int(v) for k, v in [line.split() for line in dict_file.readlines()]}
    train_X, train_y = read_data(train_input_file)
    valid_X, valid_y = read_data(valid_input_file)
    learned_params, train_losses, valid_losses = train(train_X, train_y, len(word_to_index), int(num_epochs),
                                                       track_loss=True, X_valid=valid_X, y_valid=valid_y)
    train_error = predict(learned_params, train_X, train_y, write_output=True, output_file=train_output_file)
    print("Train error:", train_error)
    test_X, test_y = read_data(test_input_file)

    test_error = predict(learned_params, test_X, test_y, write_output=True, output_file=test_output_file)
    print("Test error:", test_error)
    with open(metrics_output_file, 'w') as m_f:
        m_f.write("error(train): {0:.6f}\n".format(train_error))
        m_f.write("error(test): {0:.6f}\n".format(test_error))

    plt.plot(range(len(train_losses)), train_losses, 'g', label="training loss")
    plt.plot(range(len(valid_losses)), valid_losses, 'b', label="validation loss")
    plt.legend(loc='upper right')
    plt.xlabel("Number of epochs")
    plt.ylabel("Avg NLL")
    plt.show()
