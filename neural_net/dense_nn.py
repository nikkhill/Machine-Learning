"""
This file implements a dense neural network
to predict handwritten letters. Using a modular approach, so we can easily add more hidden layers.
@author nbhale
"""

import sys
import numpy as np


def initialize_weights(method, shape):
    """
    Initialize weights accoreding to initializing method.
    :param method: string specifying which method to use
    :param shape: shape of np array to return
    :return:  weights array
    """
    if method == "zeros":
        return np.zeros(shape, dtype=np.float64)
    elif method == "random0.1":
        return np.random.uniform(-0.1, 0.1, size=shape)
    else:
        raise NotImplementedError("Only zeros and random0.1 supported.")


class SigmoidLayer:
    # z = sigmoid(a)
    def forward(self, a):
        return np.reciprocal(1 + np.exp(-a)).reshape(a.shape[0], 1)

    def backward(self, z, gz):
        return (gz * (z * (1 - z))).reshape(gz.shape[0], 1)


class LinearLayer:
    # a = Wx + w0
    def __init__(self, input_dim, output_dim, weight_init_method):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.w = initialize_weights(weight_init_method, shape=(self.output_dim, self.input_dim))
        self.w0 = initialize_weights(weight_init_method, shape=(self.output_dim, 1))

    def forward(self, x):
        return self.w0 + np.matmul(self.w, x)

    def backward(self, x, ga):
        return np.matmul(ga, x.T), ga, np.matmul(self.w.T, ga)

    def update_weights(self, gw, gw0, learning_rate):
        self.w = self.w - learning_rate * gw
        self.w0 = self.w0 - learning_rate * gw0


class SoftmaxCrossEntropyLayer:
    # loss = - np.sum(y * np.log(yh)) and  yh = softmax(b)
    def forward(self, b, y):
        expb = np.exp(b)
        yh = expb / np.sum(expb)
        return yh, - np.matmul(y.T, np.log(yh))

    def backward(self, y, yh):
        # output_dim = y.shape[0]
        # return ( np.matmul((np.eye(output_dim) - np.matmul(yh, np.ones((1, output_dim)))), -y).reshape(output_dim, 1))
        return yh - y


class NN:
    def __init__(self, input_dim, hidden_dim, output_dim, weight_init_method):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.lin0 = LinearLayer(self.input_dim, self.hidden_dim, weight_init_method)
        self.sigm0 = SigmoidLayer()
        self.lin1 = LinearLayer(self.hidden_dim, self.output_dim, weight_init_method)
        self.softmax_crossentropy1 = SoftmaxCrossEntropyLayer()

    def forward(self, x, y):
        self.x = x.reshape(self.input_dim, 1)
        self.y = y.reshape(self.output_dim, 1)
        self.a = self.lin0.forward(self.x).reshape(self.hidden_dim, 1)
        self.z = self.sigm0.forward(self.a).reshape(self.hidden_dim, 1)
        self.b = self.lin1.forward(self.z).reshape(self.output_dim, 1)
        self.yh, self.loss = self.softmax_crossentropy1.forward(self.b, self.y)
        return self.yh, self.loss

    def backward(self):
        gb = self.softmax_crossentropy1.backward(self.y, self.yh)
        self.gbeta, self.gbeta0, gz = self.lin1.backward(self.z, gb)
        ga = self.sigm0.backward(self.z, gz)
        self.galpha, self.galpha0, _ = self.lin0.backward(self.x, ga)
        grads = {
            "alpha": self.galpha,
            "alpha0": self.galpha0,
            "beta": self.gbeta,
            "beta0": self.gbeta0
        }
        return grads

    def update_weights(self, learning_rate):
        self.lin0.update_weights(self.galpha, self.galpha0, learning_rate)
        self.lin1.update_weights(self.gbeta, self.gbeta0, learning_rate)


def evaluate(nn, X, Y, output_file=None):
    """
    Evaluate loss and print prediction if desired.
    :param nn: NN object
    :param X: Input
    :param Y: Output (one hots)
    :param output_file: file path where we will write predictions.
    :return: avg loss, error
    """
    total_loss = 0.0
    predictions = []
    errors = 0
    for i in range(X.shape[0]):
        yh, loss = nn.forward(X[i], Y[i])
        total_loss += loss
        predicted_class = np.argmax(yh)
        if Y[i][predicted_class] != 1:
            errors += 1
        predictions.append(predicted_class)
    if output_file:
        with open(output_file, "w") as of:
            for predicted_class in predictions:
                of.write(str(predicted_class) + '\n')
    return total_loss.squeeze() / X.shape[0], errors / X.shape[0]


def read_data(in_file):
    """
    Reads data from a csv file with output value and other values in each line
    :param in_file: data file
    :return: X (a list of dicts), y (a list of labels)
    """
    with open(in_file) as inf:
        data = [line.split(',') for line in inf.readlines()]
    data = np.array(data, dtype=np.float64)
    y = data[:, 0]
    X = data[:, 1:]
    Y = np.eye(10)[y.astype(int)]
    return X, Y


def train(nn, X, Y, X_test, Y_test, num_epochs, learning_rate):
    """
    Trains the neural net using SGD
    :param nn: NN object to train
    :param X: Input data (num_examples, input_dim)
    :param Y: One hot output examples (num_examples, output_dim)
    :param num_epochs: Iterations over the data
    :param learning_rate: :)
    """
    print("Using learning rate =", learning_rate)
    loss_trend_train = []
    loss_trend_test = []
    for e in range(num_epochs):
        for i in range(X.shape[0]):
            nn.forward(X[i], Y[i])
            nn.backward()
            nn.update_weights(learning_rate)
        avg_loss, _ = evaluate(nn, X, Y)
        avg_loss_test, _ = evaluate(nn, X_test, Y_test)
        loss_trend_train.append(avg_loss)
        loss_trend_test.append(avg_loss_test)
        print("Avg loss after epoch {}: {}".format(e, avg_loss))
    return loss_trend_train, loss_trend_test


if __name__ == "__main__":
    _, train_input_file, test_input_file, train_output_file, test_output_file \
        , metrics_output_file, num_epochs, hidden_units, init_flag, learning_rate = sys.argv
    num_epochs, hidden_units, learning_rate = int(num_epochs), int(hidden_units), float(learning_rate)
    # init_flag tells whether to randomly initialize weights.
    if init_flag == "2":
        method = "zeros"
    else:
        method = "random0.1"
    print(sys.argv)

    train_X, train_Y = read_data(train_input_file)
    test_X, test_Y = read_data(test_input_file)

    nn = NN(train_X.shape[1], hidden_units, train_Y.shape[1], method)
    loss_trend_train, loss_trend_test = train(nn, train_X, train_Y, test_X, test_Y, num_epochs, learning_rate)

    train_loss, train_err = evaluate(nn, train_X, train_Y, train_output_file)
    print("Avg training loss:", train_loss)
    print("train error:", train_err)
    test_loss, test_err = evaluate(nn, test_X, test_Y, test_output_file)
    print("Avg test loss:", test_loss)
    print("Test error:", test_err)

    with open(metrics_output_file, 'w') as m_file:
        for epoch in range(len(loss_trend_train)):
            m_file.write("epoch={} crossentropy(train): {}\n".format(epoch + 1, loss_trend_train[epoch]))
            m_file.write("epoch={} crossentropy(test): {}\n".format(epoch + 1, loss_trend_test[epoch]))
        m_file.write("error(train): {}\n".format(train_err))
        m_file.write("error(test): {}\n".format(test_err))
