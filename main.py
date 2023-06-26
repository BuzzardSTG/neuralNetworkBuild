# building the neural network for image recognitipn, based on Digit Rocognizer

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


data = pd.read_csv("digit-recognizer/train.csv")
# print(data.head()) #printing our data

data = np.array(data)
m, n = data.shape
np.random.shuffle(data)

data_dev = data[0:1000].T
y_dev = data_dev[0]
x_dev = data_dev[1:n]
x_dev = x_dev / 255

data_train = data[1000:m].T
y_train = data_train[0]
x_train = data_train[1:n]
x_train = x_train / 255
_, m_train = x_train.shape


# print(x_train[:, 0].shape)

def init_params():  # def params
    w1 = np.random.rand(10, 784)
    b1 = np.random.rand(10, 1)
    w2 = np.random.rand(10, 10)
    b2 = np.random.rand(10, 1)
    return w1, b1, w2, b2


def ReLU(z):  # activation algo
    return np.maximum(z, 0)


def softmax(z):
    a = np.exp(z) / sum(np.exp(z))
    return a


def forward_prop(w1, b1, w2, b2, x):
    z1 = w1.dot(x) + b1
    a1 = ReLU(z1)
    z2 = w2.dot(a1) + b2
    a2 = softmax(a1)
    return z1, a1, z2, a2


def one_hot(y):
    one_hot_y = np.zeros((y.size, y.max() + 1))  # y.size - m
    one_hot_y[np.arange(y.size), y] = 1
    one_hot_y = one_hot_y.T
    return one_hot_y


def deriv_ReLU(z):  # derivative of ReLU for backpropagation
    return z > 0


def back_prop(z1, a1, z2, a2, w1, w2, x, y):
    one_hot_y = one_hot(y)
    dz2 = a2 - one_hot_y
    dw2 = 1 / m * dz2.dot(a1.T)
    db2 = 1 / m * np.sum(dz2)
    dz1 = w2.T.dot(dz2) * deriv_ReLU(z1)
    dw1 = 1 / m * dz2.dot(x.T)
    db1 = 1 / m * np.sum(dz1)
    return dw1, db1, dw2, db2


def update_params(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha):
    w1 = w1 - alpha * dw1
    b1 = b1 - alpha * db1
    w2 = w2 - alpha * dw2
    b2 = b2 - alpha * db2
    return w1, b1, w2, b2


def get_predictions(a2):
    return np.argmax(a2, 0)


def get_accuracy(predictions, y):
    print(predictions, y)
    return np.sum(predictions == y) / y.size


def grad_d(x, y, iterations, alpha):
    w1, b1, w2, b2 = init_params()
    for i in range (iterations):
        z1, a1, z2, a2 = forward_prop(w1, b1, w2, b2, x)
        dw1, db1, dw2, db2 = back_prop(z1, a1, z2, a2, w1, w2, x, y)
        w1, b1, w2, b2 = update_params(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha)
        if i % 10 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(a2)
            print(get_accuracy(predictions, y))
    return w1, b1, w2, b2


w1, b1, w2, b2 = grad_d(x_train, y_train, 500, 0.1)
