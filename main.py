#building the neural network for image recognitipn, based on Digit Rocognizer

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv("digit-recognizer/train.csv")
#print(data.head()) #printing our data

data = np.array(data)
m, n =data.shape
np.random.shuffle(data)
data_dev = data [0:1000].T
y_dev = data_dev[0]
x_dev = data_dev[1:n]

data_train = data[1000:m].T
y_train = data_train[0]
x_train = data_train[1:n]
print(y_train)