import numpy as np


def generate_data(size):
    x = np.random.randn(size, 2)
    y = np.tanh(x[:, 0] + x[:, 1])
    y = 1. / (1. + np.exp(-(y + y)))
    y = y > 0.5

    return x, y


# Definim variabile
iterations = 100000
number_of_samples = 100

# Generam date de antrenare
X_train, Y_train = generate_data(100)


# sigmoid function
def sigmoid(x):
    return 1. / (1. + np.exp(-x))


# # input dataset
# X = np.array([[0, 0, 1],
#               [0, 1, 1],
#               [1, 0, 1],
#               [1, 1, 1]])
#
# # output dataset
# y = np.array([[0, 0, 1, 1]]).T

# seed random numbers to make calculation
# deterministic (just a good practice)
np.random.seed(1)

# initialize weights randomly with mean 0
syn0 = 2 * np.random.random((100, 2)) - 1

w11 = np.random.randn(1, 100)
w12 = np.random.randn(1, 100)
w21 = np.random.randn(1, 100)
w22 = np.random.randn(1, 100)
w31 = np.random.randn(1, 100)
w32 = np.random.randn(1, 100)

x1 = X_train[:, 0]
x2 = X_train[:, 1]

l1 = []
for _ in range(iterations):
    # forward propagation
    x3 = np.tanh(w11 * x1 + w12 * x2)
    x4 = np.tanh(w21 * x1 + w22 * x2)

    l0 = X_train
    l1 = sigmoid(w31 * x3 + w32 * x4)

    l1_error = Y_train - l1

    l1_delta = l1_error * sigmoid(l1)

    w11 += x1 * l1_delta
    w12 += x2 * l1_delta
    w21 += x1 * l1_delta
    w22 += x2 * l1_delta
    w31 += x1 * l1_delta
    w32 += x2 * l1_delta

l1[l1 < 0.5] = 0
l1[l1 >= 0.5] = 1
print("Accuracy on train data: ", (Y_train == l1).mean())

X_test, Y_test = generate_data(100)

x1 = X_test[:, 0]
x2 = X_test[:, 1]

x3 = np.tanh(w11 * x1 + w12 * x2)
x4 = np.tanh(w21 * x1 + w22 * x2)

l1_test = sigmoid(w31 * x3 + w32 * x4)
l1_test[l1_test < 0.5] = 0
l1_test[l1_test >= 0.5] = 1
print("Accuracy on test data: ", (Y_test == l1_test).mean())
