# import numpy as np
#
#
# def generate_data(size):
#     x = np.random.randn(size, 2)
#     y = np.tanh(x[:, 0] + x[:, 1])
#     y = 1. / (1. + np.exp(-(y + y)))
#     y = y > 0.5
#
#     return x, y
#
#
# # Definim variabile
# inputSize = 2
# hiddenSize = 2
# outputSize = 1
#
# # Generam date de antrenare
# X_train, Y_train = generate_data(100)
#
# # Weights
# w11 = np.random.randn(inputSize, hiddenSize)
# w12 = np.random.randn(hiddenSize, hiddenSize)
# w21 = np.random.randn(hiddenSize, hiddenSize)
# w22 = np.random.randn(hiddenSize, hiddenSize)
# w31 = np.random.randn(hiddenSize, hiddenSize)
# w32 = np.random.randn(hiddenSize, hiddenSize)
#
# x1 = X_train[:, 0]
# x2 = X_train[:, 1]
#
#
# def forward(x1, x2):
#     x3 = np.tanh(w11 * x1 + w12 * x2)
#     x4 = np.tanh(w21 * x1 + w22 * x2)
#
#     data = w31 * x3 + w32 * x4
#
#     return sigmoid(data)

import numpy as np

# X = (hours studying, hours sleeping), y = score on test, xPredicted = 4 hours studying & 8 hours sleeping (input data for prediction)
X = np.array(([2, 9], [1, 5], [3, 6]), dtype=float)
y = np.array(([92], [86], [89]), dtype=float)
xPredicted = np.array(([4, 8]), dtype=float)

# scale units
X = X / np.amax(X, axis=0)  # maximum of X array
xPredicted = xPredicted / np.amax(xPredicted, axis=0)  # maximum of xPredicted (our input data for the prediction)
y = y / 100  # max test score is 100


class Neural_Network(object):
    def __init__(self):
        # parameters
        self.inputSize = 2
        self.outputSize = 1
        self.hiddenSize = 3
        self.iterations = 20000
        self.number_of_samples = 100
        self.number_of_features = 100

        # weights
        self.w11 = np.random.randn(self.number_of_features, self.number_of_samples)
        self.w12 = np.random.randn(self.number_of_features, self.number_of_samples)
        self.w21 = np.random.randn(self.number_of_features, self.number_of_samples)
        self.w22 = np.random.randn(self.number_of_features, self.number_of_samples)
        self.w31 = np.random.randn(self.number_of_features, self.number_of_samples)
        self.w32 = np.random.randn(self.number_of_features, self.number_of_samples)

    def forward(self, x1, x2):
        x3 = np.tanh(self.w11 * x1 + self.w12 * x2)
        x4 = np.tanh(self.w21 * x1 + self.w22 * x2)

        self.z2 = self.sigmoid(self.w31 * x3 + self.w32 * x4)
        return self.z2

    def sigmoid(self, s):
        # activation function
        return 1 / (1 + np.exp(-s))

    def sigmoidPrime(self, s):
        # derivative of sigmoid
        return s * (1 - s)

    def backward(self, x1, x2, y, o):
        # backward propagate through the network
        self.o_error = y - o  # error in output
        self.o_delta = self.o_error * self.sigmoidPrime(o)  # applying derivative of sigmoid to error

        # self.z2_error = self.o_delta.dot(self.w11.T)  # z2 error: how much our hidden layer weights contributed to output error
        # self.z2_delta = self.o_delta.dot(self.w11.T) * self.sigmoidPrime(self.z2)  # applying derivative of sigmoid to z2 error

        self.w11 += x1.T.dot(
            self.o_delta.dot(self.w11.T) * self.sigmoidPrime(self.z2))  # adjusting first set (input --> hidden) weights
        self.w12 += x2.T.dot(self.o_delta.dot(self.w12.T) * self.sigmoidPrime(self.z2))

        self.w21 += x1.T.dot(
            self.o_delta.dot(self.w21.T) * self.sigmoidPrime(self.z2))  # adjusting first set (input --> hidden) weights
        self.w22 += x2.T.dot(self.o_delta.dot(self.w22.T) * self.sigmoidPrime(self.z2))

        self.w31 += self.z2.T.dot(self.o_delta)  # adjusting second set (hidden --> output) weights
        self.w32 += self.z2.T.dot(self.o_delta)  # adjusting second set (hidden --> output) weights

    def train(self, x1, x2, y):
        o = self.forward(x1, x2)
        self.backward(x1, x2, y, o)

    def generate_data(self):
        x = np.random.randn(self.number_of_samples, 2)
        y = np.tanh(x[:, 0] + x[:, 1])
        y = 1. / (1. + np.exp(-(y + y)))
        y = y > 0.5

        return x, y


NN = Neural_Network()
X_train, Y_train = NN.generate_data()
x1 = X_train[:, 0]
x2 = X_train[:, 1]
for i in range(1000):
    print("# " + str(i + 1) + "\n")
    print("Loss: \n" + str(np.mean(np.square(Y_train - NN.forward(x1, x2)))))  # mean sum squared loss
    print("\n")
    NN.train(x1, x2, Y_train)
