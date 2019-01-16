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


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


# Definim w-urile
w11 = np.random.randn(1, 100)
w12 = np.random.randn(1, 100)
w21 = np.random.randn(1, 100)
w22 = np.random.randn(1, 100)
w31 = np.random.randn(1, 100)
w32 = np.random.randn(1, 100)

# Putem incerca si cu W-urile fixate la 1 unde obtinem acuratete 1.0 atat pe test cat si pe train
# w11 = np.ones([1, 100])
# w12 = np.ones([1, 100])
# w21 = np.ones([1, 100])
# w22 = np.ones([1, 100])
# w31 = np.ones([1, 100])
# w32 = np.ones([1, 100])

x1 = X_train[:, 0]
x2 = X_train[:, 1]

# Definim layerul
l0 = []

# Antrenam
for _ in range(iterations):
    x3 = np.tanh(w11 * x1 + w12 * x2)
    x4 = np.tanh(w21 * x1 + w22 * x2)

    l0 = sigmoid(w31 * x3 + w32 * x4)
    l1_error = Y_train - l0
    l1_delta = l1_error * sigmoid(l0)

    w11 += x1 * l1_delta
    w12 += x2 * l1_delta
    w21 += x1 * l1_delta
    w22 += x2 * l1_delta
    w31 += x1 * l1_delta
    w32 += x2 * l1_delta

# Acuratetea
print("\nOne sample of weights (w11, w12, w21, w22, w31, w32):\n", w11[-1][-1], w12[-1][-1], w21[-1][-1], w22[-1][-1],
      w31[-1][-1], w32[-1][-1])

l0[l0 < 0.5] = 0
l0[l0 >= 0.5] = 1
print("Accuracy on train data: ", (Y_train == l0).mean())

X_test, Y_test = generate_data(100)
x1 = X_test[:, 0]
x2 = X_test[:, 1]

x3 = np.tanh(w11 * x1 + w12 * x2)
x4 = np.tanh(w21 * x1 + w22 * x2)

l1_test = sigmoid(w31 * x3 + w32 * x4)
l1_test[l1_test < 0.5] = 0
l1_test[l1_test >= 0.5] = 1
print("Accuracy on test data: ", (Y_test == l1_test).mean())
