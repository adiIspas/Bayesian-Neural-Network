import numpy as np
import scipy
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale


def sigmoid(x, deriv=False):
    if deriv:
        return x * (1 - x)
    return scipy.special.expit(x)  # 1 / (1 + np.exp(-x))
    # return scipy.special.expit(x) / np.sum(scipy.special.expit(x))


# Setam un seed pentru a genera aceleasi rezultate
np.random.seed(3721)

# Definim variabile
n_samples = 150
iterations = 100000
number_of_samples = np.int32(n_samples / 2)
number_of_features = 2

# Pregatim datele de antrenare/testare
iris = load_iris()
X_data = iris.data[:, :2]
Y_data = iris.target

X_data = scale(X_data)
X_data = X_data.astype(float)
Y_data = Y_data.astype(float)
X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=.5)
Y_train = np.reshape(Y_train, [number_of_samples, 1])

# Initializam w-urile
w_l0 = 2 * np.random.random((number_of_features, number_of_samples)) - 1
w_l1 = 2 * np.random.random((number_of_samples, 1)) - 1

# Definim cele 3 layere din retea
l0 = []
l1 = []
l2 = []
for index in range(iterations):

    # Primul layer este reprezentat de datele de antrenare
    l0 = X_train

    # Al doilea/treilea layer este rezultatul functiei sigmoid cu w-urile din primul/al doilea layer
    l1 = sigmoid(np.dot(l0, w_l0))
    l2 = sigmoid(np.dot(l1, w_l1))

    # Calculam eroarea fata de datele de antrenare
    l2_error = Y_train - l2

    if (index % 1000) == 0:
        print("Loss:" + str(np.mean(np.abs(l2_error))))

    # Calculam eroarea retelei
    l2_delta = l2_error * sigmoid(l2)
    l1_error = l2_delta.dot(w_l1.T)
    l1_delta = l1_error * sigmoid(l1)

    # Actualizam w-urile pentru a minimiza eroarea fata de datele de antrenare
    w_l1 += l1.T.dot(l2_delta)
    w_l0 += l0.T.dot(l1_delta)

# Verificam acuratetea
y_pred_train = l2
y_pred_train[y_pred_train <= 3] = 2
y_pred_train[y_pred_train <= 2] = 1
y_pred_train[y_pred_train <= 1] = 0

l0 = X_test
l1 = sigmoid(np.dot(l0, w_l0))
l2 = sigmoid(np.dot(l1, w_l1))

y_pred_test = l2
y_pred_test[y_pred_test <= 3] = 2
y_pred_test[y_pred_test <= 2] = 1
y_pred_test[y_pred_test <= 1] = 0

print("Accuracy on train data: ", (Y_train == y_pred_train).mean())
print("Accuracy on test data: ", (Y_test == y_pred_test).mean())
