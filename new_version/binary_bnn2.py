import numpy as np
import pymc as pm
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale

# Definim variabile
n_samples = 300
iterations = 10000
number_of_samples = np.int32(n_samples / 2)
number_of_features = 2

# Pregatim datele de antrenare/testare
X_data, Y_data = make_moons(noise=0.2, random_state=0, n_samples=n_samples)
X_data = scale(X_data)
X_data = X_data.astype(np.double)
Y_data = Y_data.astype(np.double)
X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=.5)
Y_train = np.reshape(Y_train, [number_of_samples, 1])

# Definim arhitectura retelei
wl0 = pm.Normal('wl0', mu=0., tau=1.)
wl1 = pm.Normal('wl1', mu=0., tau=1.)


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


l0 = pm.Lambda('l0', lambda x=X_train: x)
l1 = pm.Lambda('l1', lambda l_0=l0, w_l0=wl0: sigmoid(np.dot(l_0, w_l0)))
l2 = pm.Lambda('l2', lambda l_1=l1, w_l1=wl1: sigmoid(np.dot(l_1, w_l1)))

l2_error = pm.Lambda('l2_error', lambda y_train=Y_train, l_2=l2: Y_train - l_2)
l2_delta = pm.Lambda('l2_delta', lambda l2_e=l2_error, l_2=l2: l2_e * sigmoid(l_2))
l1_error = pm.Lambda('l1_error', lambda l2_d=l2_delta, w_l1=wl1: l2_d.dot(w_l1.T))
l1_delta = pm.Lambda('l1_delta', lambda l1_e=l1_error, l_1=l1: l1_e * sigmoid(l_1))

wl0_l = pm.Lambda('wl0_l', lambda l_0=l0, l1_d=l1_delta: l_0.T.dot(l1_d))
wl1_l = pm.Lambda('wl1_l', lambda l_1=l1, l2_d=l2_delta: l_1.T.dot(l2_d))

# wl1 += l1.T.dot(l2_delta)
# wl0 += l0.T.dot(l1_delta)

# y = pm.Bernoulli('y', sigmoid, observed=True, value=Y_train)
#
# # Definim modelul si antrenam
model = pm.Model([wl0, wl1, wl0_l, wl1_l, l0, l1, l2, l2_error, l2_delta, l1_error, l1_delta])
inference = pm.MCMC(model)
inference.sample(iterations)

# y_pred_train = pm.Bernoulli('y_pred_train', sigmoid)
#
# # Plotam/afisam valorile posterior pentru W-uri
wl0 = inference.trace("wl0")[:]
# plt.hist(w11)
# plt.title('Posterior of w11')
# plt.show()
#
# w12 = inference.trace("w12")[:]
# plt.hist(w12)
# plt.title('Posterior of w12')
# plt.show()
#
# w21 = inference.trace("w21")[:]
# plt.hist(w21)
# plt.title('Posterior of w21')
# plt.show()
#
# w22 = inference.trace("w22")[:]
# plt.hist(w22)
# plt.title('Posterior of w22')
# plt.show()
#
# w31 = inference.trace("w31")[:]
# plt.hist(w31)
# plt.title('Posterior of w31')
# plt.show()
#
# w32 = inference.trace("w32")[:]
# plt.hist(w32)
# plt.title('Posterior of w32')
# plt.show()
#
# x1 = X_test[:, 0]
# x2 = X_test[:, 1]
#
# # Evaluam pe datele de test
# inference.sample(iterations)
# y_pred_test = pm.Bernoulli('y_pred_test', sigmoid)
#
# # Verificam acuratetea
# print("Accuracy on train data: ", (Y_train == y_pred_train.value).mean())
# print("Accuracy on test data: ", (Y_test == y_pred_test.value).mean())
