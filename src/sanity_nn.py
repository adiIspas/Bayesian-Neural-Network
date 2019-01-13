import numpy as np
import pymc as pm
from matplotlib import pyplot as plt

# def generate_data(size):
#     X = np.random.randn(size, 2)
#     Y = np.tanh(X_train[:, 0] + X_train[:, 1])
#     Y = 1. / (1. + np.exp(-(Y + Y)))
#     Y = Y > 0.5

# TRAIN #
number_of_samples = 10000
X_train = np.random.randn(100, 2)
Y_train = np.tanh(X_train[:, 0] + X_train[:, 1])
Y_train = 1. / (1. + np.exp(-(Y_train + Y_train)))
Y_train = Y_train > 0.5

w11 = pm.Normal('w11', mu=0., tau=1.)
w12 = pm.Normal('w12', mu=0., tau=1.)
w21 = pm.Normal('w21', mu=0., tau=1.)
w22 = pm.Normal('w22', mu=0., tau=1.)
w31 = pm.Normal('w31', mu=0., tau=1.)
w32 = pm.Normal('w32', mu=0., tau=1.)

x1 = X_train[:, 0]
x2 = X_train[:, 1]

x3 = pm.Lambda('x3', lambda w1=w11, w2=w12: np.tanh(w1 * x1 + w2 * x2))
x4 = pm.Lambda('x4', lambda w1=w21, w2=w22: np.tanh(w1 * x1 + w2 * x2))


@pm.deterministic
def sigmoid(x=w31 * x3 + w32 * x4):
    return 1. / (1. + np.exp(-x))


y = pm.Bernoulli('y', sigmoid, observed=True, value=Y_train)
model = pm.Model([w11, w12, w21, w22, w31, w32, y])
inference = pm.MCMC(model)

inference.sample(number_of_samples)
y_pred_train = pm.Bernoulli('y_pred_train', sigmoid)

traces = inference.trace("w11")[:]
plt.hist(traces)
plt.title('Posterior of w11')
plt.show()

traces = inference.trace("w12")[:]
plt.hist(traces)
plt.title('Posterior of w12')
plt.show()

traces = inference.trace("w21")[:]
plt.hist(traces)
plt.title('Posterior of w21')
plt.show()

traces = inference.trace("w22")[:]
plt.hist(traces)
plt.title('Posterior of w22')
plt.show()

traces = inference.trace("w31")[:]
plt.hist(traces)
plt.title('Posterior of w31')
plt.show()

traces = inference.trace("w32")[:]
plt.hist(traces)
plt.title('Posterior of w32')
plt.show()

# TEST #
X_test = np.random.randn(100, 2)
Y_test = np.tanh(X_test[:, 0] + X_test[:, 1])
Y_test = 1. / (1. + np.exp(-(Y_test + Y_test)))
Y_test = Y_test > 0.5

x1 = X_test[:, 0]
x2 = X_test[:, 1]

inference.sample(number_of_samples)
y_pred_test = pm.Bernoulli('y_pred_test', sigmoid)

print('Accuracy on train data = {}%'.format((y_pred_train.value == Y_train).mean() * 100))
print('Accuracy on test data = {}%'.format((y_pred_test.value == Y_test).mean() * 100))
