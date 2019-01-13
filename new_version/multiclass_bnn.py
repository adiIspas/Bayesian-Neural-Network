import numpy as np
import pymc as pm
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale

# Definim variabile
iterations = 2000
number_of_samples = 100

# Generam date de antrenare si testare
iris = load_iris()
X = iris.data[:, :]  # we only take the first two features.
Y = iris.target

X = scale(X)
X = X.astype(float)
Y = Y.astype(float)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.5)

# Definim arhitectura retelei
w11 = pm.Normal('w11', mu=0., tau=1.)
w12 = pm.Normal('w12', mu=0., tau=1.)
w21 = pm.Normal('w21', mu=0., tau=1.)
w22 = pm.Normal('w22', mu=0., tau=1.)
w31 = pm.Normal('w31', mu=0., tau=1.)
w32 = pm.Normal('w32', mu=0., tau=1.)

x1 = X_train[:, 0]
x2 = X_train[:, 1]
# x3 = X_train[:, 1]
# x2 = X_train[:, 1]

x3 = pm.Lambda('x3', lambda w1=w11, w2=w12: np.tanh(w1 * x1 + w2 * x2))
x4 = pm.Lambda('x4', lambda w1=w21, w2=w22: np.tanh(w1 * x1 + w2 * x2))


@pm.deterministic
def sigmoid(x=w31 * x3 + w32 * x4):
    return 1. / (1. + np.exp(-x))


y = pm.Container([pm.Categorical('y', sigmoid, observed=True, value=Y_train)])

# Definim modelul si antrenam
model = pm.Model([w11, w12, w21, w22, w31, w32, y])
inference = pm.MCMC(model)
inference.sample(iterations)

# Plotam/afisam valorile posterior pentru W-uri
w11 = inference.trace("w11")[:]
plt.hist(w11)
plt.title('Posterior of w11')
plt.show()

w12 = inference.trace("w12")[:]
plt.hist(w12)
plt.title('Posterior of w12')
plt.show()

w21 = inference.trace("w21")[:]
plt.hist(w21)
plt.title('Posterior of w21')
plt.show()

w22 = inference.trace("w22")[:]
plt.hist(w22)
plt.title('Posterior of w22')
plt.show()

w31 = inference.trace("w31")[:]
plt.hist(w31)
plt.title('Posterior of w31')
plt.show()

w32 = inference.trace("w32")[:]
plt.hist(w32)
plt.title('Posterior of w32')
plt.show()

x1 = X_test[:, 0]
x2 = X_test[:, 1]

# Evaluam pe datele de test
inference.sample(iterations)
y_pred_test = pm.Container([pm.Categorical('y_pred_test', sigmoid)])

# Verificam acuratetea pe datele de antrenare si pe datele de test
print("Accuracy on test data = {}".format((y_pred_test.value == Y_test).mean()))
