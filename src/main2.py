import edward as ed
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from edward.models import Normal, Bernoulli
from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score

K = 2
D = 2
BATCH_SIZE = 100
STEPS = 10000
N_TRAIN = STEPS * BATCH_SIZE
N_TEST = 100
number_of_samples = 100
noise = 0.3

# Tensorflow Placeholders:
w = Normal(loc=tf.zeros([D]), scale=tf.ones([D]))
b = Normal(loc=tf.zeros([]), scale=tf.ones([]))
x = tf.placeholder(tf.float32, [BATCH_SIZE, D])
y = Bernoulli(ed.dot(x, w) + b)
y_ph = tf.placeholder(tf.int32, [BATCH_SIZE])  # For the labels

qw = Normal(loc=tf.Variable(tf.random_normal([D])),
            scale=tf.nn.softplus(tf.Variable(tf.random_normal([D]))))
qb = Normal(loc=tf.Variable(tf.random_normal([])),
            scale=tf.nn.softplus(tf.Variable(tf.random_normal([]))))

inference = ed.KLqp({w: qw, b: qb}, data={y: y_ph})
inference.initialize(n_iter=STEPS, n_print=10, scale={y: N_TRAIN / BATCH_SIZE})

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

for _ in range(inference.n_iter):
    train_data_generator, train_labs_generator = make_moons(noise=noise, n_samples=BATCH_SIZE)
    X_batch = train_data_generator
    Y_batch = train_labs_generator

    info_dict = inference.update(feed_dict={x: X_batch, y_ph: Y_batch})
    inference.print_progress(info_dict)

# X_test, Y_test = make_moons(noise=noise, n_samples=BATCH_SIZE)
# X_test = tf.cast(X_test, tf.float32)

prob_lst, samples, w_samples, b_samples = [], [], [], []
accuracy = []
w_values = []
for index in range(number_of_samples):
    X_test, Y_test = make_moons(noise=noise, n_samples=BATCH_SIZE)
    X_test = tf.cast(X_test, tf.float32)

    w_samp = qw.sample()
    b_samp = qb.sample()
    w_samples.append(w_samp)
    b_samples.append(b_samp)
    w_values.append(w_samp.eval())

    # Probability of each class for each sample.
    prob = tf.nn.sigmoid(ed.dot(X_test, w_samp) + b_samp)

    y_pred = prob.eval()
    y_pred[y_pred > 0.5] = 1
    y_pred[y_pred <= 0.5] = 0
    accuracy.append(accuracy_score(Y_test, y_pred))
    prob_lst.append(prob.eval())
    # sample = tf.concat([tf.reshape(w_samp, [-1]), b_samp], 0)
    # samples.append(sample.eval())

w_values = np.reshape(w_values, [1, -1])
# y_pred = prob_lst[0]
# y_pred[y_pred > 0.5] = 1
# y_pred[y_pred <= 0.5] = 0

# y_pred = np.argmax(np.mean(prob_lst, axis=0), axis=1)
# print("Accuracy in predicting the test data = ", (y_pred == Y_test).mean() * 100)

# accuracy = accuracy_score(Y_test, y_pred)
print("Accuracy in predicting the test data = ", np.mean(accuracy) * 100)

plt.hist(accuracy)
plt.title("Histogram of prediction accuracies in the MNIST test data")
plt.xlabel("Accuracy")
plt.ylabel("Frequency")
plt.show()

plt.hist(w_values[0])
plt.title("Histogram of W in the MNIST test data")
plt.xlabel("W samples")
plt.ylabel("Frequency")
plt.show()
