import edward as ed
import numpy as np
import tensorflow as tf
from edward.models import Normal, Categorical
from sklearn.datasets import make_moons

# Define:
#  ~ K number of classes, e.g. 8
#  ~ D size of input, e.g. 10 * 40
#  ~ BATCH_SIZE, N_TRAIN, N_TEST, STEPS (how many training steps)

K = 2
D = 2
BATCH_SIZE = 100
N_TRAIN = 100
N_TEST = 100
STEPS = 1000
number_of_samples = 100

# Tensorflow Placeholders:
w = Normal(loc=tf.zeros([D, K]), scale=tf.ones([D, K]))
b = Normal(loc=tf.zeros(K), scale=tf.ones(K))
x = tf.placeholder(tf.float32, [None, D])
y = Categorical(tf.matmul(x, w) + b)
y_ph = tf.placeholder(tf.int32, [BATCH_SIZE])  # For the labels

qw = Normal(loc=tf.Variable(tf.random_normal([D, K])),
            scale=tf.nn.softplus(tf.Variable(tf.random_normal([D, K]))))
qb = Normal(loc=tf.Variable(tf.random_normal([K])),
            scale=tf.nn.softplus(tf.Variable(tf.random_normal([K]))))

inference = ed.KLqp({w: qw, b: qb}, data={y: y_ph})

inference.initialize(n_iter=STEPS, n_print=10, scale={y: N_TRAIN / BATCH_SIZE})

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# train_data_generator, train_labs_generator = make_moons(noise=0.3, n_samples=BATCH_SIZE)
# train_data_generator = iter(train_data_generator)
# train_labs_generator = iter(train_labs_generator)
for _ in range(inference.n_iter):
    train_data_generator, train_labs_generator = make_moons(noise=0.3, n_samples=BATCH_SIZE)
    X_batch = train_data_generator
    X_batch = tf.constant(X_batch, shape=[BATCH_SIZE, 2]).eval()
    Y_batch = train_labs_generator
    Y_batch = tf.constant(Y_batch, shape=[BATCH_SIZE]).eval()

    info_dict = inference.update(feed_dict={x: X_batch, y_ph: Y_batch})
    inference.print_progress(info_dict)

X_test, Y_test = make_moons(noise=0.3, n_samples=BATCH_SIZE)
X_test = tf.cast(X_test, tf.float32)

prob_lst, samples, w_samples, b_samples = [], [], [], []
for _ in range(number_of_samples):
    w_samp = qw.sample()
    b_samp = qb.sample()
    w_samples.append(w_samp)
    b_samples.append(b_samp)

    # Probability of each class for each sample.
    prob = tf.nn.softmax(tf.matmul(X_test, w_samp) + b_samp)
    prob_lst.append(prob.eval())
    sample = tf.concat([tf.reshape(w_samp, [-1]), b_samp], 0)
    samples.append(sample.eval())

y_pred = np.argmax(np.mean(prob_lst, axis=0), axis=1)
print("Accuracy in predicting the test data = ", (y_pred == Y_test).mean() * 100)
