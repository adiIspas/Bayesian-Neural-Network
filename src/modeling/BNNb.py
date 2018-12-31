import edward as ed
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from edward.models import Bernoulli, Normal
from sklearn.metrics import accuracy_score

from src.preparation.dataset_reader import Reader


class BNNb(object):

    def __init__(self, batch_size, number_of_features, number_of_classes):
        self.w_values = []
        self.accuracy = []
        self.N = batch_size
        self.noise = 0.2
        self.D = number_of_features
        self.K = number_of_classes

        # Create a placeholder to hold the data (in mini batches).
        self.x = tf.placeholder(tf.float32, [self.N, self.D])

        # Normal(0,1) priors for the variables.
        self.w = Normal(loc=tf.zeros([self.D]), scale=tf.ones([self.D]))
        self.b = Normal(loc=tf.zeros([]), scale=tf.ones([]))

        # Bernoulli likelihood for binary classification.
        self.y = Bernoulli(ed.dot(self.x, self.w) + self.b)

        # Construct the q(w) and q(b) as Normal distributions.
        self.qw = Normal(loc=tf.Variable(tf.random_normal([self.D])),
                         scale=tf.nn.softplus(tf.Variable(tf.random_normal([self.D]))))
        self.qb = Normal(loc=tf.Variable(tf.random_normal([])),
                         scale=tf.nn.softplus(tf.Variable(tf.random_normal([]))))

        # We use a placeholder for the labels.
        self.y_ph = tf.placeholder(tf.int32, [self.N])

        # Define the inference - minimise the KL divergence between q and p.
        self.inference = ed.KLqp({self.w: self.qw, self.b: self.qb}, data={self.y: self.y_ph})

    def __initialize__(self, number_of_examples, iterations):
        # Initialize the inference variable
        self.inference.initialize(n_iter=iterations, n_print=10, scale={self.y: number_of_examples / self.N})

        # Show device used, CPU Or GPU
        config = tf.ConfigProto(log_device_placement=True)
        config.gpu_options.allow_growth = True

        # Initialise all variables in session.
        self.session = tf.Session(config=config)
        tf.global_variables_initializer().run()

    def train(self, iterations=5000):
        self.__initialize__(iterations * self.N, iterations)

        # Training - load the data in mini batches and update the inference using each new batch.
        for index in range(self.inference.n_iter):
            x_batch, y_batch = Reader.next_make_moons_batch(self.noise, self.N)
            info_dict = self.inference.update(feed_dict={self.x: x_batch, self.y_ph: y_batch})

            self.inference.print_progress(info_dict)

    def evaluating(self, number_of_samples):
        self.w_values = []
        self.accuracy = []

        for _ in range(number_of_samples):
            x_test, y_test = Reader.next_make_moons_batch(self.noise, self.N)
            x_test = tf.cast(x_test, tf.float32)

            w_samp = self.qw.sample()
            b_samp = self.qb.sample()

            self.w_values.append(w_samp.eval())

            # Compute the probability of each class for each (w, b) sample.
            prob = tf.nn.sigmoid(ed.dot(x_test, w_samp) + b_samp)

            y_pred = prob.eval()
            y_pred[y_pred > 0.5] = 1
            y_pred[y_pred <= 0.5] = 0

            self.accuracy.append(accuracy_score(y_test, y_pred))

        self.w_values = np.reshape(self.w_values, [-1])

        # Compute the mean of probabilities for each class for all the (w,b) samples.
        print("Accuracy in predicting the test data = ", np.mean(self.accuracy) * 100)

    def plot_accuracy(self):
        # Plot a histogram of accuracies for the test data.
        plt.hist(self.accuracy)

        plt.title("Histogram of prediction accuracies in the MNIST test data")
        plt.xlabel("Accuracy")
        plt.ylabel("Frequency")

        plt.show()

    def plot_w(self):
        # Plot a histogram of W values for the test data.
        plt.hist(self.w_values)

        plt.title("Histogram of W[0] in the MNIST test data")
        plt.xlabel("W samples")
        plt.ylabel("Frequency")

        plt.show()
