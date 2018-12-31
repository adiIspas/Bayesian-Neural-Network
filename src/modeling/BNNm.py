import edward as ed
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from edward.models import Categorical, Normal


class BNNm(object):

    def __init__(self, batch_size, number_of_features, number_of_classes, dataset):
        self.probabilities = []
        self.w_values = []
        self.accuracy = []
        self.N = batch_size
        self.D = number_of_features
        self.K = number_of_classes
        self.data = dataset

        # Create a placeholder to hold the data (in mini batches).
        self.x = tf.placeholder(tf.float32, [self.N, self.D])

        # Normal(0,1) priors for the variables.
        self.w = Normal(loc=tf.zeros([self.D, self.K]), scale=tf.ones([self.D, self.K]))
        self.b = Normal(loc=tf.zeros(self.K), scale=tf.ones(self.K))

        # Categorical likelihood for multi class classification.
        self.y = Categorical(tf.matmul(self.x, self.w) + self.b)

        # Construct the q(w) and q(b) as Normal distributions.
        self.qw = Normal(loc=tf.Variable(tf.random_normal([self.D, self.K])),
                         scale=tf.nn.softplus(tf.Variable(tf.random_normal([self.D, self.K]))))
        self.qb = Normal(loc=tf.Variable(tf.random_normal([self.K])),
                         scale=tf.nn.softplus(tf.Variable(tf.random_normal([self.K]))))

        # We use a placeholder for the labels.
        self.y_ph = tf.placeholder(tf.int32, [self.N])

        # Define the inference - minimise the KL divergence between q and p.
        self.inference = ed.KLqp({self.w: self.qw, self.b: self.qb}, data={self.y: self.y_ph})

    def __initialize__(self, number_of_examples, iterations):
        # Initialize the inference variables
        self.inference.initialize(n_iter=iterations, n_print=self.N, scale={self.y: float(number_of_examples) / self.N})

        # Show device used, CPU Or GPU
        config = tf.ConfigProto(log_device_placement=True)
        config.gpu_options.allow_growth = True

        # Initialise all the variables in the session.
        self.session = tf.Session(config=config)
        tf.global_variables_initializer().run()

    def __load_test_data__(self):
        # Load the test images.
        self.x_test = self.data.test.images

        # TensorFlow method gives the label data in a one hot vector format.
        # We convert that into a single label.
        self.y_test = np.argmax(self.data.test.labels, axis=1)

    def train(self, iterations=5000):
        self.__initialize__(self.data.train.num_examples, iterations)

        # Training - load the data in mini batches and update the inference using each new batch.
        for _ in range(self.inference.n_iter):
            x_batch, y_batch = self.data.train.next_batch(self.N)
            # TensorFlow method gives the label data in a one hot vector format.
            # We convert that into a single label.
            y_batch = np.argmax(y_batch, axis=1)
            info_dict = self.inference.update(feed_dict={self.x: x_batch, self.y_ph: y_batch})

            self.inference.print_progress(info_dict)

    def evaluating(self, number_of_samples):
        self.__load_test_data__()
        self.w_values = []
        self.accuracy = []

        for _ in range(number_of_samples):
            w_samp = self.qw.sample()
            b_samp = self.qb.sample()

            self.w_values.append(w_samp.eval())

            # Compute the probability of each class for each (w, b) sample.
            prob = tf.nn.softmax(tf.matmul(self.x_test, w_samp) + b_samp)
            self.probabilities.append(prob.eval())

            y_pred = np.argmax(prob, axis=0).astype(np.float32)
            self.accuracy.append((y_pred == self.y_test).mean() * 100)

        self.w_values = np.reshape(self.w_values, [-1])

        # Compute the mean of probabilities for each class for all the (w,b) samples.
        y_pred = np.argmax(np.mean(self.probabilities, axis=0), axis=1)
        print("Accuracy in predicting the test data = ", (y_pred == self.y_test).mean() * 100)

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

        plt.title("Histogram of W in the MNIST test data")
        plt.xlabel("W samples")
        plt.ylabel("Frequency")

        plt.show()
