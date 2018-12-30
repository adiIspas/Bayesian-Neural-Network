import edward as ed
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from edward.models import Bernoulli, Normal
from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score


class BNNb(object):

    def __init__(self, batch_size, number_of_features, number_of_classes, dataset):
        self.prob_lst = []
        self.w_values = []
        self.accuracy = []
        self.N = batch_size
        self.noise = 0.2
        self.D = number_of_features
        self.K = number_of_classes

        # Create a placeholder to hold the data (in mini batches) in a TensorFlow graph.
        self.x = tf.placeholder(tf.float32, [self.N, self.D])

        # Normal(0,1) priors for the variables.
        self.w = Normal(loc=tf.zeros([self.D]), scale=tf.ones([self.D]))
        self.b = Normal(loc=tf.zeros([]), scale=tf.ones([]))

        # Bernoulli likelihood for binary classification.
        self.y = Bernoulli(ed.dot(self.x, self.w) + self.b)

        # Construct the q(w) and q(b) - we assume Normal distributions.
        self.qw = Normal(loc=tf.Variable(tf.random_normal([self.D])),
                         scale=tf.nn.softplus(tf.Variable(tf.random_normal([self.D]))))
        self.qb = Normal(loc=tf.Variable(tf.random_normal([])),
                         scale=tf.nn.softplus(tf.Variable(tf.random_normal([]))))

        # We use a placeholder for the labels in anticipation of the training data.
        self.y_ph = tf.placeholder(tf.int32, [self.N])

        # Define the VI inference technique, ie. minimise the KL divergence between q and p.
        self.inference = ed.KLqp({self.w: self.qw, self.b: self.qb}, data={self.y: self.y_ph})

    def __initialize__(self, number_of_examples, iterations):
        # Initialise the inference variables
        self.inference.initialize(n_iter=iterations, n_print=10, scale={self.y: number_of_examples / self.N})

        # Show device used, CPU Or GPU
        config = tf.ConfigProto(log_device_placement=True)
        config.gpu_options.allow_growth = True

        # Initialise all the variables in the session.
        self.session = tf.Session(config=config)
        tf.global_variables_initializer().run()

    # def __load_test_data__(self):
    #     # Load the test images.
    #     self.x_test = self.data.test.images
    #     # TensorFlow method gives the label data in a one hot vetor format. We convert that into a single label.
    #     self.y_test = np.argmax(self.data.test.labels, axis=1)

    def train(self, iterations=5000):
        self.__initialize__(iterations * self.N, iterations)

        # Let the training begin. We load the data in mini batches and update the VI inference using each new batch.
        for index in range(self.inference.n_iter):
            x_batch, y_batch = make_moons(noise=self.noise, n_samples=self.N)

            info_dict = self.inference.update(feed_dict={self.x: x_batch, self.y_ph: y_batch})
            self.inference.print_progress(info_dict)

    def evaluating(self, number_of_samples):
        # self.__load_test_data__()

        self.prob_lst = []
        self.w_values = []
        self.accuracy = []

        n_samples = number_of_samples
        w_samples = []
        b_samples = []

        for _ in range(n_samples):
            x_test, y_test = make_moons(noise=self.noise, n_samples=self.N)
            x_test = tf.cast(x_test, tf.float32)

            w_samp = self.qw.sample()
            b_samp = self.qb.sample()
            w_samples.append(w_samp)
            b_samples.append(b_samp)
            self.w_values.append(w_samp.eval())

            # Also compute the probability of each class for each (w, b) sample.
            prob = tf.nn.sigmoid(ed.dot(x_test, w_samp) + b_samp)

            y_pred = prob.eval()
            y_pred[y_pred > 0.5] = 1
            y_pred[y_pred <= 0.5] = 0
            self.accuracy.append(accuracy_score(y_test, y_pred))
            self.prob_lst.append(prob.eval())

            # self.prob_lst.append(prob.eval())
            # sample = tf.concat([tf.reshape(w_samp, [-1]), b_samp], 0)
            # samples.append(sample.eval())

        self.w_values = np.reshape(self.w_values, [1, -1])

        # Here we compute the mean of probabilities for each class for all the (w,b) samples.
        # We then use the class with maximum of the mean probabilities as the prediction.
        # In other words, we have used (w,b) samples to construct a set of models and
        # used their combined outputs to make the predictions.
        print("Accuracy in predicting the test data = ", np.mean(self.accuracy) * 100)

    def plot_accuracy(self):
        # Compute the accuracy of the model.
        # For each sample we compute the predicted class and compare with the test labels.
        # Predicted class is defined as the one which as maximum probability.
        # We perform this test for each (w,b) in the posterior giving us a set of accuracies
        # Finally we make a histogram of accuracies for the test data.
        plt.hist(self.accuracy)
        plt.title("Histogram of prediction accuracies in the MNIST test data")
        plt.xlabel("Accuracy")
        plt.ylabel("Frequency")
        plt.show()

    def plot_w(self):
        plt.hist(self.w_values[0])
        plt.title("Histogram of W in the MNIST test data")
        plt.xlabel("W samples")
        plt.ylabel("Frequency")
        plt.show()
