import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
# from tensorflow_probability import edward2 as ed
import pymc as pm
from edward.models import Bernoulli, Normal, Categorical
import edward as ed
from sklearn.datasets import make_moons


class BNNb(object):

    def __init__(self, mini_batch_size, number_of_features, number_of_classes, dataset):
        self.prob_lst = []
        self.w_values = []
        self.N = mini_batch_size
        self.D = number_of_features
        self.K = number_of_classes
        self.x_train = dataset["x_train"]
        self.x_test = dataset["x_test"]
        self.y_train = dataset["y_train"]
        self.y_test = dataset["y_test"]

        # Create a placeholder to hold the data (in mini batches) in a TensorFlow graph.
        self.x = tf.placeholder(tf.float32, [None, self.D])

        # Normal(0,1) priors for the variables.
        self.w = Normal(loc=tf.zeros([self.D, self.K]), scale=tf.ones([self.D, self.K]))
        self.b = Normal(loc=tf.zeros(self.K), scale=tf.ones(self.K))

        # Bernoulli likelihood for binary classification.
        self.y = Bernoulli(tf.matmul(self.x, self.w) + self.b)

        # Construct the q(w) and q(b) - we assume Normal distributions.
        self.qw = Normal(loc=tf.Variable(tf.random_normal([self.D, self.K])),
                         scale=tf.nn.softplus(tf.Variable(tf.random_normal([self.D, self.K]))))
        self.qb = Normal(loc=tf.Variable(tf.random_normal([self.K])),
                         scale=tf.nn.softplus(tf.Variable(tf.random_normal([self.K]))))

        # We use a placeholder for the labels in anticipation of the training data.
        self.y_ph = tf.placeholder(tf.int32, [self.N, self.K])

        # Define the VI inference technique, ie. minimise the KL divergence between q and p.
        # self.inference = ed.HMC({self.w: self.qw, self.b: self.qb}, data={self.y: self.y_ph})

        self.model = pm.Model([self.qw, self.qb, self.y_ph])
        self.mcmc = pm.MCMC(self.model)

    def __initialize__(self, number_of_examples, iterations):
        # Initialise the inference variables
        # self.inference.initialize(n_iter=iterations, n_print=self.N, scale={self.y: float(number_of_examples) / self.N})

        config = tf.ConfigProto(log_device_placement=True)
        config.gpu_options.allow_growth = True
        self.session = tf.Session(config=config)

        # We will use an interactive session.
        # sess = tf.InteractiveSession()
        # Initialise all the variables in the session.
        tf.global_variables_initializer().run()

    # def __load_test_data__(self):
    #     # Load the test images.
    #     self.x_test = self.data.test.images
    #     # TensorFlow method gives the label data in a one hot vetor format. We convert that into a single label.
    #     self.y_test = np.argmax(self.data.test.labels, axis=1)

    def train(self, iterations=10):
        self.__initialize__(len(self.y_train), iterations)

        # Let the training begin. We load the data in mini batches and update the VI inference using each new batch.
        for index in range(self.inference.n_iter):
            # x_batch = self.x_train[index:(index + self.N)]
            # y_batch = self.y_train[index:(index + self.N)]
            x_batch, y_batch = make_moons(noise=0.3, n_samples=self.N)
            # TensorFlow method gives the label data in a one hot vector format. We convert that into a single label.
            # y_batch = np.argmax(y_batch, axis=1)

            info_dict = self.inference.update(feed_dict={self.x: x_batch, self.y_ph: y_batch})
            self.inference.print_progress(info_dict)

    def evaluating(self, number_of_samples):
        # self.__load_test_data__()

        self.prob_lst = []
        self.w_values = []

        n_samples = number_of_samples
        samples = []
        w_samples = []
        b_samples = []

        for _ in range(n_samples):
            w_samp = self.qw.sample()
            b_samp = self.qb.sample()
            w_samples.append(w_samp)
            self.w_values.append(w_samp.eval())
            b_samples.append(b_samp)

            # Also compute the probability of each class for each (w, b) sample.
            prob = tf.nn.softmax(tf.matmul(self.x_test, w_samp) + b_samp)
            self.prob_lst.append(prob.eval())
            sample = tf.concat([tf.reshape(w_samp, [-1]), b_samp], 0)
            samples.append(sample.eval())

        self.w_values = np.reshape(self.w_values, [1, -1])

        # Here we compute the mean of probabilities for each class for all the (w,b) samples.
        # We then use the class with maximum of the mean probabilities as the prediction.
        # In other words, we have used (w,b) samples to construct a set of models and
        # used their combined outputs to make the predictions.
        y_pred = np.argmax(np.mean(self.prob_lst, axis=0), axis=1)
        print("Accuracy in predicting the test data = ", (y_pred == self.y_test).mean() * 100)

    def plot_accuracy(self):
        # Compute the accuracy of the model.
        # For each sample we compute the predicted class and compare with the test labels.
        # Predicted class is defined as the one which as maximum probability.
        # We perform this test for each (w,b) in the posterior giving us a set of accuracies
        # Finally we make a histogram of accuracies for the test data.
        accuracy_test = []
        for prob in self.prob_lst:
            y_trn_prd = np.argmax(prob, axis=1).astype(np.float32)
            acc = (y_trn_prd == self.y_test).mean() * 100
            accuracy_test.append(acc)

        plt.hist(accuracy_test)
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
