from sklearn.datasets import make_moons
from tensorflow.examples.tutorials.mnist import input_data

from src.preparation.clasification_type import Type


class Reader(object):

    @staticmethod
    def read(classification_type):
        if classification_type == Type.MULTI_CLASS:
            return Reader.read_multi_class()

    @staticmethod
    def read_multi_class():
        return input_data.read_data_sets("../data/raw", one_hot=True)

    @staticmethod
    def next_make_moons_batch(noise, batch_size):
        # Generate batch of data
        x_batch, y_batch = make_moons(noise=noise, n_samples=batch_size)

        return x_batch, y_batch
