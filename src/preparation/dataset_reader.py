from src.preparation.clasification_type import Type
from tensorflow.examples.tutorials.mnist import input_data


class Reader(object):

    @staticmethod
    def read(classification_type):
        if classification_type == Type.MULTI_CLASS:
            return Reader.read_multi_class()
        elif classification_type == Type.BINARY:
            return Reader.read_binary()

    @staticmethod
    def read_multi_class():
        return input_data.read_data_sets("../data/raw", one_hot=True)

    @staticmethod
    def read_binary():
        return None
