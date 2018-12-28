from src.preparation.dataset_reader import Reader
from src.preparation.clasification_type import Type
from src.modeling.BNNb import BNNb

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class Binary(object):

    @staticmethod
    def run():
        dataset = Reader.read(Type.BINARY)
        print("\n\n")

        mini_batch_size = 100
        number_of_features = 2
        number_of_classes = 2

        iterations = 10000
        samples_1 = 100
        samples_2 = 1

        model = BNNb(mini_batch_size, number_of_features, number_of_classes, dataset)

        print("Train and evaluation " + str(samples_1) + " sample(s)\n")
        model.train(iterations)
        model.evaluating(samples_1)
        model.plot_accuracy()
        model.plot_w()

        print("\n\n")

        print("Train and evaluation " + str(samples_2) + " sample(s)\n")
        model.train(iterations)
        model.evaluating(samples_2)
        model.plot_accuracy()
        model.plot_w()
