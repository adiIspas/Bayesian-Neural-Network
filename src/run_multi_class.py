import os

from src.modeling.BNNm import BNNm
from src.preparation.clasification_type import Type
from src.preparation.dataset_reader import Reader

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class MultiClass(object):

    @staticmethod
    def run():
        print("\n=== Multi class classification ===\n")

        dataset = Reader.read(Type.MULTI_CLASS)
        print("\n")

        batch_size = 100
        number_of_features = 784
        number_of_classes = 10

        iterations = 10000
        samples_1 = 100
        samples_2 = 1

        model = BNNm(batch_size, number_of_features, number_of_classes, dataset)

        print("Training ... \n")
        model.train(iterations)

        print("\n\nEvaluation " + str(samples_1) + " sample(s)\n")
        model.evaluating(samples_1)
        model.plot_accuracy()
        model.plot_w()

        print("\n")

        print("Evaluation " + str(samples_2) + " sample(s)\n")
        # model.train(iterations)
        model.evaluating(samples_2)
        model.plot_accuracy()
        model.plot_w()
