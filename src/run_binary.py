import os

from src.modeling.BNNb import BNNb

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class Binary(object):

    @staticmethod
    def run():
        print("\n=== Binary class classification ===\n")
        print("\n")

        batch_size = 100
        number_of_features = 2
        number_of_classes = 2

        iterations = 10000
        samples_1 = 100
        samples_2 = 1

        model = BNNb(batch_size, number_of_features, number_of_classes)

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
