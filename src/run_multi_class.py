from src.preparation.dataset_reader import Reader
from src.preparation.clasification_type import Type
from src.modeling.BNNm import BNNm

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

dataset = Reader.read(Type.MULTI_CLASS)
print("\n\n")

mini_batch_size = 100
number_of_features = 784
number_of_classes = 10

iterations = 10000
samples_1 = 100
samples_2 = 1

model = BNNm(mini_batch_size, number_of_features, number_of_classes, dataset)

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
