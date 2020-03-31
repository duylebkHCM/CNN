from __future__ import print_function
from Duy.cnn.convnetfactory import ConvNetFactory
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import cifar10
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
import argparse
import matplotlib.pyplot as plt
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("-n", "--network", required=True, help="name of network to build")
ap.add_argument("-m", "--model", required=True, help="path to output model file")
ap.add_argument("-d", "--dropout", type=int, default=-1,help="whether or not dropout should be used")
ap.add_argument("-f", "--activation", type=str, default="tanh",help="activation function to use (LeNet only)")
ap.add_argument("-e", "--epochs", type=int, default=20, help="# of epochs")
ap.add_argument("-b", "--batch-size", type=int, default=32,help="size of mini-batches passed to network")
ap.add_argument("-v", "--verbose", type=int, default=1,help="verbosity level")
args = vars(ap.parse_args())

print("[INFO] loading training data ...")
((trainData, trainLabels), (testData, testLabels)) = cifar10.load_data()
trainData = trainData.astype("float") / 255.0
testData = testData.astype("float") / 255.0

lb = LabelBinarizer()
trainLabels = lb.fit_transform(trainLabels)
testLabels = lb.fit_transform(testLabels)

kargs = {"dropout" : args["dropout"] > 0, "activation" : args["activation"]}

print("[INFO] compiling model...")
model = ConvNetFactory.build(args["network"], 3, 32, 32, 10, **kargs)
sgd = SGD(lr = 0.01, decay = 1e-6, momentum = 0.9, nesterov = True)
model.compile(loss = "categorical_crossentropy", optimizer = sgd, metrics = ["accuracy"])
print("[INFO] starting training ...")
model.fit(trainData, trainLabels, batch_size = args['batch_size'], nb_epoch = args["epochs"], verbose = args["verbose"])

(loss, accuracy) = model.evaluate(testData, testLabels, batch_size = args["batch_size"], verbose = args["verbose"])
print("[INFO] accuracy: {:.2f}%".format(accuracy*100))


print("[INFO] dumping architecture and weights to life...")
model.save(args["model"])

