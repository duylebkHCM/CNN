from __future__ import print_function
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import cifar10
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2 as cv

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True, help="path to output model file")
ap.add_argument("-t", "--test-images", required=True,help="path to the directory of testing images")
ap.add_argument("-b", "--batch-size", type=int, default=32,help="size of mini-batches passed to network")
args = vars(ap.parse_args())

gtLabels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

print("[INFO] loading network architecture and weights...")

model = load_model(args["model"])

print("[INFO] sampling CIFAR10...")
(testData, testLabels) = cifar10.load_data()[1]
testData = testData.astype('float')/255.0
np.random.seed(42)
idxs = np.random.choice(testData.shape[0], size = (15,), replace = False)
(testData, testLabels) = (testData[idxs], testLabels[idxs])
testLabels = testLabels.flatten()

print("[INFO] predicting on testing data...")
probs = model.predict(testData, batch_size = args["batch_size"])
predictions = probs.argmax(axis=1)
print(prediction)
for (i, prediction) in enumerate(predictions):
    image = testData[i].astype(np.float32)
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    image = imutils.resize(image, width=128, inter = cv.INTER_CUBIC)

    print("[INFO] predicted :{}, actual: {} ".format(gtLabels[prediction], gtLabels[testLabels[i]]))

    cv.imshow("Image", image)
    cv.waitKey(0)

cv.destroyAllWindows()
print("[INFO] testing on images NOT part of CIFAR-10")

for imagePath in paths.list_images(args["test_images"]):
    print("[INFO] classifying {}".format(imagePath[imagePath.rfind('/') + 1:]))
    image = cv.imread(imagePath)
    kerasImage = cv.resize(image, (32,32))
    kerasImage = cv.cvtColor(kerasImage, cv.COLOR_BGR2RGB)
    kerasImage = np.array(kerasImage, dtype = 'float')/255.0

    kerasImage = kerasImage[np.newaxis,...]
    probs = model.predict(kerasImage, batch_size = args["batch_size"])
    print(probs)
    prediction = probs.argmax(axis=1)[0]
    print(prediction)

    cv.putText(image, gtLabels[prediction], (10,35), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 3)
    cv.imshow("Image", image)
    cv.waitKey(0)
    