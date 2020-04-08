from __future__ import print_function
import tensorflow
from tensorflow.keras.models import load_model
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2 as cv
import h5py

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True, help="path to output model file")
ap.add_argument("-d", "--db", required=True,help="path HDF5 database")
ap.add_argument("-s", "--dataset", required=True,help="path to images dataset")
ap.add_argument("-t", "--test-images", required=True,help="path to the directory of testing images")
ap.add_argument("-b", "--batch-size", type=int, default=32,help="size of mini-batches passed to network")
args = vars(ap.parse_args())

model = load_model(args['model'])

db = h5py.File(args['db'], 'r')

test = int(db['labels'].shape[0]*0.25)
np.random.seed(42)
idxs = np.random.choice(db['labels'][-test], size = (15, ), replace = False)
testFeatures, testLabels, testNames = db['features'][idxs], db['labels'][idxs], db['image_names'][idxs]

predictions = model.predict(testFeatures)

for (i, prediction) in enumerate(predictions):
    image = cv.imread(args['dataset'] + '/' + testNames[i])
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    print('[INFO] predicted: {}, actual: {}'.format(db['image_name'][prediction], db['image_names'][i]))
    cv.putText(image, db['image_name'][prediction], (10,35), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 3)
    cv.imshow('Image', image)
    cv.waitKey(0)