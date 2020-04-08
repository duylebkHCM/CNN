from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import argparse
import numpy as np
import pickle
import h5py

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--db", required=True,help="path HDF5 database")
ap.add_argument("-m", "--model", required=True,help="path to output model")
ap.add_argument("-j", "--jobs", type=int, default=-1,help="# of jobs to run when tuning hyperparameters")
args = vars(ap.parse_args())

db = h5py.File(args['db'], 'r')
train = int(db['labels'].shape[0]*0.75)

print('[INFO] tuning hyperparameter...')
params = {'C' : [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]}
model = GridSearchCV(LogisticRegression(), params, cv=3, n_jobs=args['jobs'])
model.fit(db['features'][:train], db['labels'][:train])
print('[INFO] best hyperparameters: {}'.format(model.best_params_))

print('[INFO] evaluating...')
preds = model.predict(db['features'][train:])
print(classification_report(db['labels'][train:], preds, target_names=db['label_names']))

print('[INFO] saving model...')
f = open(args['model'], 'wb')
f.write(pickle.dumps(model.best_estimator_))
f.close()
db.close()
