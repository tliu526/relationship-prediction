"""Script for training auto-sklearn classifier on relationship roles.

"""

import argparse
from collections import Counter
import pickle

import numpy as np
import pandas as pd

from autosklearn.classification import AutoSklearnClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score

rand_seed = 2

parser = argparse.ArgumentParser()
parser.add_argument('in_name', help='input prefix, either top_5 or top_10')
parser.add_argument('out_name', help='output model name')
parser.add_argument('--resample', action='store_true', help='whether to resample classes using SMOTE')

args = parser.parse_args()

# vanilla auto-sklearn
automl = AutoSklearnClassifier(
    #per_run_time_limit=10,
    #time_left_for_this_task=20,
    seed=rand_seed)
    #initial_configurations_via_metalearning=0)
    #ensemble_size=1, 

# load data
train_data = pickle.load(open("../data/{}_train_features.df".format(args.in_name), 'rb'))
test_data = pickle.load(open("../data/{}_test_features.df".format(args.in_name), 'rb'))

# data-preprocessing
replace_dict = {
    'contact_type': {
        "work": 0,
        "friend": 1,
        "task": 2,
        "family_live_separate": 3,
        "family_live_together": 4,
        "other": 5,
        "sig_other": 6
    }
}

train_data = train_data.replace(replace_dict)
test_data = test_data.replace(replace_dict)

train_y = train_data['contact_type']
train_X = train_data.drop(['contact_type', 'pid', 'combined_hash'], axis=1)
test_y = test_data['contact_type']
test_X = test_data.drop(['contact_type', 'pid', 'combined_hash'], axis=1)

if resample:
    print("original shape %s" % Counter(train_y))

    sm = SMOTE(random_state=rand_seed)

    train_X, train_y = sm.fit_resample(train_X, train_y)

    print("resampled shape %s" % Counter(train_y))

# training and testing
automl.fit(train_X, train_y)
predictions = automl.predict(test_X)
print("Accuracy:", accuracy_score(test_y, predictions))

# model saving: https://github.com/automl/auto-sklearn/issues/5
pickle.dump(automl, open("{}.automl".format(args.out_name), "wb"))
print(automl.get_models_with_weights())
print(automl.sprint_statistics())
