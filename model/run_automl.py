"""Script for training auto-sklearn classifier on relationship roles.

"""

import argparse
from collections import Counter
import pickle

import numpy as np
import pandas as pd

from autosklearn.classification import AutoSklearnClassifier
from autosklearn.regression import AutoSklearnRegressor
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import GroupKFold

from model_util import build_cv_groups

rand_seed = 2
run_time = 360 # wallclock time limit for a given model, in sec
task_time = 3600 # wallclock time limit for the autoML run, in sec 

predict_targets = [
    'contact_type',
    'q1_want',
    'q2_talk',
    'q3_loan',
    'q4_closeness'
]

parser = argparse.ArgumentParser()
parser.add_argument('in_name', help='input prefix, either top_5 or top_10')
parser.add_argument('out_name', help='output model name')
parser.add_argument('predict_target', help='target value to predict: contact_type, q1_want, q2_talk, q3_loan, q4_closeness')
parser.add_argument('--resample', action='store_true', help='whether to resample classes using SMOTE')
parser.add_argument('--test', action='store_true', help='whether to make a test run of the model training')

args = parser.parse_args()

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
        "sig_other": 5
    }
}

train_data = train_data.replace(replace_dict)
test_data = test_data.replace(replace_dict)

train_y = train_data[args.predict_target]
train_X = train_data.drop(['pid', 'combined_hash'] + predict_targets, axis=1, errors='ignore')
test_y = test_data[args.predict_target]
test_X = test_data.drop(['pid', 'combined_hash'] + predict_targets, axis=1, errors='ignore')

if args.test:
    run_time = 10
    task_time = 15

if args.resample:
    print("original shape %s" % Counter(train_y))

    sm = SMOTE(random_state=rand_seed)

    train_X, train_y = sm.fit_resample(train_X, train_y)

    print("resampled shape %s" % Counter(train_y))

pid_groups = build_cv_groups(train_data['pid'])

# vanilla auto-sklearn
if args.predict_target == 'contact_type':
    automl = AutoSklearnClassifier(
        per_run_time_limit=run_time,
        time_left_for_this_task=task_time,
        resampling_strategy=GroupKFold,
        resampling_strategy_arguments={
            'folds': 5,
            'groups': np.array(pid_groups)
        },
        #initial_configurations_via_metalearning=0,
        #ensemble_size=1, 
        seed=rand_seed)
else:
    automl = AutoSklearnRegressor(
        per_run_time_limit=run_time,
        time_left_for_this_task=task_time,
        resampling_strategy=GroupKFold,
        resampling_strategy_arguments={
            'folds': 5,
            'groups': np.array(pid_groups)
        },
        #initial_configurations_via_metalearning=0,
        #ensemble_size=1, 
        seed=rand_seed)

# training and testing
automl.fit(train_X, train_y)
# refit() necessary when using cross-validation, see documentation:
# https://automl.github.io/auto-sklearn/stable/api.html#autosklearn.classification.AutoSklearnClassifier.refit
automl.refit(train_X, train_y)
predictions = automl.predict(test_X)

if args.predict_target == 'contact_type':
    print("Accuracy:", accuracy_score(test_y, predictions))
else:
    print("MSE:", mean_squared_error(test_y, predictions)) 

# model saving: https://github.com/automl/auto-sklearn/issues/5
pickle.dump(automl, open("{}.automl".format(args.out_name), "wb"))
pickle.dump(predictions, open("{}.predict".format(args.out_name), "wb"))
print(automl.sprint_statistics())
