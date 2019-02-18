"""Script for refitting AutoML models for 10-fold CV Min et al performance evaluation.

"""
import argparse
from collections import Counter
import pickle

import pandas as pd
from sklearn.model_selection import GroupKFold
import sklearn.metrics as sk_metrics
from imblearn.under_sampling import RandomUnderSampler

from model_util import build_cv_groups


parser = argparse.ArgumentParser()
parser.add_argument('data', help='path to evaluation data')
parser.add_argument('model', help='path to evaluation data')
parser.add_argument('predict_target', help='target value to predict')
parser.add_argument('num_folds', help='number of folds to create')
parser.add_argument('out_name', help='output prediction file name')
parser.add_argument('--test', action='store_true', help='whether to make a test run of the model training')
parser.add_argument('--zimmerman', action='store_true', help='whether CV over Zimmerman classification')

args = parser.parse_args()

contact_dict =  {
            "work": 0,
            "social": 1,
            "family": 2
        }


replace_dict = {'contact_type': contact_dict}

predict_targets = [
    'contact_type',
    'q1_want',
    'q2_talk',
    'q3_loan',
    'q4_closeness',
    'tie_str_score',
    'tie_str_rank',
    'tie_str_class'
]

# load data
#data = pickle.load(open('../data/zimmerman_features/zimmerman_contact_type_baseline_train_features.df', 'rb'))
data = pickle.load(open(args.data, 'rb'))
model = pickle.load(open(args.model, 'rb'))

train_data = data.replace(replace_dict)

train_y = train_data[args.predict_target]
train_X = train_data.drop(['contact_type',  'combined_hash'] + predict_targets, axis=1, errors='ignore')

# create group folds like run_automl
rand_seed = 2

if args.zimmerman:
    print("original shape %s" % Counter(train_y))
    sm = RandomUnderSampler(random_state=rand_seed)

    train_X, train_y = sm.fit_resample(train_X, train_y)
    print("resampled shape %s" % Counter(train_y))

    pid_groups = build_cv_groups(pd.Series(train_X[:,0])) # pid col
    train_X = train_X[:, 1:]



pid_groups = build_cv_groups(train_X['pid'])
train_X = train_X.drop(['pid'], axis=1)
train_X = train_X.values
train_y = train_y.values

group_kfold = GroupKFold(n_splits=int(args.num_folds))

fold_preds = []
for train_idx, test_idx in group_kfold.split(train_X, train_y, pid_groups):
    train_fold_X = train_X[train_idx]
    train_fold_y = train_y[train_idx]
    
    test_fold_X = train_X[test_idx]
    test_fold_y = train_y[test_idx]
    print("fold shape %s" % Counter(train_fold_y))
    print("fold shape %s" % Counter(test_fold_y))
    
    model.refit(train_fold_X, train_fold_y)
    predictions = model.predict(test_fold_X)
    
    fold_preds.append(predictions)
    print("Acc: ", sk_metrics.accuracy_score(test_fold_y, predictions))
    if args.test:
        break

pickle.dump(fold_preds, open(args.out_name + ".cv_predict", "wb"), -1)
