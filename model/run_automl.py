"""Script for training auto-sklearn classifier on relationship roles.

"""

import argparse
from collections import Counter
import pickle

import numpy as np
import pandas as pd

from autosklearn.classification import AutoSklearnClassifier
import autosklearn.metrics
from autosklearn.regression import AutoSklearnRegressor
from imblearn.over_sampling import SMOTENC, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import accuracy_score, mean_squared_error, f1_score
import sklearn.model_selection as sk_model_select

from model_util import build_cv_groups

# horrible hack to get around auto-sklearn limitations, shadowing names
from GroupKFoldSample import GroupKFold

rand_seed = 2
run_time = 360 # wallclock time limit for a given model, in sec
task_time = 3600 # wallclock time limit for the autoML run, in sec 

ensemble_size = 50 # default ensemble size for AutoML
ensemble_nbest = 50
estimators = None # default estimators, selects all of them

ensemble_mem_limit = 5120

predict_targets = [
    'contact_type',
    'q1_want',
    'q2_talk',
    'q3_loan',
    'q4_closeness',
    'tie_str_score',
    'tie_str_rank',
    'tie_str_class',
    'ego_age_q'
]

parser = argparse.ArgumentParser()
parser.add_argument('in_name', help='input prefix, either top_5 or top_10')
parser.add_argument('out_name', help='output model name')
parser.add_argument('predict_target', help='target value to predict: contact_type, q1_want, q2_talk, q3_loan, q4_closeness')

parser.add_argument('--group_res', action='store_true', help='whether to run GroupKFold resampling')
parser.add_argument('--resample', action='store_true', help='whether to resample classes')
parser.add_argument('--smote', action='store_true', help='whether to resample classes using SMOTENC')
parser.add_argument('--rand_downsample', action='store_true', help='whether to resample classes by downsampling')

parser.add_argument('--rand_forest', action='store_true', help='run only random forest')
parser.add_argument('--svc', action='store_true', help='run only SVC')

parser.add_argument('--test', action='store_true', help='whether to make a test run of the model training')
parser.add_argument('--run_time', help='optionally specify run time')
parser.add_argument('--task_time', help='optionally specify task time')

parser.add_argument('--log_loss', action='store_true', help='use log loss for classification')
parser.add_argument('--weighted_f1', action='store_true', help='optionally makes classification loss weighted F1')
parser.add_argument('--micro_f1', action='store_true', help='optionally makes classification loss micro F1')
parser.add_argument('--macro_f1', action='store_true', help='optionally makes classification loss macro F1')

parser.add_argument('--collapse_classes', action='store_true', help='optionally collapse relationship classes')
parser.add_argument('--zimmerman_classes', action='store_true', help='use Zimmerman contact classes')
parser.add_argument('--emc_clf', action='store_true', help='optionally makes EMC prediction task into classification')
parser.add_argument('--tie_str_verystrong', action='store_true', help='optionally makes tie strength into 2-class \"very strong\" classification')
parser.add_argument('--tie_str_medstrong', action='store_true', help='optionally makes tie strength into 2-class \"med strong\" classification')

parser.add_argument('--age_gender_only', action='store_true', help='optionally makes a run with only age and gender as features')

args = parser.parse_args()

# load data
train_data = pickle.load(open("{}_train_features.df".format(args.in_name), 'rb'))
test_data = pickle.load(open("{}_test_features.df".format(args.in_name), 'rb'))

# data-preprocessing
contact_dict = {
        "work": 0,
        "friend": 1,
        "task": 2,
        "family_live_separate": 3,
        "family_live_together": 4,
        "sig_other": 5
}

if args.collapse_classes:
    contact_dict =  {
        "work": 0,
        "friend": 1,
        "family_live_separate": 1,
        "task": 2,
        "family_live_together": 3,
        "sig_other": 3
    }

if args.zimmerman_classes:
    contact_dict =  {
                "work": 0,
                "social": 1,
                "family": 2
            }

replace_dict = {
    'contact_type': contact_dict
}

# combines weak and med ties
if args.tie_str_verystrong:
    map_dict = {
        0: 0,
        1: 1,
        2: 1
    }
    train_data['tie_str_class'] = train_data['tie_str_class'].map(map_dict)

# combines strong and med ties
if args.tie_str_medstrong:
    map_dict = {
        0: 0,
        1: 0,
        2: 1
    }
    train_data['tie_str_class'] = train_data['tie_str_class'].map(map_dict)

train_data = train_data.replace(replace_dict)
test_data = test_data.replace(replace_dict)

train_y = train_data[args.predict_target]
train_X = train_data.drop(['combined_hash'] + predict_targets, axis=1, errors='ignore')
test_y = test_data[args.predict_target]
test_X = test_data.drop(['pid', 'combined_hash'] + predict_targets, axis=1, errors='ignore')

if args.age_gender_only:
    keep_cols = ['pid', 'ego_age', 'ego_gender_female', 'ego_gender_male', 'ego_gender_other']
    train_X = train_X[keep_cols]
    test_X = test_X[keep_cols[1:]]
    if args.test:
        print(train_X.columns)
        print(test_X.columns)

if args.run_time:
    run_time = int(args.run_time)

if args.task_time:
    task_time = int(args.task_time)

if args.test:
    # set some trivially small training time to test script
    run_time = 10
    task_time = 15
    print("Class counts %s" % Counter(train_y))

if args.rand_forest:
    estimators = ['random_forest']
    ensemble_size = 1
    ensemble_nbest = 1

if args.svc:
    estimators = ['libsvm_svc']
    ensemble_size = 1
    ensemble_nbest = 1
    
if args.resample:
    print("original shape %s" % Counter(train_y))

    sm = RandomOverSampler(random_state=rand_seed)

    if args.smote:
        sm = SMOTENC([0], random_state=rand_seed)

    if args.rand_downsample:
        sm = RandomUnderSampler(random_state=rand_seed)

    train_X, train_y = sm.fit_resample(train_X, train_y)

    print("resampled shape %s" % Counter(train_y))


    pid_groups = build_cv_groups(pd.Series(train_X[:,0])) # pid col
    train_X = train_X[:, 1:]

else:
    pid_groups = build_cv_groups(train_X['pid'])
    train_X = train_X.drop(['pid'], axis=1)

cv_method = sk_model_select.GroupKFold
cv_params = {
            'folds': 5,
            'groups': np.array(pid_groups)
            }

if args.group_res:
    cv_method = GroupKFold
    cv_params['random_state'] = rand_seed
    
# rebin for classification, 3 classes per previous literature
if args.emc_clf:
    _, bins = pd.qcut(train_y.append(test_y), 3, labels=False, retbins=True)
    bins[0] -= 0.001 # for non-inclusive lower bound on first bin
    train_y = pd.cut(train_y, bins, labels=False)
    test_y = pd.cut(test_y, bins, labels=False)

# classification task
if (args.predict_target in ['contact_type', 'tie_str_class']) or args.emc_clf:
    
    automl = AutoSklearnClassifier(
        per_run_time_limit=run_time,
        time_left_for_this_task=task_time,
        resampling_strategy=cv_method,
        resampling_strategy_arguments=cv_params,
        #initial_configurations_via_metalearning=0,
        ensemble_size=ensemble_size, 
        ensemble_nbest=ensemble_nbest,
        ensemble_memory_limit=ensemble_mem_limit,
        include_estimators=estimators,
        seed=rand_seed
    )

    clf_metric = autosklearn.metrics.accuracy
    if args.log_loss:
        clf_metric = autosklearn.metrics.log_loss
    if args.weighted_f1:
        clf_metric = autosklearn.metrics.f1_weighted
    if args.micro_f1:
        clf_metric = autosklearn.metrics.f1_micro
    if args.macro_f1:
        clf_metric = autosklearn.metrics.f1_macro

    automl.fit(train_X.copy(), train_y.copy(), metric=clf_metric)

else:
    automl = AutoSklearnRegressor(
        per_run_time_limit=run_time,
        time_left_for_this_task=task_time,
        resampling_strategy=cv_method,
        resampling_strategy_arguments=cv_params,
        #initial_configurations_via_metalearning=0,
        ensemble_size=ensemble_size, 
        ensemble_nbest=ensemble_nbest,
        include_estimators=estimators,
        seed=rand_seed
    )

    automl.fit(train_X.copy(), train_y.copy())

# refit() necessary when using cross-validation, see documentation:
# https://automl.github.io/auto-sklearn/stable/api.html#autosklearn.classification.AutoSklearnClassifier.refit
if args.group_res:
    resampler = RandomOverSampler(random_state=rand_seed)
    X_res, y_res = resampler.fit_resample(train_X.copy(), train_y.copy())
    automl.refit(X_res, y_res)
else:
    automl.refit(train_X.copy(), train_y.copy())
    
predictions = automl.predict(test_X)

if (args.predict_target in ['contact_type', 'tie_str_class']) or args.emc_clf:
    if args.weighted_f1:
        print("Weighted F1:", f1_score(test_y, predictions, average='weighted'))
    if args.micro_f1:
        print("Micro F1:", f1_score(test_y, predictions, average='micro'))

    else:
        print("Accuracy:", accuracy_score(test_y, predictions))
else:
    print("MSE:", mean_squared_error(test_y, predictions)) 

# model saving: https://github.com/automl/auto-sklearn/issues/5
pickle.dump(automl, open("{}.automl".format(args.out_name), "wb"))
pickle.dump(predictions, open("{}.predict".format(args.out_name), "wb"))
pickle.dump(automl.predict_proba, open("{}.predict_prob".format(args.out_name), "wb"))
print(automl.sprint_statistics())
