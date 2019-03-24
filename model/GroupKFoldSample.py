"""
Subclass of scikit-learn's GroupKFold cross validator to perform re-sampling
within each fold.

Due to how scikit-learn implements the split() function via indices, only
random oversampling and random undersampling can be performed.
"""

from collections import Counter
from imblearn.over_sampling import RandomOverSampler
import numpy as np
import pandas as pd
import pickle

import sklearn
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.preprocessing import normalize
from sklearn.metrics import f1_score, accuracy_score
import sklearn.model_selection as sk_model_select
from sklearn.utils.validation import indexable

from model_util import build_cv_groups

class GroupKFold(sk_model_select.GroupKFold):
    def __init__(self, n_splits, random_state):
        super().__init__(n_splits)
        self.random_state = random_state

    def split(self, X, y=None, groups=None):
        """
        Calls GroupKFold split function, then resamples the training index.
        """
        #train_idx, test_idx = super().split(X, y=y, groups=groups)
        for train_idx, test_idx in super().split(X, y=y, groups=groups):
            #X, y, groups = indexable(X, y, groups)
            fold_train_X = X[train_idx]        
            if hasattr(y, "values"):
                fold_train_y = y.values[train_idx]        
            else:
                fold_train_y = y[train_idx]          
            # print(X.shape)
            # print(y.shape)
            # print(fold_train_X.shape)
            # print(fold_train_y.shape)

            # Need to remap fold indices back to the overall indices
            idx_dict = {fold_idx : train_idx[fold_idx] for fold_idx in range(len(fold_train_y))}

            # print(Counter(y.values[train_idx]))
            sm = RandomOverSampler(random_state=self.random_state)
            # X_res, y_res actually thrown away
            X_res, y_res = sm.fit_resample(fold_train_X, fold_train_y)
            idx_res = sm.sample_indices_
            train_idx_res = [idx_dict[idx] for idx in idx_res]
            # print(Counter(y_res))
            # print(Counter(y.values[train_idx_res]))

            yield train_idx_res, test_idx


if __name__ == '__main__':
    predict_targets = [
    'contact_type',
    'q1_want',
    'q2_talk',
    'q3_loan',
    'q4_closeness'
    ]

    replace_dict_4clf = {
        'contact_type': {
            "work": 0,
            "friend": 1,
            "task": 2,
            "family_live_separate": 1,
            "family_live_together": 3,
            "sig_other": 3
        }
    }

    replace_dict_6clf = {
        'contact_type': {
            "work": 0,
            "friend": 1,
            "task": 2,
            "family_live_separate": 3,
            "family_live_together": 4,
            "sig_other": 5
        }
    }

    contact_types_6clf = list(replace_dict_6clf['contact_type'].keys())
    contact_types_4clf = ["work", "social", "task", "family_together"]

    age_qlabels = ["age_q" + str(x) for x in range(1,5)]

    top5_all_sb_train_data = pickle.load(open('../data/final_sandbox/top5_all_train_features.df', 'rb'))
    top5_all_sb_test_data = pickle.load(open('../data/final_sandbox/top5_all_test_features.df', 'rb'))

    top5_all_data = pd.concat([top5_all_sb_train_data, top5_all_sb_test_data], axis=0)
    top5_all_data = top5_all_data.replace(replace_dict_4clf)
    top5_all_data['ego_age_q'], bins = pd.qcut(top5_all_data['ego_age'], q=4, labels=age_qlabels, retbins=True)

    """Try a random forest estimator"""
    top5_all_q1 = top5_all_data.loc[top5_all_data['ego_age_q'] == 'age_q1']
    top5_all_q1_train_y = top5_all_q1['contact_type']
    top5_all_q1_train_X = top5_all_q1.drop(['pid', 'combined_hash', 'ego_age_q'] + predict_targets, axis=1, errors='ignore')
    top5_all_q1_train_X = normalize(top5_all_q1_train_X)

    groups = build_cv_groups(top5_all_q1['pid'])

    group_kfold_res = GroupKFold(n_splits=5, random_state=2)

    cv_scores = []

    for fold_train_idx, fold_test_idx in group_kfold_res.split(top5_all_q1_train_X, top5_all_q1_train_y, groups):

        top5_all_q1_clf = RandomForestClassifier(random_state=2, n_jobs=4, n_estimators=100, warm_start=False)
        fold_train_X = top5_all_q1_train_X[fold_train_idx]
        fold_train_y = top5_all_q1_train_y.values[fold_train_idx]
        fold_test_X = top5_all_q1_train_X[fold_test_idx]
        fold_test_y = top5_all_q1_train_y.values[fold_test_idx]
        
        print(Counter(fold_train_y))
        top5_all_q1_clf.fit(fold_train_X, fold_train_y)
        predict = top5_all_q1_clf.predict(fold_test_X)
        cv_scores.append(accuracy_score(fold_test_y, predict))

    print(cv_scores)    
    print(np.mean(cv_scores))