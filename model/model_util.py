"""Utilities for the AutoML model.

"""

import numpy as np
import pandas as pd
import sklearn.metrics as sk_metrics


def build_cv_groups(pids):
    """Builds cross-validation groups for GroupKFold model selection class.

    Parameters
    ----------
    pids : pandas.Series
        Series of pids, for example pulled from a pandas.DataFrame column.

    Yields
    ------
    groups : list
        A list with indices corresponding to each group entry in pids.
    """
    unique_pids = pids.unique().tolist()
    
    groups = [unique_pids.index(x) for x in pids.values]
    
    return groups


def print_clf_metrics(test_y, predictions, contact_types):
    """Prints classification metrics for the given predictions.

    - Micro statistics take global counts of TP, FP, etc
    - Macro statistics take per class metrics and averages them,not accounting 
    for class imbalance
    - Weighted statistics weight macro metrics by number of true examples in each class
    """
    print("Accuracy:", sk_metrics.accuracy_score(test_y, predictions))
    
    # precision, recall, F1
    metrics = np.zeros((2,3))
    
    # micro
    avgs = ['macro', 'weighted']
    for i, avg in enumerate(avgs):
        metrics[i,0] = sk_metrics.precision_score(test_y, predictions, average=avg)
        metrics[i,1] = sk_metrics.recall_score(test_y, predictions, average=avg)
        metrics[i,2] = sk_metrics.f1_score(test_y, predictions, average=avg)

    metrics_df = pd.DataFrame(metrics, index=avgs, columns=['precision', 'recall', 'F1'])
    display(metrics_df)
    
    confusion_mat = sk_metrics.confusion_matrix(test_y, predictions)
    confuse_df = pd.DataFrame(confusion_mat, index=contact_types, columns=["p_" + x for x in contact_types])
    display(confuse_df)


def get_best_val_score(model):
    """Extracts the best validation score from sprint_statistics() call.
    
    """
    stats = model.sprint_statistics()
    target_str = "Best validation score: "
    start_idx = stats.index(target_str) + len(target_str)
    end_idx = stats.index("\n", start_idx)

    return float(stats[start_idx:end_idx])