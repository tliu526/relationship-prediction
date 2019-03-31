"""Utilities for the AutoML model.

"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics as sk_metrics
from sklearn.model_selection import BaseCrossValidator


def build_allq_features(feature_df, n=12, seed=2):
    """Samples n participants in each quartile from feature_df.

    Parameters
    ----------
    feature_df : pandas.DataFrame
        The input feature dataframe, must include 'ego_age_q' column.
    n : int
        The number of participants to sample from each quartile, defaults to 12.
    seed : int
        The random seed for numpy choice selection
    Returns
    -------
    (train_df, test_df) : tuple
        A tuple of the train and test feature splits.
    """

    q_pids = {}
    for i in range(1,5):
        cur_q = 'q' + str(i)
        q_pids[cur_q] = feature_df.loc[feature_df['ego_age_q'] == ('age_' + cur_q)]['pid'].unique()

    selected_pids = []
    for q, pids in q_pids.items():
        print(seed)
        np.random.seed(seed)
        selected_pids.extend(np.random.choice(pids, n, replace=False))
    print(len(selected_pids))
    print(sorted(selected_pids))
    
    train_df = feature_df.loc[feature_df['pid'].isin(selected_pids)]
    print(train_df.shape)
    test_df = feature_df.loc[~feature_df['pid'].isin(selected_pids)]
    print(test_df.shape)

    return train_df, test_df
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
    
    print_confusion_matrix(test_y, predictions, contact_types)


def print_confusion_matrix(test_y, predictions, contact_types):
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

        
def annotate_bar(ax):
    for i, bar in enumerate(ax.patches):
        height = bar.get_height()
        sign = np.sign(height)
        text = format(height, ".3f")
        #height*0.95
        plt.text((bar.get_x() + bar.get_width()/2), 0.01, text, ha='center')        


def plot_results(rf_test_scores, 
                 automl_test_scores, 
                 baseline_label, 
                 x_labels, 
                 y_label, 
                 title, 
                 majority_score=None,
                 automl_stderrs=None,
                 baseline_stderrs=None):
    x = np.arange(0,len(automl_test_scores))
    width = 0.35
    plt.rcParams["figure.figsize"] = [12,6]
    ax = plt.bar(x-(width/2), rf_test_scores, width, yerr=baseline_stderrs, label=baseline_label)
    annotate_bar(ax)
    ax = plt.bar(x+(width/2), automl_test_scores, width, yerr=automl_stderrs, label='auto-sklearn')
    annotate_bar(ax)

    if majority_score is not None:    
        plt.axhline(y=majority_score, color='black', ls='dotted', label='majority')
    plt.yticks(np.arange(0, 0.8, 0.05))
    plt.ylabel(y_label)
    plt.xlabel("Features")
    plt.xticks(x, x_labels)
    plt.title(title)
    plt.legend(loc='upper right')
    plt.show()
            
    
def print_ensemble(ensemble, latex=False):
    delim = ","
    end = ""
    for weight, pipeline in ensemble:
        if latex:
            print("{} & {} \\\\".format(weight, 
                                        pipeline.configuration['classifier:__choice__']))
        else:
            print("Weight: {}, classifier: {}".format(weight, 
                                                      pipeline.configuration['classifier:__choice__']))
            
            
if __name__ == '__main__':
    import pickle
    feature_dir = '/home/tliu/relationship-prediction/data/subpop_features/'
    features = 'base'
    df = pickle.load(open(feature_dir + "top5_{}_features.df".format(features), "rb"))
    seed = 3
    train_df, test_df = build_allq_features(df, seed=3)
    print(train_df['ego_age_q'].value_counts())
    pickle.dump(train_df, open(feature_dir + "top5_{}_allq_s{}_train_features.df".format(features, seed), "wb"), -1)
    pickle.dump(test_df, open(feature_dir + "top5_{}_allq_s{}_test_features.df".format(features, seed), "wb"), -1)
