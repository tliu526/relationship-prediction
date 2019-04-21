"""
Statistics convenience methods.
"""

import numpy as np

from rpy2.robjects import r, pandas2ri
import rpy2.robjects as robjects
import rpy2
from rpy2.robjects.packages import importr
utils = importr('utils')
 # TODO is this line still needed?
lmtest = importr('lmtest')
# https://stackoverflow.com/questions/32983365/rpy2-cannot-find-installed-external-r-packages
Hmisc = importr("Hmisc")
pandas2ri.activate()


def run_r_corr(df, corr_type='spearman', p_correction='BH'):
    """
    Runs R correlation calculations and p-value corrections on the given dataframe.
    
    :returns: a tuple of (correlations, counts, p_values)
    """
    num_cols = len(df.columns.values)
    r_dataframe = pandas2ri.py2ri(df)
    r_as = r['as.matrix']
    rcorr = r['rcorr'] 
    r_p_adjust = r['p.adjust']
    result = rcorr(r_as(r_dataframe), type=corr_type)
    rho = result[0]
    n = result[1]
    p = result[2]
    
    if p_correction is not None:
        p = r_p_adjust(p, p_correction)
    r_corrs = pandas2ri.ri2py(rho)
    r_p_vals = pandas2ri.ri2py(p)
    r_counts = pandas2ri.ri2py(n)
    r_p_vals = np.reshape(r_p_vals, (num_cols,num_cols))
    return r_corrs, r_counts, r_p_vals


def build_corr_mat(corrs, p_vals, labels, title, alpha):
    """
    returns the matplotlib plt object for the specified correlations.
    """
    plt.rcParams["figure.figsize"] = [20,12]
    plt.imshow(corrs)
    for i in range(len(labels)):
        for j in range(len(labels)):
            text = "{0:.2f}".format(r_corrs[i, j])
            p = p_vals[i,j]
            if p < alpha / len(labels):
                text = text + "*"
            plt.text(j,i, text, ha="center", va="center", color="w")
    plt.xticks([x for x in range(len(labels))], labels, rotation=45, ha="right", rotation_mode='anchor')
    plt.yticks([x for x in range(len(labels))], labels)
    plt.colorbar()
    plt.title(title)
    return plt


def get_sig_features(cols, target, corrs, p_vals, exclude_cols=[], alpha=0.05):
    """Returns dataframe of features significantly correlated with the target variable.
    
    Args:
        cols (list): list of column names
        target (str): target feature to extract correlations from
        corrs (numpy.array): correlation matrix
        p_vals (numpy.array): p value matrix
        exclude_cols (list): features to exclude
        alpha (float): significance threshold
        
    Returns:
        pandas.DataFrame: df of significant features
        
    """
    idx = np.where(cols == target)
    sel_corrs = corrs[idx]
    sel_p_vals = p_vals[idx]
    sel_stats = np.transpose(np.vstack((sel_corrs, sel_p_vals)))
    sel_df = pd.DataFrame(sel_stats, index=cols, columns=['corr', 'p'])
    sig_features = sel_df.loc[sel_df['p'] < alpha]
    sig_features = sig_features[~sig_features.index.isin(exclude_cols)]
    sig_features = sig_features.sort_values(by='p')
    
    return sig_features
