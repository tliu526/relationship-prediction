"""Utilities for running the AutoML model.

"""

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