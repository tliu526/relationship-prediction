"""Short script to build train and test datasets.

TODO Hardcoded, for now.
"""

import argparse
import pickle

#from imblearn.over_sampling import SMOTE
from feature_extract import *


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Builds features from input dataframes.")
    parser.add_argument('out_name', help='output features name')
    parser.add_argument('--comm_file', help='communication df file')
    parser.add_argument('--emc_features', action='store_true', help='adds contact EMA features')
    parser.add_argument('--demo_features', action='store_true', help='adds demographic features')
    parser.add_argument('--age_gender_only', action='store_true', help='only use age/gender demographics')
    parser.add_argument('--loc_features', action='store_true', help='adds location features')
    parser.add_argument('--impute_missing', action='store_true', help='whether to impute NaNs and create additional features')
    args = parser.parse_args()
    
    # load data
    full_df = pickle.load(open(args.comm_file, "rb"))

    # TODO move to arguments
    emm_df = pickle.load(open("../data/emm_raw.df", "rb"))
    test_pids = pickle.load(open("../data/test_pids.list", "rb"))
    
    # feature extraction
    if args.comm_file is not None:
        full_features = comm_feature_extract(full_df, emm_df)

    if args.emc_features:
        emc_all = pickle.load(open('../data/emc_all.df', 'rb'))
        hash_dict = pickle.load(open('../data/emc_to_canonical.dict', 'rb'))
        pr_dict = pickle.load(open('../data/pr.dict', 'rb'))
        full_features = build_emc_features(full_features, full_df, emc_all, hash_dict, pr_dict)
        # emc_features tiles all combined_hashes, so drop all nans
        #full_features = full_features.dropna()

    if args.demo_features:
        demo_df = pickle.load(open('../data/demographics.df', 'rb'))
        full_features = build_demo_features(full_features, demo_df, args.age_gender_only)

    if args.loc_features:
        full_features = build_location_features(full_features, full_df)

    if args.impute_missing:
        full_features = build_nan_features(full_features)


    # split test and train data
    train_features = full_features.loc[~full_features['pid'].isin(test_pids)]
    print(train_features.shape)
    test_features = full_features.loc[full_features['pid'].isin(test_pids)]
    print(test_features.shape)

    pickle.dump(
        train_features, 
        open("../data/{}_train_features.df".format(args.out_name), "wb"), -1)
    pickle.dump(
        test_features, 
        open("../data/{}_test_features.df".format(args.out_name), "wb"), -1)