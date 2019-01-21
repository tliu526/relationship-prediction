"""Unit tests for feature extraction methods.


Test contact hashes are:
['1002060a7f4fe408f8137f12982e5d64cf34693',
'10413044ad5f1183e38f5ddf17259326e976231']

"""

import datetime
import os
import pickle

import numpy as np
import pandas as pd
import unittest

from feature_extract import *

class FeatureExtractTests(unittest.TestCase):
    
    def setUp(self):
        self.pid1 = '1002060'
        self.pid2 = '1041304'

        self.combined_hash1 = '1002060a7f4fe408f8137f12982e5d64cf34693'
        self.combined_hash2 = '10413044ad5f1183e38f5ddf17259326e976231'

        with open("../data/test_comm.df", 'rb') as comm_file:
            self.raw_df = pickle.load(comm_file)
            self.call_df = self.raw_df.loc[self.raw_df['comm_type'] == 'PHONE']
            self.sms_df = self.raw_df.loc[self.raw_df['comm_type'] == 'SMS']

        with open("../data/test_emm.df", 'rb') as emm_file:
            self.emm_df = pickle.load(emm_file)
    

    def test_init_feature_df(self):
        expected_dict = {
            (self.pid1, self.combined_hash1): [8, 2, 'friend'],
            (self.pid2, self.combined_hash2): [6, 3, 'family_live_together']
        }

        expected_df = pd.DataFrame.from_dict(expected_dict).T 
        expected_df.index = expected_df.index.rename(['pid', 'combined_hash'])
        expected_df = expected_df.rename({
                                            0: "total_comms", 
                                            1: "total_comm_days",
                                            2: "contact_type"
                                         }, 
                                         axis='columns')
        expected_df['total_comms'] = expected_df['total_comms'].astype(int)
        expected_df['total_comm_days'] = expected_df['total_comm_days'].astype(int)
        actual_df = init_feature_df(self.raw_df)

        pd.testing.assert_frame_equal(actual_df, expected_df)
    

    def test_build_count_features(self):
        actual_df = init_feature_df(self.raw_df)
        actual_df = build_count_features(actual_df, 
                                         self.call_df, 
                                         self.sms_df, 
                                         self.emm_df)
        
        # no NaNs should be present
        self.assertFalse(actual_df.isnull().values.any())
        
        columns = ['pid', 'combined_hash', 'total_comms', 'total_comm_days', 
                   'contact_type', 'total_calls', 'total_sms', 'total_sms_days',
                   'total_call_days', 'total_days', 'reg_call', 'reg_sms', 
                   'reg_comm']

        expected_dict = {
            0: [self.pid1, self.combined_hash1, 8, 2, 'friend', 0, 8, 2, 0, 58,
                0, 2/58, 2/58],
            1: [self.pid2, self.combined_hash2, 6, 3, 'family_live_together', 6, 
                0, 0, 3, 58, 3/58, 0, 3/58]
        }

        expected_df = pd.DataFrame.from_dict(expected_dict).T
        expected_df.columns = columns
        # expected_df[columns[2:4]] = expected_df[columns[2:4]].astype(int)        
        # expected_df[columns[5:]] = expected_df[columns[5:]].astype(float)
        # expected_df['total_days']= expected_df['total_days'].astype(int)

        pd.testing.assert_frame_equal(actual_df, expected_df, check_dtype=False)


    @unittest.skip("TODO finish calculating std dev")
    def test_build_intensity_features(self):
        # TODO test these: 'mean_in_call', 'std_in_call'
        columns = ['mean_out_call', 'std_out_call', 'mean_in_sms', 'std_in_sms',
                   'mean_out_sms', 'std_out_sms']

        expected_dict = {
            0: [0, 0, 6/58],
            1: [] 
        }

        actual_df = init_feature_df(self.raw_df)
        actual_df = build_count_features(actual_df, self.call_df, self.sms_df,
                                         self.emm_df)
        actual_df = build_intensity_features(actual_df, self.call_df, self.sms_df)
        print(actual_df[columns])

    @unittest.skip("TODO implement")
    def test_temporal_tendency_helper(self):
        pass


    @unittest.skip("TODO implement")
    def test_build_temporal_features(self):
        pass

    def test_build_channel_selection_features(self):
        columns = ['out_comm', 'call_tendency']

        expected_dict = {
            0: [2/8, 0],
            1: [4/6, 1]
        }

        expected_df = pd.DataFrame.from_dict(expected_dict).T
        expected_df.columns = columns        

        actual_df = init_feature_df(self.raw_df)
        actual_df = build_count_features(actual_df, self.call_df, self.sms_df,
                                         self.emm_df)
        actual_df = build_channel_selection_features(actual_df, self.raw_df)

        pd.testing.assert_frame_equal(actual_df[columns], expected_df)         
    
    
    def test_build_avoidance_features(self):
        columns = ['missed_in_calls', 'missed_out_calls', 'in_out_sms']

        expected_dict = {
            0: [np.nan, np.nan, 6/2],
            1: [np.nan, 2/4, np.nan], 
        }

        expected_df = pd.DataFrame.from_dict(expected_dict).T
        expected_df.columns = columns
        
        actual_df = init_feature_df(self.raw_df)
        actual_df = build_avoidance_features(actual_df, 
                                             self.call_df, 
                                             self.sms_df)  

        pd.testing.assert_frame_equal(actual_df[columns], expected_df) 


    def test_build_demo_features(self):
        """
        TODO test other demo features other than age/gender
        """
        columns = ['ego_age', 'ego_gender']

        demo_dict = {
            0: [self.pid1, 42, 'male'],
            1: [self.pid2, 55, 'female']    
        }

        demo_df = pd.DataFrame.from_dict(demo_dict).T
        demo_df.columns = ['pid', 'age', 'gender']

        expected_dict = {
            0: [42, 'male'],
            1: [55, 'female']
        }

        expected_df = pd.DataFrame.from_dict(expected_dict).T
        expected_df.columns = columns
        expected_df['ego_age'] = expected_df['ego_age'].astype(int)

        actual_df = init_feature_df(self.raw_df)
        actual_df = actual_df.reset_index()
        actual_df = build_demo_features(actual_df, demo_df)

        pd.testing.assert_frame_equal(actual_df[columns], expected_df)

if __name__ == '__main__':
    unittest.main()


        
        
