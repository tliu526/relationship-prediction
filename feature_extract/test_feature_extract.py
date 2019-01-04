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

        self.raw_df = pickle.load(open("../data/test_comm.df", 'rb'))
        
        # # combined hash counts
        # self.num_a1 = 3
        # self.num_a2 = 2
        # self.num_b1 = 4
        # self.num_b2 = 1

        # self.raw_df = pd.DataFrame(columns=['pid', 'combined_hash', 'contact_type'])
        # self.raw_df['pid'] = ['a']*5 + ['b']*5
        # self.raw_df['combined_hash'] = ['a1'] * self.num_a1 + ['a2'] * self.num_a2 + \
        #                                ['b1'] * self.num_b1 + ['b2'] * self.num_b2
        # self.raw_df['contact_type'] = ['family_live_together'] * self.num_a1 + \
        #                               ['family_live_separate'] * self.num_a2 + \
        #                               ['work'] * self.num_b1 + \
        #                               ['friend'] * self.num_b2
        # self.raw_df['date_days'] = [datetime.datetime(2018,1,x) for x in range(1, self.num_a1 + 1)] + \
        #                            [datetime.datetime(2018,1,x) for x in range(1, self.num_a2 + 1)] + \
        #                            [datetime.datetime(2018,1,1)] * self.num_b1 + \
        #                            [datetime.datetime(2018,1,1)]

    
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
        """
        TODO make sure to check the NaN cases
        """
        # total_calls, total_sms, total_sms_days, total_call_days, total_days, reg_calls, reg_sms, reg_comm


if __name__ == '__main__':
    unittest.main()


        
        

