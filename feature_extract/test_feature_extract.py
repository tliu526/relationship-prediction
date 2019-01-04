"""Unit tests for feature extraction methods."""

import datetime
import os

import numpy as np
import pandas as pd
import unittest

from feature_extract import *

class FeatureExtractTests(unittest.TestCase):
    
    raw_df = pd.DataFrame()
    
    def setUp(self):
        pass
    
    def test_init_feature_df(self):

        # combined hash counts
        num_a1 = 3
        num_a2 = 2
        num_b1 = 4
        num_b2 = 1

        raw_df = pd.DataFrame(columns=['pid', 'combined_hash', 'contact_type'])
        raw_df['pid'] = ['a']*5 + ['b']*5
        raw_df['combined_hash'] = ['a1'] * num_a1 + ['a2'] * num_a2 + \
                                  ['b1'] * num_b1 + ['b2'] * num_b2
        raw_df['contact_type'] = ['family_live_together'] * num_a1 + \
                                 ['family_live_separate'] * num_a2 + \
                                 ['work'] * num_b1 + \
                                 ['friend'] * num_b2
        raw_df['date_days'] = [datetime.datetime(2018,1,x) for x in range(1,num_a1 + 1)] + \
                              [datetime.datetime(2018,1,x) for x in range(1,num_a2 + 1)] + \
                              [datetime.datetime(2018,1,1)] * num_b1 + \
                              [datetime.datetime(2018,1,1)]
                              


        """
        the gewse sat on de toilet and made a beeeeeeeeeeg boom at clogged de toilet
        de gewse den sat on her gewse and made a beeeeeeeeeg tewt
        de gewse den sat on her gewse and berk de beep boop code 
        eheheehekslhfksdhferheejeehejejeheeuehrughughehehughggueheueuehueheueheue
        """

        expected_dict = {
            ('a', 'a1'): [num_a1, num_a1, 'family_live_together'],
            ('a', 'a2'): [num_a2, num_a2, 'family_live_separate'],
            ('b', 'b1'): [num_b1, 1, 'work'],
            ('b', 'b2'): [num_b2, 1, 'friend']
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
        actual_df = init_feature_df(raw_df)

        pd.testing.assert_frame_equal(actual_df, expected_df)
    
    
    def test_build_count_features(self):
        """
        TODO make sure to check the NaN cases
        """

if __name__ == '__main__':
    unittest.main()


        
        

