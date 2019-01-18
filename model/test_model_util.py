"""Unit tests for model utilities.

"""

import numpy as np
import pandas as pd
import unittest

from model_util import *

class ModelUtilTests(unittest.TestCase):

    def test_build_cv_groups(self):
        test_pids = pd.Series(['a', 'a', 'b', 'b', 'a', 'c', 'c', 'd', 'd'])
        expected_groups = [0,0,1,1,0,2,2,3,3]
        actual_groups = build_cv_groups(test_pids)

        assert actual_groups == expected_groups

if __name__ == '__main__':
    unittest.main()
