'''
Unit tests for optimized Lake Model. Comparison with lakemodel.py
'''

import unittest
import numpy as np
from lake.lakemodel import lake
from lake.lakemodel_fast import lake_fast


class TestOptimized(unittest.TestCase):
    ''' 
    Compare the optimized and standard lake model. Assert they produce the same 
    result.
    '''

    # generate random RBF weight inputs, as defined in Quinn et al. 2017
    def random_inputs(self):
        C = np.random.uniform(-2, 2, size=2)
        R = np.random.uniform(0, 2, size=2)
        W = np.random.uniform(0, 1, size=2)
        return [C[0], R[0], W[0], C[1], R[1], W[1]]

    # ensure objective and constraint values are equal for both implementations
    def is_equal(self, fast, slow):
        return (np.around(fast[0], 3).all() == np.around(slow[0], 3).all()) \
            and (round(fast[1][0], 3) == round(slow[1][0], 3))

    # generate random input and compare the optimized and normal
    def test_compare(self):
        n_compares = 100
        for _ in range(n_compares):
            inputs = self.random_inputs()
            assert self.is_equal(lake_fast(*inputs), lake(*inputs))
