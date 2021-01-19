'''
Unit tests for the Lake Model.
'''

import sys
import unittest
from lake.lakemodel import lake, rbf


class TestRBF(unittest.TestCase):
    ''' 
    Test the radial basis function representation.
    '''

    # test the correctness of RBF evaluation with dictionary input
    def rbf_correct(self, case):
        result = rbf(case.get('X'), case.get('C'), case.get(
            'R'), case.get('W'))

        # use precision of 0.001
        assert round(result, 3) == round(case.get('expected'), 3)

    # test that RBF produces the correct values, as defined in Quinn et al. 2017
    def test_rbf(self):
        cases = []
        cases.append({
            'n': 1,
            'X': 0,
            'C': [-2],
            'R': [0],
            'W': [1],
            'expected': 0.01
        })
        cases.append({
            'n': 1,
            'X': 100,
            'C': [-2],
            'R': [1],
            'W': [1],
            'expected': 0.1
        })
        cases.append({
            'n': 1,
            'X': 1,
            'C': [0],
            'R': [3],
            'W': [1],
            'expected': 1/27
        })
        cases.append({
            'n': 2,
            'X': 1,
            'C': [0, 0],
            'R': [3, 3],
            'W': [1, 1],
            'expected': 2/27
        })
        cases.append({
            'n': 2,
            'X': 0,
            'C': [0, 0],
            'R': [0, 3],
            'W': [0, 1],
            'expected': 0.01
        })

        # assert that all cases are valid
        for case in cases:
            self.rbf_correct(case)


class TestLake(unittest.TestCase):
    ''' 
    Test the Lake DPS model.
    '''

    # test the correctness of lake problem evaluation with dictionary input
    def lake_correct(self, case):
        (objs, constr) = lake(case.get('C')[0], case.get('R')[0], case.get(
            'W')[0], case.get('C')[1], case.get('R')[1], case.get('W')[1])

        print(objs)

        # assert each returned objective value is correct, precision of 0.001
        for i, obj in enumerate(objs):
            assert round(obj, 3) == round(case.get('exp_objs')[i], 3)

        # assert returned constraint is correct
        assert round(constr[0], 3) == round(case.get('exp_constrs')[0], 3)

    # test lake model accuracy on precalculated inputs
    def test_lake(self):
        cases = []
        cases.append({
            'C': [0, 0],
            'R': [0, 0],
            'W': [0, 0],
            'exp_objs': [-0.173, 0.144, -1.0, -1.0],
            'exp_constrs': [0]
        })
        cases.append({
            'C': [1, 1],
            'R': [1, 1],
            'W': [1, 1],
            'exp_objs': [-1.536, 2.317, -0.980, -0.05],
            'exp_constrs': [0.8]
        })

        # assert correctness of all explicit cases
        for case in cases:
            self.lake_correct(case)
