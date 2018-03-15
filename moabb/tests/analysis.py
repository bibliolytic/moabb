import unittest
import pkg_resources
import shutil
import numpy as np
import moabb.analysis.meta_analysis as ma
from moabb.analysis.results import Results
from moabb.analysis import 
import os
from moabb.datasets.base import BaseDataset
from moabb.evaluations.base import BaseEvaluation
from moabb.paradigms.base import BaseParadigm
# dummy evaluation


class DummyEvaluation(BaseEvaluation):

    def evaluate(self, dataset, subject, clf, paradigm):
        raise NotImplementedError('dummy')

    def preprocess_data(self):
        pass


class DummyParadigm(BaseParadigm):

    def __init__(self):
        pass

    def scoring(self):
        raise NotImplementedError('dummy')


# dummy datasets
class DummyDataset(BaseDataset):

    def __init__(self, code):
        """

        """
        super().__init__(list(range(5)), 2, {
            'a': 1, 'b': 2}, code, [1, 2], 'imagery')


# Create dummy data for tests
d1 = {'time': 1,
      'dataset': DummyDataset('d1'),
      'id': 1,
      'score': 0.9,
      'n_samples': 100,
      'n_channels': 10}

d2 = {'time': 2,
      'dataset': DummyDataset('d1'),
      'id': 2,
      'score': 0.9,
      'n_samples': 100,
      'n_channels': 10}


d3 = {'time': 2,
      'dataset': DummyDataset('d2'),
      'id': 2,
      'score': 0.9,
      'n_samples': 100,
      'n_channels': 10}

d4 = {'time': 2,
      'dataset': DummyDataset('d2'),
      'id': 1,
      'score': 0.9,
      'n_samples': 100,
      'n_channels': 10}


def to_result_input(pnames, dsets):
    return dict(zip(pnames, dsets))


class Test_Stats(unittest.TestCase):

    def test_rmanova(self):
        matrix = np.asarray([[45, 50, 55],
                             [42, 42, 45],
                             [36, 41, 43],
                             [39, 35, 40],
                             [51, 55, 59],
                             [44, 49, 56]])
        f, p = ma._rmanova(matrix)
        self.assertAlmostEqual(f, 12.53, places=2)
        self.assertAlmostEqual(p, 0.002, places=3)

    def test_interleave(self):
        a = np.array([1,3,5])
        b = np.array([2,4,6])
        o = ma.interleave_vectors(a,b)
        self.assertTrue((np.arange(1,7) == o.ravel()).all(),o)

    def test_permmatrix(self):
        b = np.array([False, False, True])
        intended = [[1,-1,0,0,0,0],
                    [0,0,1,-1,0,0],
                    [0,0,0,0,-1,1]]
        M = ma.generate_perm_matrix(b)
        self.assertTrue(np.array_equal(np.array(intended),M),M)

    def test_null(self):
        data = np.array([[1,2],
                         [3,4],
                         [5,6]])
        samples = ma.return_null_distribution(data[:,0],data[:,1],10000).astype(int)
        self.assertTrue(np.in1d(samples,np.array([1,-1,3,-3])).all())
        
        

class Test_Integration(unittest.TestCase):

    def setUp(self):
        self.obj = Results(evaluation_class=type(DummyEvaluation()),
                           paradigm_class=type(DummyParadigm()),
                           suffix='test')

    def tearDown(self):
        path = self.obj.filepath
        if os.path.isfile(path):
            os.remove(path)

    def test_rmanova(self):
        _in = to_result_input(['a', 'b', 'c'], [[d1]*5, [d1]*5, [d4]*5])
        self.obj.add(_in)
        _in = to_result_input(['a', 'b', 'c'], [[d2]*5, [d2]*5, [d3]*5])
        self.obj.add(_in)
        df = self.obj.to_dataframe()
        ma.rmANOVA(df)


class Test_Results(unittest.TestCase):

    def setUp(self):
        self.obj = Results(evaluation_class=type(DummyEvaluation()),
                           paradigm_class=type(DummyParadigm()),
                           suffix='test')

    def tearDown(self):
        path = self.obj.filepath
        if os.path.isfile(path):
            os.remove(path)

    def testCanAddSample(self):
        self.obj.add(to_result_input(['a'], [d1]))

    def testRecognizesAlreadyComputed(self):
        _in = to_result_input(['a'], [d1])
        self.obj.add(_in)
        not_yet_computed = self.obj.not_yet_computed(
            {'a': 1}, d1['dataset'], d1['id'])
        self.assertTrue(len(not_yet_computed) == 0)

    def testCanAddMultiplePipelines(self):
        _in = to_result_input(['a', 'b', 'c'], [d1, d1, d2])
        self.obj.add(_in)

    def testCanAddMultipleValuesPerPipeline(self):
        _in = to_result_input(['a', 'b'], [[d1, d2], [d2, d1]])
        self.obj.add(_in)
        not_yet_computed = self.obj.not_yet_computed(
            {'a': 1}, d1['dataset'], d1['id'])
        self.assertTrue(len(not_yet_computed) == 0, not_yet_computed)
        not_yet_computed = self.obj.not_yet_computed(
            {'b': 2}, d2['dataset'], d2['id'])
        self.assertTrue(len(not_yet_computed) == 0, not_yet_computed)
        not_yet_computed = self.obj.not_yet_computed(
            {'b': 1}, d1['dataset'], d1['id'])
        self.assertTrue(len(not_yet_computed) == 0, not_yet_computed)

    def testCanExportToDataframe(self):
        _in = to_result_input(['a', 'b', 'c'], [d1, d1, d2])
        self.obj.add(_in)
        _in = to_result_input(['a', 'b', 'c'], [d2, d2, d3])
        self.obj.add(_in)
        df = self.obj.to_dataframe()
        self.assertTrue(set(np.unique(df['pipeline'])) == set(
            ('a', 'b', 'c')), np.unique(df['pipeline']))
        self.assertTrue(df.shape[0] == 6, df.shape[0])

if __name__ == "__main__":
    unittest.main()
