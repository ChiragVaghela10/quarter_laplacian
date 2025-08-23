import unittest

import numpy as np

from imageprocessing.filters import QuarterLaplacian, LaplacianFilter


class TestQuarterLaplacian(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.qlf = QuarterLaplacian()

    def test_compute_qlf_result(self):
        expected = np.array(
            [[0.,       -0.66666667,    -0.66666667, 0.66666667,         0.],
             [-2.66666667, 2.66666667, -2.66666667, -2.66666667, 2.66666667],
             [-2.66666667, -2.66666667, 2.66666667, -2.66666667, 2.66666667],
             [-2.66666667, -2.66666667, -2.66666667, 2.66666667, 2.66666667],
             [0.,       0.66666667,     0.66666667, -0.66666667,        0.]])

        result = self.qlf._compute_qlf_result(U=np.arange(25).reshape(5, 5))
        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-8)

    def test_apply_filter(self):
        expected = np.array(
            [[0, 0, 1, 2, 4],
             [2, 3, 4, 10, 11],
             [7, 8, 9, 10, 16],
             [12, 13, 14, 15, 21],
             [20, 21, 22, 23, 24]])

        result = self.qlf.apply_filter(U=np.arange(25).reshape(5, 5), iterations=1)
        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-8)

class TestLaplacianFilter(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.laplacian = LaplacianFilter()

    def test_apply_filter(self):
        expected = np.array(
            [[0, 1, 2, 3, 4],
             [5, 6, 7, 8, 8],
             [10, 11, 12, 13, 13],
             [15, 16, 17, 18, 18],
             [19, 20, 21, 22, 23]])

        result = self.laplacian.apply_filter(U=np.arange(25).reshape(5, 5), iterations=1)
        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-8)
