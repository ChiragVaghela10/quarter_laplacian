import unittest

import numpy as np

from imageprocessing.filters import QuarterLaplacian


class TestQuarterLaplacian(unittest.TestCase):
    def test_compute_qlf_result(self):
        qlf = QuarterLaplacian()
        expected = np.array(
            [[0., -0.66666667, -0.66666667, 0.66666667, 0.],
            [-2.66666667, 2.66666667, -2.66666667, -2.66666667, 2.66666667],
            [-2.66666667, -2.66666667, 2.66666667, -2.66666667, 2.66666667],
            [-2.66666667, -2.66666667, -2.66666667, 2.66666667, 2.66666667],
            [0., 0.66666667, 0.66666667, -0.66666667, 0.]])

        result = qlf._compute_qlf_result(U=np.arange(25).reshape(5, 5))
        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-8)

