import unittest
import numpy as np
from make_data import grouped, missingness, regression
from conformal import conformal_quantile


class ProtocolTests(unittest.TestCase):
    def test_finite_sample_quantile_uses_order_statistic(self):
        self.assertEqual(9,conformal_quantile(np.arange(10),.1))
        self.assertEqual(8,conformal_quantile(np.arange(10),.2))
        self.assertTrue(np.isinf(conformal_quantile(np.arange(2),.01)))

    def test_independent_seeds_do_not_reuse_rows(self):
        self.assertFalse(regression(42).equals(regression(43)))
        self.assertTrue(regression(42).equals(regression(42)))

    def test_group_generator_and_missingness(self):
        frame=grouped()
        self.assertEqual(30,frame.groupby('group_id').size().min())
        frame=missingness()
        self.assertTrue(frame.x0.isna().any())
        self.assertFalse(frame.target.isna().any())


if __name__ == '__main__':
    unittest.main()
