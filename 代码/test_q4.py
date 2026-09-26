"""Focused checks for fixed-schedule grouping and half-open resource intervals."""
import unittest

from q4 import Q4Model, SITE_INDEX, peak


class Q4Tests(unittest.TestCase):
    def test_reuse_at_same_time(self):
        self.assertEqual(peak([(0, 10), (10, 20)]), 1)

    def test_charging_overlap(self):
        self.assertEqual(peak([(0, 15), (10, 25)]), 2)

    def test_transitive_trip_components(self):
        model = Q4Model()
        sites = ('S003', 'S007', 'S011', 'S015')
        mask = sum(1 << SITE_INDEX[s] for s in sites)
        self.assertIn(mask, model.components)
        with self.assertRaises(ValueError):
            model.metrics(1 << SITE_INDEX['S003'])


if __name__ == '__main__':
    unittest.main()
