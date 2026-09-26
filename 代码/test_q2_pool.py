"""Regression checks for count aggregation and exact box restoration."""
import json
import os
import unittest

from q2 import Q2Model, RESULT_DIR, hard_deadline
from q2_pool_search import Pool


class PoolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = Q2Model()
        cls.pool = Pool(cls.model)

    def test_classes_preserve_physics_and_deadlines(self):
        for key, ids in zip(self.pool.keys, self.pool.ids):
            for bid in ids:
                box = self.model.box_by_id[bid]
                self.assertEqual(key, (box['service'], box['mass_kg'], box['volume_m3'],
                    min(box['expected_time_s'], hard_deadline(box) or float('inf'))))

    def test_restore_candidate_covers_each_box_once(self):
        with open(os.path.join(RESULT_DIR, 'Q2_pool_candidate.json'), encoding='utf-8') as f:
            batches = json.load(f)['batches']
        patterns = [(self.pool.counts(b), 1) for b in batches]
        restored = self.pool.restore(patterns)
        self.assertEqual(sorted(b for batch in restored for b in batch), sorted(self.model.box_by_id))
        for old, new in zip(batches, restored):
            variants_old = self.model.variants(tuple(sorted(old)))
            variants_new = self.model.variants(new)
            self.assertEqual(len(variants_old), len(variants_new))
            for a, b in zip(variants_old, variants_new):
                self.assertEqual((a.aircraft_type, a.order), (b.aircraft_type, b.order))
                self.assertAlmostEqual(a.energy_kwh, b.energy_kwh)
                self.assertAlmostEqual(a.duration_s, b.duration_s)

    def test_pattern_deduplication(self):
        pool = Pool(self.model)
        pool.add(((0, 1), (1, 0)))
        pool.add(((0, 1),))
        self.assertEqual(len(pool.patterns), 1)


if __name__ == '__main__':
    unittest.main()
