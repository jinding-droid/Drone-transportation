"""Regression tests for relay resource assignment independent of list order."""
import json
import os
import tempfile
import unittest

import q3


class RelayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.link=q3.LinkModel()
        with open(os.path.join(q3.RESULT_DIR,'Q3_relay_optimized.json'),encoding='utf-8') as f:
            cls.config=json.load(f)

    def build(self,config):
        previous=q3.RESULT_DIR
        try:
            with tempfile.TemporaryDirectory(prefix='q3-relay-test-') as directory:
                q3.RESULT_DIR=directory
                with open(os.path.join(directory,'Q3_relay_optimized.json'),'w',encoding='utf-8') as f:
                    json.dump(config,f)
                return q3.build_relay_missions(self.link)
        finally:
            q3.RESULT_DIR=previous

    def test_order_independent(self):
        normal={m.mission_id:m for m in self.build(self.config)}
        reverse={m.mission_id:m for m in self.build(list(reversed(self.config)))}
        self.assertEqual(normal,reverse)

    def test_drone_names_can_swap(self):
        swapped=[dict(m,drone_id='R02' if m['drone_id']=='R01' else 'R01') for m in self.config]
        self.assertEqual(len(self.build(swapped)),3)

    def test_overlap_rejected(self):
        first=dict(self.config[0])
        duplicate=dict(first,mission_id='duplicate')
        with self.assertRaises(ValueError):
            self.build([first,duplicate])


if __name__ == '__main__':
    unittest.main()
