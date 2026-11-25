import unittest
import json
from pathlib import Path
import torch
from geometry_metrics import recurrence, budget, score_map


class TestGeometryMetrics(unittest.TestCase):
    def test_recurrence_bounds(self):
        eye=torch.eye(4)
        self.assertAlmostEqual(recurrence(eye,eye,2),1.0)
        self.assertAlmostEqual(recurrence(eye,eye[:,[2,3,0,1]],2),0.0)

    def test_full_rank_coverage_and_budget(self):
        eye=torch.eye(4)
        grads={'a':torch.arange(1,17,dtype=torch.float32).reshape(4,4)}
        self.assertEqual(budget({'a':4},{'a':(4,4)}),32)
        result=score_map({'a':4},{'a':eye},{'a':eye},grads)
        self.assertAlmostEqual(result['q_grad'],1.0,places=6)
        self.assertAlmostEqual(result['q_geom'],0.0,places=6)

    def test_nonorthogonal_rejected(self):
        eye=torch.eye(4)
        with self.assertRaises(AssertionError):recurrence(eye*2,eye,2)

    def test_overlap_and_budget_mismatch_rejected(self):
        ids=json.loads((Path(__file__).resolve().parents[1]/'data/sample_ids_A_B1_B2.json').read_text())
        for task,parts in ids.items():
            a,b1,b2=(set(parts[k]) for k in ('A','B1','B2'))
            self.assertEqual((len(a),len(b1),len(b2)),(512,128,128),task)
            self.assertFalse(a&b1 or a&b2 or b1&b2,task)
        with self.assertRaises(AssertionError):budget({'a':2},{'b':(4,4)})


if __name__=='__main__':unittest.main()
