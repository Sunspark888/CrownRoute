import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np

from crownroute.pipeline import generate_candidates
from scripts.build_supervision import assemble
from scripts.make_c1_demo import make_demo


class ConstructionTests(unittest.TestCase):
    def test_frozen_generation_settings_and_mask_filtering(self):
        masks = np.full((3, 16, 16), 0.5, dtype=np.float32)
        masks[1, :11, :9] = 0.6  # 99 foreground pixels.
        masks[2, :10, :10] = 0.6  # 100 foreground pixels.
        result = Mock()
        result.masks.data.cpu.return_value.numpy.return_value = masks
        model = Mock(return_value=[result])
        candidates = generate_candidates(model, Path("example.png"), "cpu")
        model.assert_called_once_with(
            "example.png", device="cpu", imgsz=512, conf=0.25, iou=0.9, retina_masks=True
        )
        self.assertEqual(len(candidates), 1)
        self.assertEqual(int(candidates[0].sum()), 100)

    def test_real_construction_and_standard_membership_on_synthetic_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            manifest = make_demo(base / "inputs")
            payload = json.loads(manifest.read_text())
            # Candidate selection is independent across boxes of the same patch.
            payload["patches"][0]["boxes"].append([20, 20, 76, 76])
            manifest.write_text(json.dumps(payload))
            dataset = base / "standard"
            assemble(manifest, dataset)
            outcomes = json.loads((dataset / "meta/route_outcomes.json").read_text())
            self.assertEqual([r["outcome"] for r in outcomes],
                             ["Final-A", "Final-A", "Final-B", "Final-C", "Final-C", "Final-C"])
            self.assertEqual([r["reason"] for r in outcomes[-3:]],
                             ["below_acceptance_threshold", "ineligible", "no_candidate"])
            self.assertIsNone(outcomes[0]["admission_score"])
            self.assertGreaterEqual(outcomes[2]["admission_score"], 0.5401658842336331)
            rows = json.loads((dataset / "meta/dataset_records.json").read_text())
            self.assertEqual([r["class_type"] for r in rows], ["A", "A", "B"])
            self.assertEqual((dataset / "split/train.txt").read_text().splitlines(),
                             [r["id"] for r in rows])
            self.assertTrue(all((dataset / r["mask_path"]).is_file() for r in rows))
            self.assertEqual(len(list((dataset / "masks").iterdir())), 3)


if __name__ == "__main__":
    unittest.main()
