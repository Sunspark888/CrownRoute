import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.build_c2_configs import assemble, load_standard
from scripts.make_c1_demo import write_png
from scripts.recover_training_delta import recover


class C2CompositionTests(unittest.TestCase):
    def test_recovery_and_source_matched_substitution(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            yy, xx = np.indices((96, 96))
            textured = np.zeros((96, 96, 3), dtype=np.uint8)
            textured[:, :, 1] = 80 + ((xx + yy) % 2) * 150
            write_png(root / "01_patch_demo.png", textured)
            write_png(root / "01_patch_blank.png", np.zeros_like(textured))
            split_csv = root / "sources.csv"
            split_csv.write_text("source_id,split\n01,train\n02,train\n03,val\n")
            inventory = []
            for name, index in [("01_patch_demo.png", 1), ("01_patch_demo.png", 0),
                                ("01_patch_blank.png", 0), ("03_patch_missing.png", 0)]:
                inventory.append({"class_type":"C", "patch_image":name, "gt_index":index,
                                  "gt_box":[20, 20, 76, 76]})
            input_path = root / "classified.json"
            input_path.write_text(json.dumps(inventory))
            delta = root / "delta"
            result = recover(input_path, split_csv, root, delta)
            self.assertEqual(result, {"training_intermediate_C": 3, "recovered": 2, "excluded": 1})
            recovered = json.loads((delta / "records.json").read_text())
            self.assertTrue(all(r["recovery_diagnostics"]["status"] == "ok" for r in recovered))
            self.assertEqual(json.loads((delta / "outcomes.json").read_text())[0]["status"], "no_initial_component")

            standard = root / "standard"
            (standard / "meta").mkdir(parents=True)
            (standard / "split").mkdir()
            write_png(standard / "image.png", textured)
            write_png(standard / "mask.png", (textured[:, :, 1] > 0).astype(np.uint8) * 255)
            rows = []
            for i, (sid, cls) in enumerate([("01","A"), ("01","B"), ("02","A"),
                                            ("02","B"), ("02","B"), ("03","A")]):
                rows.append({"id":f"sample{i}", "class_type":cls, "source_disjoint_v3_source_id":sid,
                             "source_disjoint_v3_split":"val" if sid == "03" else "train",
                             "image_path":"image.png", "mask_path":"mask.png", "patch_image":"image.png"})
            (standard / "meta/dataset_records.json").write_text(json.dumps(rows))
            (standard / "split/train.txt").write_text("".join(r["id"] + "\n" for r in rows[:-1]))
            (standard / "split/val.txt").write_text("sample5\n")
            output = root / "three"
            result = assemble(standard, recovered, output, standard, delta)
            self.assertEqual(result, {"standard": 5, "relaxed": 7, "fixed": 5, "validation": 1,
                                      "within_source": 1, "proportional": 1})
            fixed = load_standard(output / "FIXED_N")[1]
            self.assertEqual([r["id"] for r in fixed[-2:]], [r["id"] for r in recovered])
            self.assertTrue(all((output / "FIXED_N" / r["mask_path"]).is_file() for r in fixed))
            self.assertEqual(sum(r["class_type"] == "B" for r in fixed), 1)
            (standard / "split/test.txt").write_text("")
            with self.assertRaisesRegex(ValueError, "test split"):
                load_standard(standard)


if __name__ == "__main__":
    unittest.main()
