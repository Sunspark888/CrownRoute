"""Small CPU check with a synthetic single-crown patch and no model download."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
import torchvision
from PIL import Image

from model import build_model
from train_c1 import QualityDataset
from validation import P06Evaluator, encode_rle, safe_mask_ap


def write_json(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def main():
    torch.manual_seed(42)
    torch.set_num_threads(2)
    with TemporaryDirectory() as directory:
        root = Path(directory)
        data = root / "data"
        samples = root / "manual_gt_assets"
        reference = root / "manual_gt_assets_final5b_repaired"
        for path in (data / "meta", data / "split", samples, reference):
            path.mkdir(parents=True)
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[8:22, 9:25] = 1
        Image.fromarray(np.repeat((mask * 110)[..., None], 3, axis=2)).save(
            data / "tile.png"
        )
        Image.fromarray(mask * 255).save(data / "mask.png")
        record = dict(
            id="one",
            image_path="tile.png",
            mask_path="mask.png",
            patch_image="tile.png",
            class_type="A",
        )
        write_json(data / "meta/dataset_records.json", [record])
        for split in ("train", "val"):
            (data / f"split/{split}.txt").write_text("one\n", encoding="utf-8")
        image, target = QualityDataset(data, "train")[0]
        assert target["boxes"].tolist() == [[9, 8, 25, 22]]
        assert int(target["masks"].sum()) == int(mask.sum())
        gt = {
            "images": [
                {
                    "id": 1,
                    "file_name": "tile.png",
                    "patch_stem": "tile",
                    "height": 32,
                    "width": 32,
                }
            ],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "iscrowd": 0,
                    "area": int(mask.sum()),
                    "bbox": [9, 8, 16, 14],
                    "segmentation": encode_rle(mask),
                }
            ],
            "categories": [{"id": 1, "name": "crown"}],
        }
        write_json(reference / "normalized_gt_val_manual_coco.json", gt)
        write_json(
            samples / "gt_val_samples.json",
            [{"id": 1, "patch_stem": "tile", "file_name": "../data/tile.png"}],
        )
        evaluator = P06Evaluator(data, reference, root / "evaluations")
        assert (
            Path(evaluator.samples[0]["file_name"]).resolve()
            == (data / "tile.png").resolve()
        )
        pred = {
            "image_id": 1,
            "category_id": 1,
            "score": 0.95,
            "segmentation": encode_rle(mask),
        }
        for predictions, expected in (([pred], 1.0), ([], 0.0)):
            scores, failed_gt, failed_pred, _ = safe_mask_ap(
                "synthetic", "saved", "val", gt, predictions, [1], 100
            )
            assert (
                abs(scores["mask_ap"] - expected) < 1e-12
                and not failed_gt
                and not failed_pred
            )
        model, _ = build_model(
            {"torch": torch, "torchvision": torchvision}, "none", False
        )
        model.eval()
        with torch.inference_mode():
            out = model([image])[0]
        assert {"boxes", "labels", "scores", "masks"} <= out.keys()
    print(
        "Passed dataset I/O, corrected-reference loader, perfect/empty AP, and CPU Mask R-CNN inference."
    )


if __name__ == "__main__":
    main()
