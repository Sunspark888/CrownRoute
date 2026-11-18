"""Verify duplicate predictions and a tie case that distinguishes the two published evaluation paths."""

import contextlib
import io
import math

import numpy as np
from pycocotools.coco import COCO

from evaluate_table4 import evaluate
from agricultural_metrics import inventory, per_image, summarize
from threshold_metrics import metrics
from validation import encode_rle, safe_mask_ap


def compare(gt, masks):
    predictions = [
        {
            "image_id": 1,
            "category_id": 1,
            "score": 1 - index * 0.1,
            "segmentation": encode_rle(mask),
        }
        for index, mask in enumerate(masks)
    ]
    coco = COCO()
    coco.dataset = gt
    with contextlib.redirect_stdout(io.StringIO()):
        coco.createIndex()
        final5b, failed_gt, failed_pred, _ = safe_mask_ap(
            "demo", "saved", "val", gt, predictions, [1], 100
        )
        official = evaluate(coco, predictions)
    assert not failed_gt and not failed_pred
    return official["segm_AP50"], final5b["mask_ap50"] * 100


def check_inventory_edges():
    def band(start, end):
        mask = np.zeros((4, 16), dtype=np.uint8)
        mask[:, start:end] = 1
        return encode_rle(mask)

    gt = {
        "images": [{"id": i, "width": 16, "height": 4} for i in (1, 2, 3, 4)],
        "categories": [{"id": 1, "name": "crown"}],
        "annotations": [
            {
                "id": i,
                "image_id": image,
                "category_id": 1,
                "iscrowd": 0,
                "area": 4 * (b - a),
                "bbox": [a, 0, b - a, 4],
                "segmentation": band(a, b),
            }
            for i, (image, a, b) in enumerate(((1, 2, 10), (1, 4, 12), (2, 2, 10)), 1)
        ],
    }
    # The first image's IoUs are [[.8,.5],[.5,.25]]. The maximum-sum assignment
    # is its diagonal; only one match survives. Filtering before assignment could retain two.
    preds = [
        {
            "image_id": image,
            "category_id": 1,
            "score": score,
            "segmentation": band(a, b),
        }
        for image, a, b, score in (
            (1, 0, 10, 0.99),
            (1, 0, 7, 0.80),
            (3, 2, 10, 0.50),
            (4, 2, 10, 0.4999),
        )
    ]
    coco = COCO()
    coco.dataset = gt
    with contextlib.redirect_stdout(io.StringIO()):
        coco.createIndex()
    rows = per_image(coco, preds, 0.50)
    assert [r["image_id"] for r in rows] == [1, 2, 3, 4]
    assert [(r["tp"], r["fp"], r["fn"], r["count_signed_error"]) for r in rows] == [
        (1, 1, 1, 0),
        (0, 0, 1, -1),
        (0, 1, 0, 1),
        (0, 0, 0, 0),
    ]
    result = summarize(rows)
    assert result["count_mae"] == 0.5 and result["count_bias"] == 0.0
    assert (
        result["matched_area_n"] == 1 and result["matched_area_relative_error"] == 0.25
    )
    print(
        "Passed post-assignment IoU cutoff, exact score cutoff and all empty-image cases."
    )


def main():
    check_inventory_edges()
    left, right = np.zeros((12, 12), np.uint8), np.zeros((12, 12), np.uint8)
    left[2:8, 2:5] = 1
    right[2:8, 5:8] = 1
    gt = {
        "info": {},
        "images": [{"id": 1, "file_name": "synthetic", "width": 12, "height": 12}],
        "categories": [{"id": 1, "name": "crown"}],
        "annotations": [
            {
                "id": i + 1,
                "image_id": 1,
                "category_id": 1,
                "area": 18,
                "bbox": [2 + 3 * i, 2, 3, 6],
                "iscrowd": 0,
                "segmentation": encode_rle(mask),
            }
            for i, mask in enumerate((left, right))
        ],
    }
    # Two identical high-score detections on the left crown, then the right crown.
    a, b = compare(gt, [left, left, right])
    expected = 100 * (51 + 50 * 2 / 3) / 101
    assert math.isclose(a, expected) and math.isclose(b, expected)
    print(f"Duplicate AP50 (points): official={a:.6f}, final5b={b:.6f}")
    # The merged prediction has IoU .50 with both GT. Annotation order is left then right.
    a, b = compare(gt, [left | right, left])
    assert math.isclose(a, 100) and math.isclose(b, 100 * 51 / 101)
    print(f"Equal-IoU tie AP50 (points): official={a:.6f}, final5b={b:.6f}")
    preds = [
        {
            "image_id": 1,
            "category_id": 1,
            "score": score,
            "segmentation": encode_rle(mask),
        }
        for score, mask in ((0.99, left | right), (0.95, left))
    ]
    c = COCO()
    c.dataset = gt
    with contextlib.redirect_stdout(io.StringIO()):
        c.createIndex()
    agricultural = inventory(c, preds, 0.50)
    training_diagnostic, _ = metrics(gt, preds, 0.50, 0.50)
    assert (agricultural["tp"], training_diagnostic["tp"]) == (2, 1)
    assert agricultural["matched_area_relative_error"] == 0.50
    print(
        "Two-target ambiguity (2 GT, 2 detections): inventory Hungarian TP=2, training diagnostic greedy TP=1."
    )
    a, b = compare(gt, [])
    assert a == b == 0


if __name__ == "__main__":
    main()
