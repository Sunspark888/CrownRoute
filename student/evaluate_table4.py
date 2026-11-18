"""Reproduce Table 4 using official COCOeval on saved validation detections."""

import argparse
import contextlib
import copy
import io
import json
import math
from pathlib import Path


from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from agricultural_metrics import (
    inventory as inventory,
    per_image,
    summarize,
    write_rows,
)


def evaluate(coco, predictions, kind="segm"):
    if kind not in {"segm", "bbox"}:
        raise ValueError("Use segm or bbox")
    if any(
        row["image_id"] not in coco.imgs
        or row["category_id"] not in coco.cats
        or not math.isfinite(row["score"])
        or not 0 <= row["score"] <= 1
        for row in predictions
    ):
        raise ValueError(
            "Prediction has an unknown image/category or invalid confidence"
        )
    data = copy.deepcopy(predictions)
    if kind == "segm":
        # COCO.loadRes uses box area if a box is present; segmentation must use mask area.
        for row in data:
            row.pop("bbox", None)
    with contextlib.redirect_stdout(io.StringIO()):
        if data:
            detections = coco.loadRes(data)
        else:
            # loadRes([]) assumes at least one row. Keep the same images and categories for zero detections.
            detections = COCO()
            detections.dataset = {
                "images": list(coco.imgs.values()),
                "categories": list(coco.cats.values()),
                "annotations": [],
            }
            detections.createIndex()
        evaluator = COCOeval(coco, detections, kind)
        evaluator.params.imgIds = sorted(coco.imgs)
        evaluator.params.catIds = sorted(coco.cats)
        evaluator.params.maxDets = [1, 10, 100]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    return {
        f"{kind}_{name}": float(value * 100)
        for name, value in zip(("AP", "AP50", "AP75"), evaluator.stats[:3])
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gt", type=Path, required=True, help="COCO validation annotation JSON"
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="Saved COCO segmentation detections",
    )
    parser.add_argument(
        "--boxes", type=Path, help="Optional saved box detections, for box AP"
    )
    parser.add_argument(
        "--inventory", action="store_true", help="Add Table 6 inventory endpoints"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.50,
        help="Inventory confidence threshold (Table 6: 0.50)",
    )
    parser.add_argument(
        "--per-image-csv",
        type=Path,
        help="New CSV of per-image inventory values (implies --inventory)",
    )
    parser.add_argument("--method", help="Method label for per-image output")
    parser.add_argument("--seed", type=int, help="Seed label for per-image output")
    args = parser.parse_args()
    if args.per_image_csv and (not args.method or args.seed is None):
        parser.error("--per-image-csv requires --method and --seed")
    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO(str(args.gt))
    predictions = json.loads(args.predictions.read_text(encoding="utf-8-sig"))
    scores = evaluate(coco, predictions)
    if args.inventory or args.per_image_csv:
        rows = per_image(coco, predictions, args.threshold)
        scores["inventory"] = summarize(rows)
        if args.per_image_csv:
            write_rows(
                args.per_image_csv,
                [{"method": args.method, "seed": args.seed, **row} for row in rows],
            )
    if args.boxes:
        scores.update(
            evaluate(
                coco, json.loads(args.boxes.read_text(encoding="utf-8-sig")), "bbox"
            )
        )
    print(
        json.dumps(
            {
                "images": len(coco.imgs),
                "instances": len(coco.anns),
                "scale": "AP points (0–100)",
                **scores,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
