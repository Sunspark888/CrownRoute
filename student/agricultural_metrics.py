"""Orchard inventory endpoints at a fixed confidence threshold, using saved masks only."""

import argparse
import contextlib
import csv
import io
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from pycocotools import mask as mask_api
from pycocotools.coco import COCO


def per_image(coco, predictions, threshold=0.50):
    """Assign by maximum total mask IoU, then retain pairs with mask IoU >= 0.50."""
    from scipy.optimize import linear_sum_assignment

    if not 0 <= threshold <= 1:
        raise ValueError("Confidence threshold must be between 0 and 1")
    if len(coco.cats) != 1 or not coco.imgs:
        raise ValueError(
            "Inventory evaluation requires one crown category and at least one validation image"
        )
    detections = defaultdict(list)
    for row in predictions:
        if (
            row["image_id"] not in coco.imgs
            or row["category_id"] not in coco.cats
            or not math.isfinite(row["score"])
            or not 0 <= row["score"] <= 1
        ):
            raise ValueError(
                "Prediction has an unknown image/category or invalid confidence"
            )
        if row["score"] >= threshold:
            detections[row["image_id"]].append(row)
    rows = []
    for image_id, image in sorted(coco.imgs.items()):
        gt_masks = [coco.annToRLE(ann) for ann in coco.imgToAnns[image_id]]
        pred_masks = [det["segmentation"] for det in detections[image_id]]
        matches = []
        if pred_masks and gt_masks:
            ious = mask_api.iou(pred_masks, gt_masks, [0] * len(gt_masks))
            pred_idx, gt_idx = linear_sum_assignment(-ious)
            matches = [(i, j) for i, j in zip(pred_idx, gt_idx) if ious[i, j] >= 0.50]
        area_errors = []
        for i, j in matches:
            pred_area, gt_area = (
                float(mask_api.area(pred_masks[i])),
                float(mask_api.area(gt_masks[j])),
            )
            area_errors.append(abs(pred_area - gt_area) / max(1, gt_area))
        n_gt, n_pred, tp = len(gt_masks), len(pred_masks), len(matches)
        delta = n_pred - n_gt
        rows.append(
            {
                "threshold": threshold,
                "image_id": image_id,
                "source": str(
                    image.get(
                        "source_stem", image.get("patch_stem", "").split("_patch_")[0]
                    )
                ),
                "gt_count": n_gt,
                "pred_count": n_pred,
                "tp": tp,
                "fp": n_pred - tp,
                "fn": n_gt - tp,
                "count_abs_error": abs(delta),
                "count_signed_error": delta,
                "matched_area_relative_error_sum": sum(area_errors),
                "matched_area_n": tp,
            }
        )
    return rows


def summarize(rows):
    if not rows:
        raise ValueError("No validation images")
    tp, fp, fn = (sum(row[key] for row in rows) for key in ("tp", "fp", "fn"))
    matched = sum(row["matched_area_n"] for row in rows)
    return {
        "score_threshold": rows[0]["threshold"],
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": tp / max(1, tp + fp),
        "recall": tp / max(1, tp + fn),
        "f1": 2 * tp / max(1, 2 * tp + fp + fn),
        "count_mae": statistics.mean(row["count_abs_error"] for row in rows),
        "count_bias": statistics.mean(row["count_signed_error"] for row in rows),
        "matched_area_n": matched,
        "matched_area_relative_error": sum(
            row["matched_area_relative_error_sum"] for row in rows
        )
        / max(1, matched),
    }


def inventory(coco, predictions, threshold=0.50):
    """Return the per-seed inventory summary (the original single-run interface)."""
    return summarize(per_image(coco, predictions, threshold))


def write_rows(path, rows):
    with path.open("x", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_runs(path):
    runs = defaultdict(dict)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            method, seed = row["method"].strip(), int(row["seed"])
            if not method or not row["predictions"].strip() or seed in runs[method]:
                raise ValueError("Each method/seed must have one prediction file")
            runs[method][seed] = path.parent / row["predictions"]
    if not runs or any(set(by_seed) != {42, 43, 44} for by_seed in runs.values()):
        raise ValueError("Each method requires seeds 42, 43 and 44")
    return runs


def batch_inventory(coco, runs, threshold):
    all_images, all_seeds, means = [], [], []
    for method, files in sorted(runs.items()):
        seeds = []
        for seed, path in sorted(files.items()):
            predictions = json.loads(path.read_text(encoding="utf-8-sig"))
            images = per_image(coco, predictions, threshold)
            all_images.extend({"method": method, "seed": seed, **row} for row in images)
            seeds.append(
                {
                    "method": method,
                    "seed": seed,
                    "images": len(images),
                    **summarize(images),
                }
            )
        all_seeds.extend(seeds)
        summary = {"method": method, "seeds": len(seeds), "score_threshold": threshold}
        for metric in (
            "count_mae",
            "recall",
            "precision",
            "f1",
            "count_bias",
            "matched_area_relative_error",
        ):
            values = [row[metric] for row in seeds]
            summary[metric + "_mean"] = statistics.mean(values)
            summary[metric + "_sample_sd"] = statistics.stdev(values)
        means.append(summary)
    return all_images, all_seeds, means


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gt", type=Path, required=True, help="COCO validation reference"
    )
    parser.add_argument(
        "--runs",
        type=Path,
        required=True,
        help="CSV with method,seed,predictions; paths relative to the CSV",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New directory for three CSV tables",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.50,
        help="Confidence threshold (Table 6: 0.50)",
    )
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("--output-dir must identify a new directory")
    runs = read_runs(args.runs)
    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO(str(args.gt))
    images, seeds, means = batch_inventory(coco, runs, args.threshold)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name, rows in (
        ("per_image.csv", images),
        ("per_seed.csv", seeds),
        ("three_seed_summary.csv", means),
    ):
        write_rows(args.output_dir / name, rows)
    print(
        f"Wrote {len(images)} image rows, {len(seeds)} seed rows and {len(means)} method summaries."
    )


if __name__ == "__main__":
    main()
