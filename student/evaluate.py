"""Replay corrected-reference Mask AP on saved COCO validation predictions."""

import argparse
import json
from pathlib import Path
from validation import safe_mask_ap


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gt",
        type=Path,
        required=True,
        help="COCO validation ground truth with RLE masks",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="COCO segmentation prediction list",
    )
    args = parser.parse_args()
    gt = json.loads(args.gt.read_text(encoding="utf-8-sig"))
    preds = json.loads(args.predictions.read_text(encoding="utf-8-sig"))
    identifiers = [int(im["id"]) for im in gt["images"]]
    known_ids = set(identifiers)
    if any(int(row["image_id"]) not in known_ids for row in preds):
        raise ValueError("Prediction with unknown image ID")
    scores, _, failed, _ = safe_mask_ap(
        "replay", "saved", "val", gt, preds, identifiers, 100
    )
    print(
        json.dumps(
            {
                "validation_images": len(identifiers),
                "instances": len(gt["annotations"]),
                "undecodable_predictions": len(failed),
                **scores,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
