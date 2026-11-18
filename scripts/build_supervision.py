"""Assemble instance-indexed supervision for a caller-supplied patch inventory."""

import argparse
import json
import re
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np

from crownroute.association import MIN_MASK_AREA
from crownroute.pipeline import construct_record, generate_candidates


def read_image(path, flags):
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), flags)
    if image is None:
        raise ValueError(f"Could not decode {path}")
    return image


def read_candidates(base, paths, shape):
    candidates = []
    for relative in paths:
        pixels = read_image(base / relative, cv2.IMREAD_GRAYSCALE)
        if pixels.shape != shape or not np.isin(pixels, [0, 1, 255]).all():
            raise ValueError(f"Expected a same-size binary mask for {relative}")
        mask = (pixels > 0).astype(np.uint8)
        if int(mask.sum()) >= MIN_MASK_AREA:
            candidates.append(mask)
    return candidates


def checked_box(box, shape):
    height, width = shape
    if len(box) != 4 or any(type(value) is not int for value in box):
        raise ValueError(f"Expected four integer pixel coordinates: {box}")
    x1, y1, x2, y2 = box
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError(f"Box outside the image or empty: {box}")
    return box


def assemble(manifest, output, weights=None, device="0"):
    patches = json.loads(manifest.read_text(encoding="utf-8"))["patches"]
    if output.exists():
        raise FileExistsError(output)
    seen = set()
    source_splits = {}
    source_types = {patch.get("candidate_masks") is None for patch in patches}
    if not patches or len(source_types) != 1:
        raise ValueError("Supply patches all using FastSAM, or all using saved candidate masks.")
    use_fastsam = source_types.pop()
    if use_fastsam != (weights is not None):
        raise ValueError("Supply --weights only for FastSAM inference.")
    model = None
    if use_fastsam:
        from ultralytics import FastSAM

        if not weights.is_file():
            raise FileNotFoundError(weights)
        model = FastSAM(str(weights))

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as temporary:
        root = Path(temporary) / "dataset"
        for name in ["common_images", "masks", "meta", "split"]:
            (root / name).mkdir(parents=True)
        records = []
        outcomes = []
        for patch in patches:
            patch_id = patch["id"]
            split = patch["split"]
            source_id = str(patch["source_id"])
            if not re.fullmatch(r"[A-Za-z0-9_-]+", patch_id) or patch_id.casefold() in seen:
                raise ValueError(f"Expected unique, filename-safe patch id: {patch_id}")
            seen.add(patch_id.casefold())
            if split not in {"train", "val", "test"} or source_splits.setdefault(source_id, split) != split:
                raise ValueError(f"Invalid or inconsistent split for source {source_id}")
            original = manifest.parent / patch["image"]
            image = read_image(original, cv2.IMREAD_COLOR)
            candidates = (
                generate_candidates(model, original, device)
                if use_fastsam
                else read_candidates(manifest.parent, patch["candidate_masks"], image.shape[:2])
            )
            image_name = f"common_images/{patch_id}{original.suffix}"
            for number, box in enumerate(patch["boxes"], 1):
                checked_box(box, image.shape[:2])
                record_id = f"{patch_id}_obj{number}"
                result = construct_record(image, box, candidates)
                outcomes.append({"id": record_id, "outcome": result.route, "reason": result.reason,
                                 "features": result.association, "admission_score": result.admission_score})
                if result.mask is None:
                    continue
                if not (root / image_name).exists():
                    shutil.copyfile(original, root / image_name)
                mask_name = f"masks/{record_id}.png"
                ok, encoded = cv2.imencode(".png", result.mask * 255)
                if not ok:
                    raise OSError(f"Could not encode mask {record_id}")
                encoded.tofile(root / mask_name)
                class_type = result.route[-1]
                records.append({
                    "id": record_id, "patch_image": image_name, "image_path": image_name,
                    "mask_path": mask_name, "gt_index": number - 1, "gt_box": box,
                    "class_type": class_type, "source_disjoint_v3_split": split,
                    "source_disjoint_v3_source_id": source_id, **result.association
                })
        # Historical balanced selection concatenated the A list with accepted B.
        records.sort(key=lambda row: row["class_type"])
        for name, rows in [("dataset_records", records), ("route_outcomes", outcomes)]:
            (root / "meta" / f"{name}.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        for split in ["train", "val", "test"]:
            ids = [row["id"] for row in records if row["source_disjoint_v3_split"] == split]
            (root / "split" / f"{split}.txt").write_text("".join(key + "\n" for key in ids), encoding="utf-8")
        root.rename(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--weights", type=Path, help="Locally supplied FastSAM-s weights")
    parser.add_argument("--device", default="0", help="Used only with --weights")
    args = parser.parse_args()
    assemble(args.manifest, args.output, args.weights, args.device)


if __name__ == "__main__":
    main()
