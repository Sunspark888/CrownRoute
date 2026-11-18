"""Apply the historical recovery operator to training-side original intermediate C."""

import argparse
import csv
import json
from pathlib import Path
import shutil
import tempfile

import cv2

from crownroute.recovery import run_frozen_recovery
from scripts.build_supervision import read_image


def patch_name(record):
    return Path(record["patch_image"].replace("\\", "/")).name


def recovery_record(row, image_name, iid, image_rel, mask_rel, area, diagnostics):
    box = [int(x) for x in row["gt_box"]]
    x1, y1, x2, y2 = box
    box_area = max(0, x2 - x1) * max(0, y2 - y1)
    return {
        "id": iid, "image_path": image_rel, "mask_path": mask_rel, "patch_image": image_name,
        "gt_index": int(row["gt_index"]), "gt_box": box, "class_type": "C", "original_class_type": "C",
        "training_mask_source": "Recovered-C", "source_branch": "C_same_frozen_B_recovery_step10_5_then_step10_7",
        **{k: row.get(k) for k in ["final_score", "bbox_iou", "inside_ratio", "green_ratio"]},
        "source_disjoint_v3_source_id": Path(image_name).stem.split("_patch_", 1)[0],
        "source_disjoint_v3_split": "train", "bbox_area": float(box_area), "mask_area": area,
        "area_ratio": float(area / box_area) if box_area else None,
        "weak_supervision_method": "post_hoc_B0_plus_Recovered-C",
        "recovery_operator": "frozen_step10_5_then_frozen_step10_7",
        "recovery_parameters_changed": False, "c_specific_tuning": False, "manual_gt_read": False,
        "validation_probe_iou_selection": False, "test_used_for_selection": False,
        "server_package_relative_paths": True, "recovery_diagnostics": diagnostics,
    }


def recover(classified_path, source_split, images, output):
    with source_split.open(encoding="utf-8-sig", newline="") as stream:
        assignments = {}
        for row in csv.DictReader(stream):
            sid, split = row["source_id"], row["split"]
            if sid in assignments and assignments[sid] != split:
                raise ValueError(f"Conflicting splits for {sid}")
            assignments[sid] = split
    selected = {}
    for row in json.loads(classified_path.read_text(encoding="utf-8-sig")):
        if row["class_type"] != "C":
            continue
        name = patch_name(row)
        stem = Path(name).stem
        split = assignments[stem.split("_patch_", 1)[0]]
        if split == "train":
            iid = f"{stem}_obj{int(row['gt_index']) + 1}"
            if iid in selected:
                raise ValueError(f"Duplicate recovery input {iid}")
            selected[iid] = row
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as temp:
        staged = Path(temp) / "delta"
        for directory in ["images", "masks"]:
            (staged / directory).mkdir(parents=True)
        records, outcomes = [], []
        for iid, row in sorted(selected.items()):
            name = patch_name(row)
            image_path = images / name
            try:
                image = read_image(image_path, cv2.IMREAD_COLOR)
            except (OSError, ValueError, cv2.error):
                outcomes.append({"id": iid, "status": "image_read_failed"})
                continue
            mask, diagnostics = run_frozen_recovery(image, [int(x) for x in row["gt_box"]])
            area = int((mask > 0).sum())
            outcomes.append({"id": iid, "status": diagnostics["status"], "mask_area": area})
            if diagnostics["status"] != "ok" or area == 0:
                continue
            image_rel, mask_rel = f"images/{name}", f"masks/{iid}.png"
            if not (staged / image_rel).exists():
                shutil.copyfile(image_path, staged / image_rel)
            ok, encoded = cv2.imencode(".png", mask * 255)
            if not ok:
                raise OSError(f"Could not encode {iid}")
            encoded.tofile(staged / mask_rel)
            records.append(recovery_record(row, name, iid, image_rel, mask_rel, area, diagnostics))
        (staged / "records.json").write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
        (staged / "outcomes.json").write_text(json.dumps(outcomes), encoding="utf-8")
        staged.rename(output)
    return {"training_intermediate_C": len(selected), "recovered": len(records), "excluded": len(selected)-len(records)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classified", type=Path, required=True, help="Original intermediate routing inventory")
    parser.add_argument("--source-split", type=Path, required=True, help="CSV with source_id and split")
    parser.add_argument("--images", type=Path, required=True, help="Directory holding named RGB patches")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(recover(args.classified, args.source_split, args.images, args.output)))


if __name__ == "__main__":
    main()
