"""The frozen recovery operator used for the training-side boundary analysis."""

from collections import Counter

import numpy as np

from . import normalization, reconstruction


def run_frozen_recovery(image: np.ndarray, bbox: list[int]) -> tuple[np.ndarray, dict]:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros(image.shape[:2], dtype=np.uint8), {
            "status": "invalid_bbox_crop",
            "step10_5_status": "not_run",
            "step10_7_status": "not_run",
        }
    _, _, _, _, best = reconstruction.build_initial_mask(image, bbox)
    if best is None:
        zero = np.zeros(image.shape[:2], dtype=np.uint8)
        after7, stats7 = normalization.repair_mask(image, zero, bbox)
        return after7, {
            "status": "no_initial_component",
            "step10_5_status": "no_mask",
            "step10_5_holes": 0,
            "step10_5_actions": {},
            "step10_7_status": "applied_to_empty",
            "step10_7": stats7,
        }
    after5_crop, _, _, _, _, holes5 = reconstruction.contrast_aware_hole_recovery(crop, best)
    after5 = reconstruction.paste_crop_mask_to_full(image.shape, after5_crop, bbox)
    after7, stats7 = normalization.repair_mask(image, after5, bbox)
    nonempty = int((after7 > 0).sum()) > 0
    return after7, {
        "status": "ok" if nonempty else "empty_after_recovery",
        "step10_5_status": "ok",
        "step10_5_holes": len(holes5),
        "step10_5_actions": dict(Counter(row["action"] for row in holes5)),
        "step10_7_status": "ok",
        "step10_7": stats7,
    }
