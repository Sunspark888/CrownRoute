"""Connect the original scientific functions using the frozen standard operating point."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import association, reconstruction
from .eligibility import classify_instance
from .normalization import repair_mask


CONFIG = json.loads(
    (Path(__file__).resolve().parents[1] / "configs" / "c1.json").read_text(encoding="utf-8")
)


@dataclass
class Record:
    route: str
    reason: str
    mask: np.ndarray | None = None
    association: dict | None = None
    admission_score: float | None = None


def generate_candidates(model, image_path, device=0):
    """Use one image-level inference per patch, without any box prompts."""
    results = model(str(image_path), device=device, **CONFIG["fastsam"])
    return association.get_candidate_masks_from_fastsam(results[0])


def admission_score(features):
    return (
        0.40 * features["final_score"]
        + 0.30 * features["bbox_iou"]
        + 0.20 * features["inside_ratio"]
        + 0.10 * features["green_ratio"]
    )


def construct_record(image_bgr, box, candidates):
    """Candidates are same-size binary masks already filtered by the generator."""
    mask, _, features = association.choose_best_mask_with_multiscore(box, candidates, image_bgr)
    if mask is None:
        return Record("Final-C", "no_candidate")
    # Only fields persisted in the original association summary participate.
    route = classify_instance({"status": "ok", **features})
    if route == "C":
        return Record("Final-C", "ineligible", association=features)
    if route == "A":
        refined = association.refine_mask_first_version(image_bgr, mask, box)["refined_mask"]
        normalized, _ = repair_mask(image_bgr, refined, box)
        return Record("Final-A", "refined", normalized, features)

    *_, initial = reconstruction.build_initial_mask(image_bgr, box)
    if initial is None:
        # The original reconstruction script wrote a no_mask entry and emitted
        # no training file. Keep this distinct from a measured gate rejection.
        return Record("unconstructed", "no_reconstructed_mask", association=features)
    x1, y1, x2, y2 = box
    crop = image_bgr[y1:y2, x1:x2]
    repaired, *_ = reconstruction.contrast_aware_hole_recovery(crop, initial)
    rebuilt = reconstruction.paste_crop_mask_to_full(image_bgr.shape, repaired, box)
    normalized, _ = repair_mask(image_bgr, rebuilt, box)
    score = admission_score(features)
    if score < CONFIG["balanced_threshold"]:
        return Record("Final-C", "below_acceptance_threshold", association=features, admission_score=score)
    return Record("Final-B", "reconstructed", normalized, features, score)
