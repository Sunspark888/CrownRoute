"""CrownRoute operations extracted from step10_7_ab_holefill_preview_300.py."""

from collections import Counter

import cv2
import numpy as np


PARAMS = {
    "MIN_HOLE_AREA": 1,
    "TINY_HOLE_AREA": 64,
    "SMALL_HOLE_AREA": 1500,
    "MAX_HOLE_AREA": 10000,
    "EDGE_MARGIN": 6,
    "MASK_BBOX_EDGE_MARGIN": 6,
    "RING_DILATE": 9,
    "MIN_RING_PIXELS": 32,
    "GREEN_HSV_LOWER": [20, 20, 20],
    "GREEN_HSV_UPPER": [100, 255, 255],
    "RING_GREEN_RATIO_MIN": 0.45,
    "HOLE_GREEN_RATIO_FILL": 0.20,
    "TEXTURE_STD_FILL": 10.0,
    "COLOR_DISTANCE_KEEP": 55.0,
    "DARK_DELTA_FILL_MIN": 12.0,
    "LOW_SATURATION_MAX": 35,
    "BRIGHT_VALUE_MIN": 160,
    "RED_ORANGE_HUE_MAX": 22,
    "BLUE_HUE_MIN": 90,
    "BLUE_HUE_MAX": 135,
    "VETO_GREEN_RATIO_MAX": 0.12,
}
FILL_ACTIONS = {
    "fill_tiny_no_ring",
    "fill_tiny",
    "fill_small_or_medium",
    "fill_dark_crown_shadow",
    "fill_texture_or_green",
}


def find_holes(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask.astype(np.uint8), 1, mode="constant", constant_values=0)
    flood = (padded * 255).astype(np.uint8)
    ff_mask = np.zeros((flood.shape[0] + 2, flood.shape[1] + 2), np.uint8)
    cv2.floodFill(flood, ff_mask, (0, 0), 255)
    outside = (flood > 0).astype(np.uint8)
    holes_padded = ((outside == 0) & (padded == 0)).astype(np.uint8)
    return holes_padded[1:-1, 1:-1]


def connected_components(mask: np.ndarray) -> list[dict]:
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    comps = []
    for idx in range(1, num_labels):
        x, y, w, h, area = [int(v) for v in stats[idx]]
        comps.append(
            {
                "label": idx,
                "mask": (labels == idx).astype(np.uint8),
                "area": area,
                "bbox": [x, y, x + w - 1, y + h - 1],
                "centroid": [float(centroids[idx][0]), float(centroids[idx][1])],
            }
        )
    return comps


def mask_bbox(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def bbox_near_edge(box: list[int], outer: list[int], margin: int) -> bool:
    return (
        box[0] <= outer[0] + margin
        or box[1] <= outer[1] + margin
        or box[2] >= outer[2] - margin
        or box[3] >= outer[3] - margin
    )


def local_std(gray: np.ndarray, selector: np.ndarray) -> float:
    values = gray[selector > 0]
    if values.size == 0:
        return 0.0
    return float(values.std())


def green_ratio(hsv: np.ndarray, selector: np.ndarray) -> float:
    values = hsv[selector > 0]
    if values.size == 0:
        return 0.0
    lower = np.array(PARAMS["GREEN_HSV_LOWER"], dtype=np.uint8)
    upper = np.array(PARAMS["GREEN_HSV_UPPER"], dtype=np.uint8)
    in_range = np.all((values >= lower) & (values <= upper), axis=1)
    return float(in_range.mean())


def mean_color(image: np.ndarray, selector: np.ndarray) -> list[float]:
    values = image[selector > 0]
    if values.size == 0:
        return [0.0, 0.0, 0.0]
    return [float(v) for v in values.mean(axis=0)]


def color_veto(hsv: np.ndarray, bgr: np.ndarray, hole_mask: np.ndarray, hole_green_ratio: float) -> tuple[bool, str]:
    pixels_hsv = hsv[hole_mask > 0]
    pixels_bgr = bgr[hole_mask > 0]
    if pixels_hsv.size == 0 or pixels_bgr.size == 0:
        return False, ""

    h_mean = float(pixels_hsv[:, 0].mean())
    s_mean = float(pixels_hsv[:, 1].mean())
    v_mean = float(pixels_hsv[:, 2].mean())
    b_mean, g_mean, r_mean = [float(v) for v in pixels_bgr.mean(axis=0)]
    low_green = hole_green_ratio <= PARAMS["VETO_GREEN_RATIO_MAX"]

    red_or_orange = (h_mean <= PARAMS["RED_ORANGE_HUE_MAX"] or h_mean >= 170) and r_mean > g_mean + 12
    bright_roof = s_mean <= PARAMS["LOW_SATURATION_MAX"] and v_mean >= PARAMS["BRIGHT_VALUE_MIN"]
    blue_or_water = PARAMS["BLUE_HUE_MIN"] <= h_mean <= PARAMS["BLUE_HUE_MAX"]
    gray_surface = abs(r_mean - g_mean) <= 12 and abs(g_mean - b_mean) <= 12 and s_mean <= 55 and v_mean >= 95

    if low_green and red_or_orange:
        return True, "red_or_orange_low_green"
    if low_green and bright_roof:
        return True, "bright_low_saturation_low_green"
    if low_green and blue_or_water:
        return True, "blue_or_water_low_green"
    if low_green and gray_surface:
        return True, "gray_surface_low_green"
    return False, ""


def classify_hole(
    image: np.ndarray,
    hsv: np.ndarray,
    gray: np.ndarray,
    mask: np.ndarray,
    mask_box: list[int] | None,
    gt_box: list[int] | None,
    comp: dict,
) -> tuple[str, dict]:
    area = int(comp["area"])
    hole_mask = comp["mask"]
    hole_box = comp["bbox"]
    h, w = mask.shape
    diag = {"area": area, "bbox": hole_box, "centroid": comp["centroid"]}

    if area < PARAMS["MIN_HOLE_AREA"]:
        return "keep_too_small", diag
    if bbox_near_edge(hole_box, [0, 0, w - 1, h - 1], PARAMS["EDGE_MARGIN"]):
        return "keep_edge_patch", diag
    if gt_box is not None and bbox_near_edge(hole_box, [int(v) for v in gt_box], PARAMS["EDGE_MARGIN"]):
        return "keep_edge_gt_box", diag
    if mask_box is not None and bbox_near_edge(hole_box, mask_box, PARAMS["MASK_BBOX_EDGE_MARGIN"]):
        return "keep_edge_mask_bbox", diag
    if area > PARAMS["MAX_HOLE_AREA"]:
        return "keep_large", diag

    kernel = np.ones((PARAMS["RING_DILATE"], PARAMS["RING_DILATE"]), np.uint8)
    dilated = cv2.dilate(hole_mask.astype(np.uint8), kernel, iterations=1)
    ring = (dilated - hole_mask).clip(0, 1).astype(np.uint8)
    ring = (ring & mask.astype(np.uint8)).astype(np.uint8)
    ring_pixels = int(ring.sum())
    diag["ring_pixels"] = ring_pixels
    if ring_pixels < PARAMS["MIN_RING_PIXELS"]:
        if area <= PARAMS["TINY_HOLE_AREA"]:
            return "fill_tiny_no_ring", diag
        return "keep_low_ring_support", diag

    hole_green = green_ratio(hsv, hole_mask)
    ring_green = green_ratio(hsv, ring)
    hole_bgr_mean = mean_color(image, hole_mask)
    ring_bgr_mean = mean_color(image, ring)
    color_dist = float(np.linalg.norm(np.array(hole_bgr_mean) - np.array(ring_bgr_mean)))
    hole_gray = float(gray[hole_mask > 0].mean())
    ring_gray = float(gray[ring > 0].mean())
    dark_delta = ring_gray - hole_gray
    hole_texture_std = local_std(gray, hole_mask)

    diag.update(
        {
            "hole_green_ratio": hole_green,
            "ring_green_ratio": ring_green,
            "hole_bgr_mean": hole_bgr_mean,
            "ring_bgr_mean": ring_bgr_mean,
            "color_distance": color_dist,
            "hole_gray_mean": hole_gray,
            "ring_gray_mean": ring_gray,
            "dark_delta": dark_delta,
            "hole_texture_std": hole_texture_std,
        }
    )

    has_veto, veto_reason = color_veto(hsv, image, hole_mask, hole_green)
    diag["color_veto"] = veto_reason
    if has_veto:
        return "keep_color_veto", diag
    if color_dist >= PARAMS["COLOR_DISTANCE_KEEP"] and hole_green < PARAMS["HOLE_GREEN_RATIO_FILL"]:
        return "keep_color_mismatch", diag
    if area <= PARAMS["TINY_HOLE_AREA"]:
        return "fill_tiny", diag
    if area <= PARAMS["SMALL_HOLE_AREA"]:
        return "fill_small_or_medium", diag

    ring_is_crown = ring_green >= PARAMS["RING_GREEN_RATIO_MIN"]
    hole_is_crown_like = hole_green >= PARAMS["HOLE_GREEN_RATIO_FILL"] or hole_texture_std >= PARAMS["TEXTURE_STD_FILL"]
    hole_is_dark_shadow = dark_delta >= PARAMS["DARK_DELTA_FILL_MIN"]
    if ring_is_crown and hole_is_dark_shadow:
        return "fill_dark_crown_shadow", diag
    if ring_is_crown and hole_is_crown_like:
        return "fill_texture_or_green", diag
    return "keep_uncertain_medium_large", diag


def repair_mask(image: np.ndarray, mask: np.ndarray, gt_box: list[int] | None) -> tuple[np.ndarray, dict]:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    holes = find_holes(mask)
    comps = connected_components(holes)
    box = mask_bbox(mask)
    repaired = mask.copy().astype(np.uint8)
    action_counts = Counter()
    hole_records = []

    for comp in comps:
        action, diag = classify_hole(image, hsv, gray, mask, box, gt_box, comp)
        action_counts[action] += 1
        if action in FILL_ACTIONS:
            repaired[comp["mask"] > 0] = 1
        diag["action"] = action
        hole_records.append(diag)

    changed = ((repaired > 0) & (mask == 0)).astype(np.uint8)
    filled_holes = sum(action_counts[a] for a in FILL_ACTIONS)
    total_holes = len(hole_records)
    return repaired, {
        "total_holes": total_holes,
        "filled_holes": int(filled_holes),
        "kept_holes": int(total_holes - filled_holes),
        "filled_pixels": int(changed.sum()),
        "before_area": int(mask.sum()),
        "after_area": int(repaired.sum()),
        "action_counts": dict(action_counts),
        "hole_records": hole_records,
    }
