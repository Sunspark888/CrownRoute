"""CrownRoute operations extracted from step10_5_hole_recovery_full.py."""

import cv2
import numpy as np


WINDOW_SIZE = 9
STD_THRESH = 10.0
LOWER_GREEN = np.array([20, 20, 20], dtype=np.uint8)
UPPER_GREEN = np.array([100, 255, 255], dtype=np.uint8)
MIN_COMPONENT_AREA = 80
MORPH_KERNEL = 5
MAX_HOLE_AREA = 1500
DELTA_THRESH = 22.0
RING_DILATE = 7
EDGE_TOUCH_MARGIN = 2


def paste_crop_mask_to_full(full_shape, crop_mask, box):
    x1, y1, x2, y2 = map(int, box)
    full = np.zeros(full_shape[:2], dtype=np.uint8)
    full[y1:y2, x1:x2] = crop_mask
    return full


def hsv_green_mask(crop_bgr):
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)
    return (mask > 0).astype(np.uint8)


def local_std_map(gray, window_size=9):
    gray_f = gray.astype(np.float32)
    mean = cv2.blur(gray_f, (window_size, window_size))
    mean_sq = cv2.blur(gray_f * gray_f, (window_size, window_size))
    var = np.maximum(mean_sq - mean * mean, 0)
    std = np.sqrt(var)
    return std


def morphology(mask):
    kernel = np.ones((MORPH_KERNEL, MORPH_KERNEL), np.uint8)
    out = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel)
    return out


def largest_component(mask):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if num_labels <= 1:
        return None

    max_area = 0
    best_idx = -1
    for idx in range(1, num_labels):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area < MIN_COMPONENT_AREA:
            continue
        if area > max_area:
            max_area = area
            best_idx = idx

    if best_idx == -1:
        return None

    return (labels == best_idx).astype(np.uint8)


def build_initial_mask(image, gt_box):
    x1, y1, x2, y2 = map(int, gt_box)
    crop = image[y1:y2, x1:x2]

    if crop.size == 0:
        return None, None, None, None, None

    green = hsv_green_mask(crop)

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    std_map = local_std_map(gray, window_size=WINDOW_SIZE)

    texture_mask = (std_map >= STD_THRESH).astype(np.uint8)

    joint = (green & texture_mask).astype(np.uint8)
    joint_refined = morphology(joint)
    best = largest_component(joint_refined)

    if best is None:
        return green, std_map, texture_mask, joint_refined, None

    return green, std_map, texture_mask, joint_refined, best


def find_holes(binary_mask):
    h, w = binary_mask.shape
    flood = (binary_mask * 255).astype(np.uint8).copy()

    ff_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, ff_mask, (0, 0), 255)

    flood_inv = cv2.bitwise_not(flood)
    holes = (flood_inv > 0).astype(np.uint8)

    holes = holes & (1 - binary_mask)
    return holes


def connected_components_with_stats(mask):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    results = []
    for idx in range(1, num_labels):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        comp = (labels == idx).astype(np.uint8)
        results.append((idx, comp, area))
    return results


def touches_crop_edge(hole_mask, margin=2):
    """
    洞如果碰到 bbox crop 边缘，就认为是贴边洞，不填
    """
    h, w = hole_mask.shape
    ys, xs = np.where(hole_mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return False

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    if x_min <= margin or y_min <= margin or x_max >= (w - 1 - margin) or y_max >= (h - 1 - margin):
        return True
    return False


def contrast_aware_hole_recovery(crop_bgr, binary_mask):
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    repaired = binary_mask.copy().astype(np.uint8)

    holes = find_holes(binary_mask)
    hole_comps = connected_components_with_stats(holes)

    debug_keep = np.zeros_like(binary_mask, dtype=np.uint8)
    debug_fill = np.zeros_like(binary_mask, dtype=np.uint8)
    debug_edge = np.zeros_like(binary_mask, dtype=np.uint8)

    kernel = np.ones((RING_DILATE, RING_DILATE), np.uint8)
    hole_records = []

    for _, hole_mask, area in hole_comps:
        # 1) 大洞默认保留
        if area > MAX_HOLE_AREA:
            debug_keep[hole_mask > 0] = 1
            hole_records.append({
                "area": area,
                "action": "keep_large"
            })
            continue

        # 2) 贴边洞默认保留
        if touches_crop_edge(hole_mask, margin=EDGE_TOUCH_MARGIN):
            debug_keep[hole_mask > 0] = 1
            debug_edge[hole_mask > 0] = 1
            hole_records.append({
                "area": area,
                "action": "keep_edge_hole"
            })
            continue

        # 3) 做 ring
        dilated = cv2.dilate(hole_mask.astype(np.uint8), kernel, iterations=1)
        ring = (dilated - hole_mask).clip(0, 1).astype(np.uint8)
        ring = ring & binary_mask.astype(np.uint8)

        if ring.sum() == 0:
            repaired[hole_mask > 0] = 1
            debug_fill[hole_mask > 0] = 1
            hole_records.append({
                "area": area,
                "action": "fill_no_ring"
            })
            continue

        hole_mean = float(gray[hole_mask > 0].mean())
        ring_mean = float(gray[ring > 0].mean())
        delta = ring_mean - hole_mean

        # 4) 相对更暗 -> 保留
        if delta >= DELTA_THRESH:
            debug_keep[hole_mask > 0] = 1
            action = "keep_dark_hole"
        else:
            repaired[hole_mask > 0] = 1
            debug_fill[hole_mask > 0] = 1
            action = "fill_false_hole"

        hole_records.append({
            "area": area,
            "hole_mean": hole_mean,
            "ring_mean": ring_mean,
            "delta": delta,
            "action": action
        })

    return repaired, holes, debug_keep, debug_fill, debug_edge, hole_records
