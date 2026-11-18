"""CrownRoute operations extracted from step10_2_fastsam_refiner_full.py."""

import cv2
import numpy as np


MIN_MASK_AREA = 100
LOWER_GREEN = np.array([25, 20, 20], dtype=np.uint8)
UPPER_GREEN = np.array([95, 255, 255], dtype=np.uint8)
KERNEL_SIZE = 5


def mask_to_bbox(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def bbox_iou_xyxy(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter = inter_w * inter_h

    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])

    union = area1 + area2 - inter + 1e-6
    return inter / union


def get_candidate_masks_from_fastsam(result):
    if result.masks is None or result.masks.data is None:
        return []

    masks = result.masks.data.cpu().numpy()
    candidate_masks = []

    for m in masks:
        m_bin = (m > 0.5).astype(np.uint8)
        area = int(m_bin.sum())
        if area < MIN_MASK_AREA:
            continue
        if mask_to_bbox(m_bin) is None:
            continue
        candidate_masks.append(m_bin)

    return candidate_masks


def bbox_center(box):
    x1, y1, x2, y2 = box
    return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32)


def mask_center(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return np.array([xs.mean(), ys.mean()], dtype=np.float32)


def compute_inside_ratio(mask, gt_box):
    """
    候选mask有多少比例落在gt bbox内
    """
    x1, y1, x2, y2 = gt_box
    total_area = mask.sum()
    if total_area <= 0:
        return 0.0

    inside = mask[y1:y2, x1:x2].sum()
    return float(inside / (total_area + 1e-6))


def compute_center_score(mask, gt_box, image_shape):
    """
    候选mask中心和gt bbox中心越近越好
    """
    h, w = image_shape[:2]
    mc = mask_center(mask)
    if mc is None:
        return 0.0

    gc = bbox_center(gt_box)
    dist = np.linalg.norm(mc - gc)

    # 用图像对角线做归一化
    diag = np.sqrt(h * h + w * w) + 1e-6
    norm_dist = dist / diag
    score = 1.0 - norm_dist
    return float(max(0.0, min(1.0, score)))


def compute_green_ratio(image_bgr, mask):
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)
    green_mask = (green_mask > 0).astype(np.uint8)

    mask_area = mask.sum()
    if mask_area <= 0:
        return 0.0

    green_inside = (green_mask & mask).sum()
    return float(green_inside / (mask_area + 1e-6))


def choose_best_mask_with_multiscore(gt_box, candidate_masks, image_bgr):
    """
    升级版：
    综合 bbox_iou + inside_ratio + center_score + green_ratio
    """
    best_score = -1
    best_mask = None
    best_box = None
    best_detail = None

    for m in candidate_masks:
        pred_box = mask_to_bbox(m)
        if pred_box is None:
            continue

        iou = bbox_iou_xyxy(gt_box, pred_box)
        inside_ratio = compute_inside_ratio(m, gt_box)
        center_score = compute_center_score(m, gt_box, image_bgr.shape)
        green_ratio = compute_green_ratio(image_bgr, m)

        score = (
            0.35 * iou +
            0.30 * inside_ratio +
            0.20 * center_score +
            0.15 * green_ratio
        )

        if score > best_score:
            best_score = score
            best_mask = m
            best_box = pred_box
            best_detail = {
                "bbox_iou": float(iou),
                "inside_ratio": float(inside_ratio),
                "center_score": float(center_score),
                "green_ratio": float(green_ratio),
                "final_score": float(score)
            }

    return best_mask, best_box, best_detail


def clip_mask_to_bbox(mask, bbox):
    x1, y1, x2, y2 = bbox
    clipped = np.zeros_like(mask, dtype=np.uint8)
    clipped[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
    return clipped


def hsv_green_filter(image_bgr):
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)
    green_mask = (green_mask > 0).astype(np.uint8)
    return green_mask


def keep_largest_connected_component(mask):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if num_labels <= 1:
        return mask.astype(np.uint8)

    largest_area = 0
    largest_idx = 1
    for idx in range(1, num_labels):
        area = stats[idx, cv2.CC_STAT_AREA]
        if area > largest_area:
            largest_area = area
            largest_idx = idx

    kept = (labels == largest_idx).astype(np.uint8)
    return kept


def morphological_refine(mask):
    kernel = np.ones((KERNEL_SIZE, KERNEL_SIZE), np.uint8)
    refined = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, kernel)
    return refined


def refine_mask_first_version(image_bgr, raw_mask, bbox):
    mask1 = clip_mask_to_bbox(raw_mask, bbox)
    green_mask = hsv_green_filter(image_bgr)
    mask2 = (mask1 & green_mask).astype(np.uint8)
    mask3 = keep_largest_connected_component(mask2)
    mask4 = morphological_refine(mask3)

    return {
        "clipped_mask": mask1,
        "green_mask": green_mask,
        "green_filtered_mask": mask2,
        "largest_cc_mask": mask3,
        "refined_mask": mask4
    }
