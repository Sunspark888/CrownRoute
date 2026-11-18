"""CrownRoute operations extracted from step10_3_classify_instances_full.py."""


def classify_instance(inst):
    """
    三分类：
    A: 高置信，直接收
    B: 中置信，走 fallback
    C: 低置信，先搁置
    """

    if inst.get("status") != "ok":
        return "C"

    final_score = inst.get("final_score", 0.0)
    gt_coverage = inst.get("gt_coverage", 0.0)
    bbox_iou = inst.get("bbox_iou", 0.0)
    inside_ratio = inst.get("inside_ratio", 0.0)
    green_ratio = inst.get("green_ratio", 0.0)
    area_score = inst.get("area_score", 0.0)

    # ===== A类：高置信 =====
    if (final_score >= 0.80 and gt_coverage >= 0.70) or \
       (final_score >= 0.75 and bbox_iou >= 0.60 and inside_ratio >= 0.75):
        return "A"

    # ===== B类：中置信，适合 fallback =====
    if ((0.45 <= final_score < 0.80) and green_ratio >= 0.50) or \
       (final_score >= 0.35 and gt_coverage >= 0.25 and area_score >= 0.30):
        return "B"

    # ===== C类：低置信 / 极难例 =====
    return "C"
