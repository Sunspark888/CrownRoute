from collections import defaultdict
import numpy as np
from pycocotools import mask as mask_utils


def dec(seg):
    if isinstance(seg, dict) and isinstance(seg.get("counts"), str):
        seg = {**seg, "counts": seg["counts"].encode("ascii")}
    m = mask_utils.decode(seg)
    if m.ndim == 3:
        m = np.any(m, axis=2)
    return m.astype(bool)


def iou(a, b):
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 0.0


def metrics(gt, preds, thr, iou_thr):
    gt_by = defaultdict(list)
    pr_by = defaultdict(list)
    for a in gt["annotations"]:
        gt_by[int(a["image_id"])].append(dec(a["segmentation"]))
    for p in preds:
        if float(p.get("score", 0)) >= thr:
            pr_by[int(p["image_id"])].append(
                (float(p["score"]), dec(p["segmentation"]))
            )
    tp = fp = fn = 0
    count_err = []
    area_err = []
    failures = []
    for im in gt["images"]:
        iid = int(im["id"])
        gs = gt_by[iid]
        ps = sorted(pr_by[iid], reverse=True, key=lambda x: x[0])
        used = set()
        local_tp = 0
        for score, pm in ps:
            best = (-1, -1.0)
            for j, gm in enumerate(gs):
                if j in used:
                    continue
                v = iou(pm, gm)
                if v > best[1]:
                    best = (j, v)
            if best[0] >= 0 and best[1] >= iou_thr:
                used.add(best[0])
                tp += 1
                local_tp += 1
            else:
                fp += 1
        fn += len(gs) - len(used)
        ce = abs(len(ps) - len(gs))
        count_err.append(ce)
        ga = sum(int(x.sum()) for x in gs)
        pa = sum(int(x[1].sum()) for x in ps)
        ae = abs(pa - ga) / max(ga, 1)
        area_err.append(ae)
        failures.append(
            {
                "image_id": iid,
                "gt_count": len(gs),
                "pred_count": len(ps),
                "tp": local_tp,
                "fp": len(ps) - local_tp,
                "fn": len(gs) - local_tp,
                "count_abs_error": ce,
                "relative_area_error": ae,
            }
        )
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "threshold": thr,
        "iou_threshold": iou_thr,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "count_mae": float(np.mean(count_err)),
        "relative_area_error_mean": float(np.mean(area_err)),
    }, failures
