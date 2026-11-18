from __future__ import annotations
import json
import math
import time
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision
from PIL import Image, ImageFilter
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from model import build_model
from threshold_metrics import metrics
from test_access_guard import install_test_access_guard


def inference_model(device):
    stack = {"np": np, "torch": torch, "torchvision": torchvision, "Image": Image}
    model, pretrained_status = build_model(stack, "none", pretrained=False)
    model.roi_heads.score_thresh = 0.001
    model.roi_heads.detections_per_img = 100
    return model.to(device).eval(), pretrained_status


def log(msg: str) -> None:
    print(msg, flush=True)


def load_checkpoint(model, path: Path) -> dict[str, Any]:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise TypeError(f"Checkpoint is not a state dict: {path}")
    result = model.load_state_dict(state, strict=True)
    return {
        "state_keys": len(state),
        "missing_keys": list(result.missing_keys),
        "unexpected_keys": list(result.unexpected_keys),
    }


def load_tensor(path: Path) -> torch.Tensor:
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def coco_from_dict(data: dict) -> COCO:
    coco = COCO()
    coco.dataset = data
    coco.createIndex()
    return coco


def encode_rle(mask: np.ndarray) -> dict[str, Any]:
    from pycocotools import mask as mask_utils

    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    counts = rle["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("ascii")
    return {"size": list(rle["size"]), "counts": counts}


def predictions_to_coco(
    image_id: int, output: dict[str, torch.Tensor], threshold: float
) -> tuple[list[dict], list[dict]]:
    boxes = output["boxes"].detach().cpu().numpy()
    scores = output["scores"].detach().cpu().numpy()
    masks = output["masks"].detach().cpu().numpy()[:, 0]
    if (
        not np.isfinite(boxes).all()
        or not np.isfinite(scores).all()
        or not np.isfinite(masks).all()
    ):
        raise FloatingPointError("NaN/Inf in model output")
    bbox_rows, segm_rows = [], []
    for idx in np.argsort(-scores):
        score = float(scores[idx])
        if score < threshold:
            continue
        x1, y1, x2, y2 = [float(v) for v in boxes[idx]]
        bbox = [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]
        common = {"image_id": image_id, "category_id": 1, "bbox": bbox, "score": score}
        bbox_rows.append(common)
        segm_rows.append(
            {**common, "segmentation": encode_rle((masks[idx] >= 0.5).astype(np.uint8))}
        )
    return bbox_rows, segm_rows


def cross_stem(value: str) -> str:
    return Path(value.replace("\\", "/")).stem


def clone_with_counts(seg: dict[str, Any], counts: Any) -> dict[str, Any]:
    out = deepcopy(seg)
    out["counts"] = counts
    return out


def decode_rle_binary(segmentation: Any) -> np.ndarray:
    """Decode compressed RLE without calling pycocotools maskUtils.iou.

    final5b keeps final5's safe numpy AP but makes GT/pred mask decoding
    tolerant to pycocotools builds that require compressed RLE counts as bytes.
    """
    attempts: list[Any] = [segmentation]
    if isinstance(segmentation, dict):
        counts = segmentation.get("counts")
        if isinstance(counts, str):
            attempts.append(clone_with_counts(segmentation, counts.encode("ascii")))
        elif isinstance(counts, bytes):
            attempts.append(
                clone_with_counts(segmentation, counts.decode("ascii").encode("ascii"))
            )
        elif isinstance(counts, list):
            h, w = segmentation.get("size", [None, None])
            if h and w:
                attempts.append(mask_utils.frPyObjects(segmentation, int(h), int(w)))
    errors = []
    for seg in attempts:
        try:
            mask = mask_utils.decode(seg)
            if mask.ndim == 3:
                mask = np.any(mask, axis=2)
            return mask.astype(bool)
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
    raise ValueError("RLE_DECODE_FAILED_FINAL5B " + " | ".join(errors))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def topk_predictions(
    rows: list[dict[str, Any]], max_dets_per_image: int
) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(int(row["image_id"]), []).append(row)
    kept: list[dict[str, Any]] = []
    for image_id in sorted(grouped):
        chunk = sorted(
            grouped[image_id], key=lambda r: float(r.get("score", 0.0)), reverse=True
        )
        kept.extend(chunk[:max_dets_per_image])
    return kept


def summarize_bbox_coco(
    method: str,
    checkpoint_tag: str,
    split: str,
    coco_gt,
    bbox_rows: list[dict],
    max_dets_per_image: int,
) -> dict[str, float]:
    if not bbox_rows:
        return {
            k: 0.0 for k in ["ap", "ap50", "ap75", "ap_small", "ap_medium", "ap_large"]
        }
    log(
        f"BBOX_COCOEVAL_START method={method} ckpt={checkpoint_tag} split={split} predictions={len(bbox_rows)}"
    )
    coco_dt = coco_gt.loadRes(bbox_rows)
    ev = COCOeval(coco_gt, coco_dt, "bbox")
    ev.params.maxDets = [1, 10, int(max_dets_per_image)]
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    s = ev.stats
    out = {
        "ap": float(s[0]),
        "ap50": float(s[1]),
        "ap75": float(s[2]),
        "ap_small": float(s[3]),
        "ap_medium": float(s[4]),
        "ap_large": float(s[5]),
    }
    log(
        f"BBOX_COCOEVAL_DONE method={method} ckpt={checkpoint_tag} split={split} ap={out['ap']:.6f}"
    )
    return out


def _safe_mask_ap(
    method: str,
    checkpoint_tag: str,
    split: str,
    gt: dict[str, Any],
    pred_rows: list[dict[str, Any]],
    image_ids: list[int],
    max_dets_per_image: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    thresholds = [round(x, 2) for x in np.arange(0.50, 0.96, 0.05)]
    image_id_set = {int(x) for x in image_ids}
    failed_gt: list[dict[str, Any]] = []
    failed_pred: list[dict[str, Any]] = []
    gt_by_image: dict[int, list[dict[str, Any]]] = {int(x): [] for x in image_ids}
    pred_by_image: dict[int, list[dict[str, Any]]] = {int(x): [] for x in image_ids}

    gt_seg_types: dict[str, int] = {}
    for ann in gt.get("annotations", []):
        img_id = int(ann["image_id"])
        if img_id not in image_id_set:
            continue
        seg = ann.get("segmentation")
        gt_seg_types[type(seg).__name__] = gt_seg_types.get(type(seg).__name__, 0) + 1
        try:
            mask = decode_rle_binary(seg)
            gt_by_image.setdefault(img_id, []).append(
                {
                    "ann_id": int(ann.get("id", len(gt_by_image.get(img_id, [])))),
                    "mask": mask,
                    "area": float(ann.get("area", mask.sum())),
                }
            )
        except Exception as exc:
            failed_gt.append(
                {
                    "image_id": img_id,
                    "annotation_id": ann.get("id"),
                    "error": repr(exc),
                    "segmentation_type": type(seg).__name__,
                }
            )

    for idx, pred in enumerate(pred_rows):
        img_id = int(pred["image_id"])
        if img_id not in image_id_set:
            continue
        try:
            mask = decode_rle_binary(pred["segmentation"])
            pred_by_image.setdefault(img_id, []).append(
                {
                    "pred_id": idx,
                    "mask": mask,
                    "score": float(pred.get("score", 0.0)),
                    "image_id": img_id,
                }
            )
        except Exception as exc:
            failed_pred.append(
                {
                    "image_id": img_id,
                    "pred_index": idx,
                    "score": pred.get("score"),
                    "error": repr(exc),
                }
            )

    for img_id in list(pred_by_image):
        pred_by_image[img_id].sort(key=lambda x: x["score"], reverse=True)
        pred_by_image[img_id] = pred_by_image[img_id][:max_dets_per_image]

    pred_mask_sizes = [
        list(x["mask"].shape)
        for rows in pred_by_image.values()
        for x in rows[:max_dets_per_image]
    ]
    width_height_stats = sorted(
        {
            (int(img.get("width", -1)), int(img.get("height", -1)))
            for img in gt.get("images", [])
            if int(img["id"]) in image_id_set
        }
    )
    debug = {
        "method": method,
        "checkpoint_tag": checkpoint_tag,
        "split": split,
        "image_count": len(image_ids),
        "gt_ann_count": sum(len(v) for v in gt_by_image.values()),
        "pred_count": sum(len(v) for v in pred_by_image.values()),
        "width_height_stats": width_height_stats,
        "pred_mask_size_stats": sorted({tuple(x) for x in pred_mask_sizes})[:20],
        "gt_seg_type_stats": gt_seg_types,
        "failed_gt_masks": len(failed_gt),
        "failed_pred_masks": len(failed_pred),
        "max_dets_per_image": max_dets_per_image,
    }
    log("SEGM_DEBUG " + json.dumps(debug, ensure_ascii=False))

    aps: dict[float, float] = {}
    total_gt = sum(len(v) for v in gt_by_image.values())
    if total_gt == 0:
        metrics = {
            "mask_ap": 0.0,
            "mask_ap50": 0.0,
            "mask_ap75": 0.0,
            "mask_ap_small": "NA",
            "mask_ap_medium": "NA",
            "mask_ap_large": "NA",
        }
        return metrics, failed_gt, failed_pred, debug

    for img_id in image_ids:
        log(
            f"SAFE_MASK_AP_IMAGE_DONE image_id={img_id} preds={len(pred_by_image.get(img_id, []))} gts={len(gt_by_image.get(img_id, []))}"
        )

    all_preds = [p for rows in pred_by_image.values() for p in rows]
    all_preds.sort(key=lambda x: x["score"], reverse=True)
    for thr in thresholds:
        matched: set[tuple[int, int]] = set()
        tp: list[float] = []
        fp: list[float] = []
        for pred in all_preds:
            img_id = int(pred["image_id"])
            best_iou = -1.0
            best_key: tuple[int, int] | None = None
            for gt_idx, gt_item in enumerate(gt_by_image.get(img_id, [])):
                key = (img_id, gt_idx)
                if key in matched:
                    continue
                inter = np.logical_and(pred["mask"], gt_item["mask"]).sum(
                    dtype=np.float64
                )
                union = np.logical_or(pred["mask"], gt_item["mask"]).sum(
                    dtype=np.float64
                )
                iou = float(inter / union) if union > 0 else 0.0
                if iou > best_iou:
                    best_iou = iou
                    best_key = key
            if best_key is not None and best_iou >= thr:
                matched.add(best_key)
                tp.append(1.0)
                fp.append(0.0)
            else:
                tp.append(0.0)
                fp.append(1.0)
        if not tp:
            aps[thr] = 0.0
            continue
        tp_c = np.cumsum(np.asarray(tp, dtype=np.float64))
        fp_c = np.cumsum(np.asarray(fp, dtype=np.float64))
        recall = tp_c / max(float(total_gt), 1.0)
        precision = tp_c / np.maximum(tp_c + fp_c, 1e-12)
        ap = 0.0
        for r in np.linspace(0.0, 1.0, 101):
            valid = precision[recall >= r]
            ap += float(valid.max()) if valid.size else 0.0
        aps[thr] = ap / 101.0
        log(
            f"SAFE_MASK_AP_THRESHOLD_DONE method={method} ckpt={checkpoint_tag} split={split} thr={thr:.2f} ap={aps[thr]:.6f}"
        )

    metrics = {
        "mask_ap": float(np.mean([aps[t] for t in thresholds])),
        "mask_ap50": float(aps[0.50]),
        "mask_ap75": float(aps[0.75]),
        "mask_ap_small": "NA",
        "mask_ap_medium": "NA",
        "mask_ap_large": "NA",
    }
    return metrics, failed_gt, failed_pred, debug


def safe_mask_ap(*args, **kwargs):
    metrics, failed_gt, failed_pred, debug = _safe_mask_ap(*args, **kwargs)
    if failed_gt:
        raise RuntimeError(
            f"FINAL5B_ABORT_FAILED_GT_MASKS failed_gt_masks={len(failed_gt)} debug={debug}"
        )
    return metrics, failed_gt, failed_pred, debug


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_manual_split_final5b(
    manual_root: Path, dataset: Path, split: str, limit: int | None
) -> tuple[dict, list[dict]]:
    manual_root = manual_root.resolve()
    samples_root = manual_root.parent / "manual_gt_assets"
    normalized_path = manual_root / f"normalized_gt_{split}_manual_coco.json"
    if not normalized_path.is_file():
        raise FileNotFoundError(
            f"final5b normalized GT is required before evaluation: {normalized_path}. "
            "Supply normalized validation annotations with decodable RLE masks."
        )
    if not samples_root.is_dir():
        raise FileNotFoundError(
            f"Missing sibling manual_gt_assets directory: {samples_root}"
        )
    gt = read_json(normalized_path)
    samples = read_json(samples_root / f"gt_{split}_samples.json")
    original_sample_paths = {
        str(sample["patch_stem"]): str(sample.get("file_name", ""))
        for sample in samples
    }
    map_path = samples_root / "patch_image_map.json"
    patch_map = read_json(map_path) if map_path.exists() else {}
    patch_map = {key: str(samples_root / path) for key, path in patch_map.items()}
    records = read_json(dataset / "meta" / "dataset_records.json")
    dataset_map = {
        cross_stem(str(r.get("patch_image", r["id"]))): str(dataset / r["image_path"])
        for r in records
    }
    for patch, local_path in dataset_map.items():
        if patch not in patch_map or not Path(str(patch_map[patch])).is_file():
            patch_map[patch] = local_path
    sample_by_id = {int(x["id"]): x for x in samples}
    for image in gt["images"]:
        patch = image.get("patch_stem") or sample_by_id[int(image["id"])]["patch_stem"]
        if patch not in patch_map:
            raise KeyError(f"No packaged image mapping for manual-GT patch: {patch}")
        mapped = Path(str(patch_map[patch]))
        original = samples_root / original_sample_paths.get(str(patch), "")
        image["file_name"] = str(mapped if mapped.is_file() else original)
    for sample in samples:
        patch = sample["patch_stem"]
        if patch not in patch_map:
            raise KeyError(f"No packaged image mapping for sample: {patch}")
        mapped = Path(str(patch_map[patch]))
        original = samples_root / original_sample_paths.get(str(patch), "")
        sample["file_name"] = str(mapped if mapped.is_file() else original)
    if limit is not None:
        keep_samples = samples[:limit]
        keep_ids = {int(x["id"]) for x in keep_samples}
        gt["images"] = [x for x in gt["images"] if int(x["id"]) in keep_ids]
        gt["annotations"] = [
            x for x in gt["annotations"] if int(x["image_id"]) in keep_ids
        ]
        samples = keep_samples
    for sample in samples:
        if not Path(sample["file_name"]).is_file():
            raise FileNotFoundError(sample["file_name"])
    return gt, samples


def evaluate_checkpoint_final5(
    method: str,
    checkpoint_tag: str,
    checkpoint: Path,
    split: str,
    gt: dict,
    samples: list[dict],
    device: torch.device,
    threshold: float,
    max_dets_per_image: int,
    save_dir: Path | None,
) -> tuple[
    dict[str, Any],
    list[dict],
    list[dict],
    dict[str, Any],
    list[dict],
    list[dict],
    dict[str, Any],
]:
    log(f"EVAL_CHECKPOINT_START method={method} ckpt={checkpoint.name} split={split}")
    wall_start = time.perf_counter()
    coco_gt = coco_from_dict(gt)
    model, _pre = inference_model(device)
    load = load_checkpoint(model, checkpoint)
    params = int(sum(p.numel() for p in model.parameters()))
    trainable = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    bbox_rows, segm_rows, timings = [], [], []
    with torch.inference_mode():
        for idx, sample in enumerate(samples, 1):
            tensor = load_tensor(Path(sample["file_name"])).to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            output = model([tensor])[0]
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            if idx > 5:
                timings.append(elapsed)
            b, s = predictions_to_coco(int(sample["id"]), output, threshold)
            bbox_rows.extend(b)
            segm_rows.extend(s)
            if idx % 25 == 0 or idx == len(samples):
                log(f"[{method} {checkpoint.name} {split}] {idx}/{len(samples)}")
    before = len(segm_rows)
    log(
        f"INFERENCE_DONE method={method} ckpt={checkpoint.name} split={split} predictions={before}"
    )
    bbox_rows = topk_predictions(bbox_rows, max_dets_per_image)
    segm_rows = topk_predictions(segm_rows, max_dets_per_image)
    log(
        f"TOPK_DONE method={method} split={split} before={before} after={len(segm_rows)} max_dets_per_image={max_dets_per_image}"
    )
    box = summarize_bbox_coco(
        method, checkpoint_tag, split, coco_gt, bbox_rows, max_dets_per_image
    )
    image_ids = [int(x["id"]) for x in gt.get("images", [])]
    mask, failed_gt, failed_pred, safe_debug = safe_mask_ap(
        method, checkpoint_tag, split, gt, segm_rows, image_ids, max_dets_per_image
    )
    mean = float(np.mean(timings)) if timings else float("nan")
    median = float(np.median(timings)) if timings else float("nan")
    row = {
        "method": method,
        "checkpoint": str(checkpoint),
        "split": split,
        "gt_type": "authoritative_manual_polygon",
        "images": len(samples),
        "annotations": len(gt["annotations"]),
        "box_ap": box["ap"],
        "box_ap50": box["ap50"],
        "box_ap75": box["ap75"],
        "box_ap_small": box["ap_small"],
        "box_ap_medium": box["ap_medium"],
        "box_ap_large": box["ap_large"],
        "mask_ap": mask["mask_ap"],
        "mask_ap50": mask["mask_ap50"],
        "mask_ap75": mask["mask_ap75"],
        "mask_ap_small": mask["mask_ap_small"],
        "mask_ap_medium": mask["mask_ap_medium"],
        "mask_ap_large": mask["mask_ap_large"],
        "mean_time_s_per_image": mean,
        "median_time_s_per_image": median,
        "fps": 1.0 / mean if mean > 0 else float("nan"),
        "parameter_count": params,
        "trainable_parameter_count": trainable,
        "checkpoint_size_bytes": checkpoint.stat().st_size,
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "torch_version": torch.__version__,
        "evaluation_wall_seconds": time.perf_counter() - wall_start,
    }
    if save_dir:
        write_json(
            save_dir / f"{split}_{checkpoint.stem}_bbox_predictions.json", bbox_rows
        )
        write_json(
            save_dir / f"{split}_{checkpoint.stem}_segm_predictions.json", segm_rows
        )
    profile = {
        "parameter_count": params,
        "trainable_parameter_count": trainable,
        "checkpoint_size_bytes": checkpoint.stat().st_size,
        "load": load,
        "pretrained_requested_during_eval": False,
        "gpu": row["gpu"],
    }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return row, bbox_rows, segm_rows, profile, failed_gt, failed_pred, safe_debug


def decode(seg):
    if isinstance(seg, dict) and isinstance(seg.get("counts"), str):
        seg = {**seg, "counts": seg["counts"].encode("ascii")}
    out = mask_utils.decode(seg)
    if out.ndim == 3:
        out = np.any(out, axis=2)
    return out.astype(np.uint8)


def iou(a, b):
    inter = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    return inter / union if union else 1.0


def boundary_f1(a, b, tolerance=4):
    occupied = (a > 0) | (b > 0)
    if occupied.any():
        ys, xs = np.where(occupied)
        margin = tolerance + 2
        y0, y1 = (
            max(0, int(ys.min()) - margin),
            min(a.shape[0], int(ys.max()) + margin + 1),
        )
        x0, x1 = (
            max(0, int(xs.min()) - margin),
            min(a.shape[1], int(xs.max()) + margin + 1),
        )
        a, b = a[y0:y1, x0:x1], b[y0:y1, x0:x1]

    def morph(mask, size, mode):
        image = Image.fromarray((mask > 0).astype(np.uint8) * 255)
        filt = (
            ImageFilter.MaxFilter(size)
            if mode == "dilate"
            else ImageFilter.MinFilter(size)
        )
        return (np.asarray(image.filter(filt)) > 0).astype(np.uint8)

    ba = ((a > 0).astype(np.uint8) - morph(a, 3, "erode")).clip(0, 1)
    bb = ((b > 0).astype(np.uint8) - morph(b, 3, "erode")).clip(0, 1)
    if ba.sum() == 0 and bb.sum() == 0:
        return 1.0
    if ba.sum() == 0 or bb.sum() == 0:
        return 0.0
    size = 2 * tolerance + 1
    precision = float((bb & morph(ba, size, "dilate")).sum() / bb.sum())
    recall = float((ba & morph(bb, size, "dilate")).sum() / ba.sum())
    return 2 * precision * recall / max(precision + recall, 1e-12)


def match_mean(targets, preds):
    if not targets:
        return math.nan, math.nan
    if not preds:
        return 0.0, 0.0
    matrix = np.asarray([[iou(a, b) for b in preds] for a in targets], dtype=float)
    mapping = {}
    used_rows, used_cols = set(), set()
    for _ in range(min(matrix.shape)):
        best = (-1.0, None, None)
        for r in range(matrix.shape[0]):
            if r in used_rows:
                continue
            for c in range(matrix.shape[1]):
                if c in used_cols:
                    continue
                if matrix[r, c] > best[0]:
                    best = (float(matrix[r, c]), r, c)
        if best[1] is None:
            break
        mapping[int(best[1])] = int(best[2])
        used_rows.add(best[1])
        used_cols.add(best[2])
    ious, bfs = [], []
    for index, target in enumerate(targets):
        if index in mapping:
            pred = preds[mapping[index]]
            ious.append(iou(target, pred))
            bfs.append(boundary_f1(target, pred, 4))
        else:
            ious.append(0.0)
            bfs.append(0.0)
    return float(np.mean(ious)), float(np.mean(bfs))


class P06Evaluator:
    def __init__(self, dataset_root: Path, manual_root: Path, output_root: Path):
        install_test_access_guard()
        self.dataset_root = dataset_root.resolve()
        self.manual_root = manual_root.resolve()
        self.output_root = output_root.resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.gt, self.samples = load_manual_split_final5b(
            self.manual_root, self.dataset_root, "val", None
        )
        self.gt_path = self.manual_root / "normalized_gt_val_manual_coco.json"
        self.gt_by = defaultdict(list)
        for ann in self.gt["annotations"]:
            self.gt_by[int(ann["image_id"])].append(decode(ann["segmentation"]))
        self.pseudo_by_stem = self._load_pseudo()
        self.image_stem = {
            int(x["id"]): str(x.get("patch_stem") or Path(x["file_name"]).stem)
            for x in self.gt["images"]
        }

    def _load_pseudo(self):
        records = json.loads(
            (self.dataset_root / "meta" / "dataset_records.json").read_text(
                encoding="utf-8-sig"
            )
        )
        val_ids = set(
            (self.dataset_root / "split" / "val.txt")
            .read_text(encoding="utf-8")
            .split()
        )
        out = defaultdict(list)
        for rec in records:
            if rec["id"] not in val_ids:
                continue
            try:
                mask = np.asarray(
                    Image.open(self.dataset_root / rec["mask_path"]).convert("L")
                )
            except Exception:
                continue
            out[Path(rec["patch_image"]).stem].append((mask > 0).astype(np.uint8))
        return out

    def evaluate(
        self, checkpoint: Path, tag: str, method: str, pseudo_val_loss: float
    ) -> dict:
        import torch

        pred_dir = self.output_root / "predictions"
        pred_dir.mkdir(parents=True, exist_ok=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        split_name = f"val_{tag}"
        row, *_ = evaluate_checkpoint_final5(
            method,
            tag,
            checkpoint,
            split_name,
            self.gt,
            self.samples,
            device,
            0.05,
            100,
            pred_dir,
        )
        pred_path = pred_dir / f"{split_name}_{checkpoint.stem}_segm_predictions.json"
        preds = json.loads(pred_path.read_text(encoding="utf-8"))
        scan = []
        for threshold in [round(x / 100, 2) for x in range(5, 96, 5)]:
            m50, _ = metrics(self.gt, preds, threshold, 0.5)
            m75, _ = metrics(self.gt, preds, threshold, 0.75)
            scan.append({**m50, "f1_at_iou75": m75["f1"]})
        scan.sort(
            key=lambda x: (
                -x["f1"],
                -x["f1_at_iou75"],
                x["count_mae"],
                x["relative_area_error_mean"],
                abs(x["threshold"] - 0.5),
            )
        )
        selected = scan[0]
        threshold = float(selected["threshold"])
        by_image = defaultdict(list)
        for pred in preds:
            if float(pred.get("score", 0)) >= threshold:
                by_image[int(pred["image_id"])].append(decode(pred["segmentation"]))
        manual_ious, pseudo_ious, boundary_scores = [], [], []
        for iid in self.image_stem:
            current = by_image.get(iid, [])
            miou, bf = match_mean(self.gt_by[iid], current)
            piou, _ = match_mean(
                self.pseudo_by_stem.get(self.image_stem[iid], []), current
            )
            if math.isfinite(miou):
                manual_ious.append(miou)
            if math.isfinite(piou):
                pseudo_ious.append(piou)
            if math.isfinite(bf):
                boundary_scores.append(bf)
        return {
            "checkpoint_tag": tag,
            "optimizer_step": int(tag.split("step")[-1]),
            "pseudo_val_total_loss": float(pseudo_val_loss),
            "box_ap": float(row["box_ap"]),
            "box_ap50": float(row["box_ap50"]),
            "box_ap75": float(row["box_ap75"]),
            "mask_ap": float(row["mask_ap"]),
            "mask_ap50": float(row["mask_ap50"]),
            "mask_ap75": float(row["mask_ap75"]),
            "selected_threshold": threshold,
            "prediction_count": len(preds),
            "coverage_recall_iou50": float(selected["recall"]),
            "boundary_f1_4px": float(np.mean(boundary_scores))
            if boundary_scores
            else math.nan,
            "prediction_to_manual_iou": float(np.mean(manual_ious))
            if manual_ious
            else math.nan,
            "prediction_to_pseudo_iou": float(np.mean(pseudo_ious))
            if pseudo_ious
            else math.nan,
            "target_preference_gap": (
                float(np.mean(pseudo_ious)) - float(np.mean(manual_ious))
            )
            if manual_ious and pseudo_ious
            else math.nan,
            "selection_split": "val",
            "test_used_for_selection": False,
            "prediction_file": str(pred_path),
        }
