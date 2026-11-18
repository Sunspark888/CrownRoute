import torch
import torch.nn.functional as F
from torchvision.models.detection.roi_heads import project_masks_on_boxes


def weighted_maskrcnn_loss(
    mask_logits, proposals, gt_masks, gt_labels, mask_matched_idxs
):
    discretization_size = mask_logits.shape[-1]
    labels = [gt_label[idxs] for gt_label, idxs in zip(gt_labels, mask_matched_idxs)]
    mask_targets = [
        project_masks_on_boxes(m, p, i, discretization_size)
        for m, p, i in zip(gt_masks, proposals, mask_matched_idxs)
    ]
    if not mask_targets:
        return mask_logits.sum() * 0
    labels_cat = torch.cat(labels, dim=0)
    targets = torch.cat(mask_targets, dim=0)
    if targets.numel() == 0:
        return mask_logits.sum() * 0
    logits = mask_logits[
        torch.arange(labels_cat.shape[0], device=labels_cat.device), labels_cat
    ]
    raw = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    weights = torch.ones_like(raw)

    return (raw * weights).sum() / weights.sum().clamp_min(1e-12)


def install_patch() -> None:
    import torchvision.models.detection.roi_heads as roi_heads

    roi_heads.maskrcnn_loss = weighted_maskrcnn_loss
