from __future__ import annotations
from typing import Any


def build_model(stack: dict[str, Any], attention: str, pretrained: bool):
    torch = stack["torch"]
    torchvision = stack["torchvision"]
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

    pretrained_status = {"requested": pretrained, "loaded": False, "error": ""}
    if pretrained:
        try:
            weights = torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights.COCO_V1
            model = torchvision.models.detection.maskrcnn_resnet50_fpn(
                weights=weights,
                min_size=512,
                max_size=512,
            )
            pretrained_status["loaded"] = True
        except Exception as exc:
            pretrained_status["error"] = repr(exc)
            raise RuntimeError(
                f"pretrained weights requested but could not be loaded: {exc!r}"
            ) from exc
    else:
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(
            weights=None,
            weights_backbone=None,
            num_classes=2,
            min_size=512,
            max_size=512,
        )

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    in_channels = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = model.roi_heads.mask_predictor.conv5_mask.out_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_channels, hidden_layer, 2)
    if attention != "none":
        raise ValueError("CrownRoute uses the unmodified R50-FPN backbone")
    return model, pretrained_status
