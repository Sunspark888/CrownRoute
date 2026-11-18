from pathlib import Path
import json
import hashlib
import os

WEIGHT_NAME = "maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth"
WEIGHT_SHA256 = "bf2d0c1efbc936eeee2bc95a48e80ebc86b891f61b0106485937fc29f9315fc0"
GT_SHA256 = "4c69835ef2513d2d4cd62f961c01f8763be58fc025dd519331960c02c19a429d"
SPLIT_IDENTITY = {
    "B0": [
        "7d6589f8affe866e7075e220cd865be906896942d8048e74811bcadc43f48a2b",
        "b67f142e26a756db8e6fadfe34eb289e934ebb25ab2a57b53a8b03bbcd5da2e6",
    ],
    "RECOVERED_C": [
        "1b0362e8bebcebe106a5d45456b6e0fd657a0029679802f76fdd7a64a1d00553",
        "b67f142e26a756db8e6fadfe34eb289e934ebb25ab2a57b53a8b03bbcd5da2e6",
    ],
    "FIXED_N": [
        "3879af0f7aab5661f6d3c70cfa8105f5a73ac805301f0368641db9ec579b95b4",
        "b67f142e26a756db8e6fadfe34eb289e934ebb25ab2a57b53a8b03bbcd5da2e6",
    ],
}


RECORD_IDENTITY = {
    "B0": "8caf8bf9a21edf5958e81bf5dd95eecea30409365711771f00e3f469473cac42",
    "RECOVERED_C": "f40349452e314372601fbd34a1d9367346bdcaf00568a87a8b9d36b0d674fea6",
    "FIXED_N": "fef70fe60c4cc5e39413d3335cced4feb1a2f4a2e2272308ba32e1c755893b83",
}


def record_identity(dataset):
    # Paths can be rebased; all scientific record fields and their ordering remain pinned.
    records = json.loads(
        (dataset / "meta" / "dataset_records.json").read_text(encoding="utf-8-sig")
    )
    payload = [
        {k: v for k, v in row.items() if k not in ("image_path", "mask_path")}
        for row in records
    ]
    text = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require_cached_weights():
    if not os.environ.get("TORCH_HOME"):
        raise ValueError(
            "Set TORCH_HOME to the cache containing the original COCO_V1 checkpoint"
        )
    weight = Path(os.environ["TORCH_HOME"]) / "hub" / "checkpoints" / WEIGHT_NAME
    if not weight.is_file() or digest(weight) != WEIGHT_SHA256:
        raise ValueError("Missing or different COCO_V1 checkpoint")
    return weight.resolve()


def verify_c2_inputs(args):
    if args.init_weight.resolve() != require_cached_weights():
        raise ValueError(
            "The initialization path must refer to the cached COCO_V1 checkpoint"
        )
    if record_identity(args.dataset) != RECORD_IDENTITY[args.arm]:
        raise ValueError("Frozen scientific record fields or order differed")
    for split, expected in zip(("train", "val"), SPLIT_IDENTITY[args.arm]):
        ids = (
            (args.dataset / "split" / (split + ".txt"))
            .read_text(encoding="utf-8-sig")
            .split()
        )
        text = "\n".join(ids) + "\n"
        if hashlib.sha256(text.encode()).hexdigest() != expected:
            raise ValueError(f"Frozen {split} membership/order differs for {args.arm}")
    if digest(args.manual_root / "normalized_gt_val_manual_coco.json") != GT_SHA256:
        raise ValueError("Corrected validation reference differs")
    reference = Path(__file__).with_name("eval_steps_47.json")
    if json.loads(args.eval_steps.read_text()) != json.loads(reference.read_text()):
        raise ValueError("Different observation grid")
