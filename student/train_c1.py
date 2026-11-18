from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset

from model import build_model
from validation import P06Evaluator
from loss import install_patch


LOSS_KEYS = [
    "loss_classifier",
    "loss_box_reg",
    "loss_mask",
    "loss_objectness",
    "loss_rpn_box_reg",
]
PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def atomic_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    tmp = path.with_suffix(path.suffix + ".partial")
    with tmp.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


class QualityDataset(Dataset):
    def __init__(self, root: Path, split: str, limit: int | None = None):
        records = json.loads(
            (root / "meta" / "dataset_records.json").read_text(encoding="utf-8-sig")
        )
        by_id = {r["id"]: r for r in records}
        ids = (root / "split" / f"{split}.txt").read_text(encoding="utf-8").split()
        if limit is not None:
            ids = ids[:limit]
        self.root = root
        self.records = [by_id[x] for x in ids]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        rec = self.records[index]
        image = (
            np.asarray(
                Image.open(self.root / rec["image_path"]).convert("RGB"),
                dtype=np.float32,
            )
            / 255.0
        )
        mask = np.asarray(Image.open(self.root / rec["mask_path"]).convert("L")) > 0
        ys, xs = np.nonzero(mask)
        if len(xs):
            box = [
                float(xs.min()),
                float(ys.min()),
                float(xs.max() + 1),
                float(ys.max() + 1),
            ]
            area = float((xs.max() + 1 - xs.min()) * (ys.max() + 1 - ys.min()))
        else:
            box, area = [0.0, 0.0, 1.0, 1.0], 1.0
        target = {
            "boxes": torch.tensor([box], dtype=torch.float32),
            "labels": torch.ones(1, dtype=torch.int64),
            "masks": torch.from_numpy(mask.astype(np.uint8)).unsqueeze(0),
            "image_id": torch.tensor([index], dtype=torch.int64),
            "area": torch.tensor([area], dtype=torch.float32),
            "iscrowd": torch.zeros(1, dtype=torch.int64),
            "quality_class": torch.tensor(
                [1 if rec.get("class_type") == "B" else 0], dtype=torch.int64
            ),
        }
        return torch.from_numpy(image).permute(2, 0, 1), target


def collate(batch):
    return tuple(zip(*batch))


def model_targets(targets, device):
    codes, clean = [], []
    for target in targets:
        codes.append(target["quality_class"].to(device))
        clean.append(
            {k: v.to(device) for k, v in target.items() if k != "quality_class"}
        )
    return clean, codes


def fixed_loader(dataset, batch_size, epoch, seed, start_batch=0):
    generator = torch.Generator().manual_seed(seed + epoch)
    order = torch.randperm(len(dataset), generator=generator).tolist()
    if start_batch:
        order = order[start_batch * batch_size :]
    return DataLoader(
        Subset(dataset, order),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate,
        pin_memory=True,
    )


def pseudo_audit(model, loader, device, variant, global_step, total_steps, amp):
    model.train()
    sums = defaultdict(float)
    count = 0
    with torch.no_grad():
        for images, targets in loader:
            images = [x.to(device) for x in images]
            clean, codes = model_targets(targets, device)
            with torch.amp.autocast("cuda", enabled=amp and device.type == "cuda"):
                losses = model(images, clean)
            for key, value in losses.items():
                sums[key] += float(value.detach().cpu())
            count += 1
    row = {"pseudo_val_total_loss": sum(sums.values()) / max(count, 1)}
    row.update({f"pseudo_val_{key}": sums[key] / max(count, 1) for key in LOSS_KEYS})
    return row


def save_resume(
    path,
    model,
    optimizer,
    scheduler,
    scaler,
    epoch,
    batch_in_epoch,
    global_step,
    rows,
    loss_sums,
    loss_batches,
):
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "batch_in_epoch": batch_in_epoch,
        "global_step": global_step,
        "metric_rows": rows,
        "loss_sums": dict(loss_sums),
        "loss_batches": loss_batches,
        "rng_python": random.getstate(),
        "rng_numpy": np.random.get_state(),
        "rng_torch": torch.random.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state_all()
        if torch.cuda.is_available()
        else None,
    }
    tmp = path.with_suffix(".partial")
    torch.save(state, tmp)
    os.replace(tmp, path)


def main():
    parser = argparse.ArgumentParser()
    parser.set_defaults(variant="B0")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--manual-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--pseudo-audit-limit", type=int, default=128)
    args = parser.parse_args()
    from protocol import require_cached_weights

    # The original R2 launcher required CUDA, checked the weight identity, and
    # refused to reuse an existing incomplete run. Each invocation starts fresh.
    if not torch.cuda.is_available():
        raise RuntimeError(
            "C1 training requires CUDA; use smoke_cpu.py for the CPU example"
        )
    require_cached_weights()
    args.output.mkdir(parents=True, exist_ok=False)
    for forbidden in ["normalized_gt_test_manual_coco.json", "gt_test_samples.json"]:
        if (args.manual_root / forbidden).exists():
            raise RuntimeError(f"TEST_ASSET_PRESENT={forbidden}")
    from test_access_guard import install_test_access_guard

    install_test_access_guard()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    train_ds = QualityDataset(args.dataset, "train")
    pseudo_ds = QualityDataset(args.dataset, "val", args.pseudo_audit_limit)
    steps_per_epoch = math.ceil(len(train_ds) / args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    install_patch()
    stack = {
        "torch": torch,
        "torchvision": __import__("torchvision"),
        "np": np,
        "Image": Image,
    }
    model, pretrained = build_model(stack, "none", True)
    model.to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[20, 35], gamma=0.1
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    evaluator = P06Evaluator(args.dataset, args.manual_root, args.output / "validation")
    pseudo_loader = DataLoader(
        pseudo_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate,
    )

    first = {250, 500, 750, 1000, 1500, 2000, 2500, 3000, steps_per_epoch}
    eval_steps = {x for x in first if x <= total_steps}
    eval_steps.update(range(4000, total_steps + 1, 1000))
    eval_steps.update({steps_per_epoch * epoch for epoch in range(1, args.epochs + 1)})
    if args.max_steps:
        eval_steps.add(int(args.max_steps))
    protocol = {
        "variant": args.variant,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "train_instances": len(train_ds),
        "pseudo_val_audit_instances": len(pseudo_ds),
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "evaluation_steps": sorted(eval_steps),
        "optimizer": "SGD",
        "lr": 0.005,
        "momentum": 0.9,
        "weight_decay": 0.0005,
        "amp": device.type == "cuda",
        "pretrained": pretrained,
        "test_access": False,
        "loss_weighting": "unweighted",
    }
    atomic_json(args.output / "PROTOCOL.json", protocol)

    resume = args.output / "resume_latest.pth"
    epoch, start_batch, global_step, rows = 1, 0, 0, []
    loss_sums, loss_batches = defaultdict(float), 0
    best_ap = max([float(x["mask_ap"]) for x in rows], default=-1.0)

    def evaluate_current(tag: str, current_epoch: int, batch_position: int):
        nonlocal best_ap, rows
        audit = pseudo_audit(
            model,
            pseudo_loader,
            device,
            args.variant,
            global_step,
            total_steps,
            device.type == "cuda",
        )
        model_path = args.output / "eval_model.pth"
        torch.save(model.state_dict(), model_path)
        metric = evaluator.evaluate(
            model_path,
            tag,
            f"p06_{args.variant.lower()}",
            audit["pseudo_val_total_loss"],
        )
        model_path.unlink(missing_ok=True)
        metric.update(
            {
                "variant": args.variant,
                "epoch_fraction": global_step / steps_per_epoch,
                "epoch_index": current_epoch,
                "epoch_end": batch_position >= steps_per_epoch,
                "train_total_loss_running": sum(loss_sums.values())
                / max(loss_batches, 1),
                **{
                    f"train_{k}_running": loss_sums[k] / max(loss_batches, 1)
                    for k in LOSS_KEYS
                },
                **audit,
                "effective_b_weight": 1.0,
            }
        )
        rows.append(metric)
        write_csv(args.output / "checkpoint_metrics.csv", rows)
        if metric["mask_ap"] > best_ap:
            best_ap = metric["mask_ap"]
            torch.save(model.state_dict(), args.output / "best_manual_val.pth")
            atomic_json(args.output / "best_manual_val.json", metric)
        save_resume(
            resume,
            model,
            optimizer,
            scheduler,
            scaler,
            current_epoch,
            batch_position,
            global_step,
            rows,
            loss_sums,
            loss_batches,
        )
        atomic_json(
            args.output / "STATUS.json",
            {
                "status": "RUNNING",
                "pid": os.getpid(),
                "variant": args.variant,
                "epoch": current_epoch,
                "batch_in_epoch": batch_position,
                "optimizer_step": global_step,
                "best_mask_ap": best_ap,
            },
        )

    atomic_json(
        args.output / "STATUS.json",
        {
            "status": "RUNNING",
            "pid": os.getpid(),
            "variant": args.variant,
            "epoch": epoch,
            "optimizer_step": global_step,
        },
    )
    try:
        while epoch <= args.epochs:
            loader = fixed_loader(
                train_ds, args.batch_size, epoch, args.seed, start_batch
            )
            model.train()
            for local_index, (images, targets) in enumerate(loader, start=start_batch):
                images = [x.to(device) for x in images]
                clean, codes = model_targets(targets, device)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                    losses = model(images, clean)
                    loss = sum(losses.values())
                if not torch.isfinite(loss):
                    raise RuntimeError(f"non-finite loss at {global_step}")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= before:
                    global_step += 1
                else:
                    with (args.output / "AMP_SKIPS.jsonl").open(
                        "a", encoding="utf-8"
                    ) as skip_log:
                        skip_log.write(
                            json.dumps(
                                {
                                    "epoch": epoch,
                                    "batch": local_index + 1,
                                    "successful_updates": global_step,
                                    "scale_before": before,
                                    "scale_after": scaler.get_scale(),
                                }
                            )
                            + "\n"
                        )
                for key, value in losses.items():
                    loss_sums[key] += float(value.detach().cpu())
                loss_batches += 1
                next_batch = local_index + 1
                if global_step in eval_steps and not any(
                    int(x["optimizer_step"]) == global_step for x in rows
                ):
                    tag = f"step{global_step:05d}"
                    evaluate_current(tag, epoch, next_batch)
                if args.max_steps and global_step >= args.max_steps:
                    save_resume(
                        resume,
                        model,
                        optimizer,
                        scheduler,
                        scaler,
                        epoch,
                        next_batch,
                        global_step,
                        rows,
                        loss_sums,
                        loss_batches,
                    )
                    atomic_json(
                        args.output / "STATUS.json",
                        {
                            "status": "SMOKE_COMPLETE",
                            "variant": args.variant,
                            "optimizer_step": global_step,
                            "test_access": 0,
                        },
                    )
                    return 0
            # Epoch-end evaluation is unconditional because AMP may skip optimizer steps.
            if not any(
                int(x["optimizer_step"]) == global_step
                and bool(x.get("epoch_end", False))
                for x in rows
            ):
                evaluate_current(f"step{global_step:05d}", epoch, steps_per_epoch)
            scheduler.step()
            epoch += 1
            start_batch = 0
            loss_sums = defaultdict(float)
            loss_batches = 0
            save_resume(
                resume,
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                0,
                global_step,
                rows,
                loss_sums,
                loss_batches,
            )
        torch.save(model.state_dict(), args.output / "last.pth")
        atomic_json(
            args.output / "STATUS.json",
            {
                "status": "COMPLETE",
                "variant": args.variant,
                "optimizer_step": global_step,
                "best_mask_ap": best_ap,
                "test_access": 0,
            },
        )
        atomic_json(
            args.output / "NO_TEST_ACCESS.json",
            {
                "status": "PASS",
                "test_files_opened": 0,
                "test_metrics_read": False,
                "test_used_for_selection": False,
            },
        )
        return 0
    except Exception as exc:
        atomic_json(
            args.output / "STATUS.json",
            {
                "status": "FAILED_RETRYABLE",
                "variant": args.variant,
                "optimizer_step": global_step,
                "error": repr(exc),
                "test_access": 0,
            },
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
