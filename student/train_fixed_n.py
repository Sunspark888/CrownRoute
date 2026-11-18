from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from model import build_model
from validation import P06Evaluator
from loss import install_patch
from train_c1 import (
    LOSS_KEYS,
    QualityDataset,
    atomic_json,
    collate,
    fixed_loader,
    model_targets,
    pseudo_audit,
    save_resume,
    write_csv,
)


from protocol import verify_c2_inputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fixed-N three-seed, fixed-update panel."
    )
    parser.add_argument("--arm", choices=["FIXED_N"], required=True)
    parser.add_argument("--init-weight", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--manual-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--eval-steps",
        type=Path,
        default=Path(__file__).with_name("eval_steps_47.json"),
    )
    parser.add_argument("--fixed-successful-updates", type=int, default=27422)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pseudo-audit-limit", type=int, default=128)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.fixed_successful_updates != 27422:
        raise RuntimeError("frozen successful-update budget must be 27422")
    if args.batch_size != 2 or args.seed not in (42, 43, 44):
        raise RuntimeError("fresh panel requires batch_size=2 and seed in {42,43,44}")
    verify_c2_inputs(args)
    if (args.dataset / "split" / "test.txt").exists():
        raise RuntimeError("test split must not be present in training dataset")
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("formal run requires CUDA")

    train_ds = QualityDataset(args.dataset, "train")
    pseudo_ds = QualityDataset(args.dataset, "val", args.pseudo_audit_limit)
    expected_train = 6857
    if (
        len(train_ds) != expected_train
        or len(QualityDataset(args.dataset, "val")) != 1459
    ):
        raise RuntimeError(f"frozen dataset count mismatch train={len(train_ds)}")
    steps_per_traversal = math.ceil(len(train_ds) / args.batch_size)
    target_updates = args.fixed_successful_updates
    eval_steps = [
        int(x) for x in json.loads(args.eval_steps.read_text(encoding="utf-8"))
    ]
    if eval_steps != sorted(set(eval_steps)) or eval_steps[-1] != target_updates:
        raise RuntimeError("evaluation schedule is not the frozen B0 actual schedule")

    install_patch()
    stack = {
        "torch": torch,
        "torchvision": __import__("torchvision"),
        "np": np,
        "Image": __import__("PIL.Image", fromlist=["Image"]),
    }
    model, pretrained = build_model(stack, "none", True)
    model.to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[20, 35], gamma=0.1
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    evaluator = P06Evaluator(args.dataset, args.manual_root, args.output / "validation")
    if len(evaluator.gt["images"]) != 235 or len(evaluator.gt["annotations"]) != 888:
        raise RuntimeError(
            "corrected manual validation must be 235 images / 888 annotations"
        )
    pseudo_loader = DataLoader(
        pseudo_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate,
    )

    protocol = {
        "experiment": f"E2_{args.arm}_seed{args.seed}",
        "role": "E2 fixed-N composition control; no final recommendation implied",
        "variant": "FIXED_N",
        "seed": args.seed,
        "batch_size": args.batch_size,
        "train_instances": len(train_ds),
        "val_instances": len(QualityDataset(args.dataset, "val")),
        "pseudo_val_audit_instances": len(pseudo_ds),
        "steps_per_combined_set_traversal": steps_per_traversal,
        "fixed_successful_optimizer_updates": target_updates,
        "equivalent_combined_set_traversals": target_updates / steps_per_traversal,
        "evaluation_steps": eval_steps,
        "optimizer": "SGD",
        "lr": 0.005,
        "momentum": 0.9,
        "weight_decay": 0.0005,
        "warmup": None,
        "gradient_accumulation": None,
        "augmentation": None,
        "lr_scheduler": {
            "name": "MultiStepLR",
            "epoch_milestones": [20, 35],
            "gamma": 0.1,
        },
        "amp": True,
        "pretrained": pretrained,
        "pretrained_weight_file": str(args.init_weight.resolve()),
        "initialization_identity": "existing frozen TORCH_HOME cache; no new hash performed",
        "loss_weighting": "B0 unweighted; no C-specific loss or weight",
        "checkpoint_selection": "maximum manual-GT validation Mask AP",
        "test_access": False,
        "manual_gt_use": "validation checkpoint selection and evaluation only",
    }
    atomic_json(args.output / "PROTOCOL.json", protocol)

    resume = args.output / "resume_latest.pth"
    epoch, start_batch, global_step, rows = 1, 0, 0, []
    loss_sums, loss_batches = defaultdict(float), 0
    if resume.exists():
        raise RuntimeError(
            "E2 existing resume requires explicit technical-failure recovery decision; do not restart blindly"
        )
        state = torch.load(resume, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        epoch = int(state["epoch"])
        start_batch = int(state["batch_in_epoch"])
        global_step = int(state["global_step"])
        rows = state["metric_rows"]
        loss_sums.update(state.get("loss_sums", {}))
        loss_batches = int(state.get("loss_batches", 0))
        random.setstate(state["rng_python"])
        np.random.set_state(state["rng_numpy"])
        torch.random.set_rng_state(state["rng_torch"])
        if torch.cuda.is_available() and state["rng_cuda"] is not None:
            torch.cuda.set_rng_state_all(state["rng_cuda"])
    raw_batches = 0
    amp_skipped = 0
    run_started = time.monotonic()
    best_ap = max([float(row["mask_ap"]) for row in rows], default=-1.0)

    def evaluate_current(tag: str, current_epoch: int, batch_position: int) -> None:
        nonlocal best_ap, rows
        audit = pseudo_audit(
            model, pseudo_loader, device, "B0", global_step, target_updates, True
        )
        model_path = args.output / "eval_model.pth"
        torch.save(model.state_dict(), model_path)
        metric = evaluator.evaluate(
            model_path, tag, f"fresh_{args.arm.lower()}", audit["pseudo_val_total_loss"]
        )
        model_path.unlink(missing_ok=True)
        metric.update(
            {
                "variant": args.arm,
                "raw_batches": raw_batches,
                "amp_skipped_updates": amp_skipped,
                "wall_clock_seconds": time.monotonic() - run_started,
                "combined_set_traversals": global_step / steps_per_traversal,
                "epoch_index": current_epoch,
                "epoch_end": batch_position >= steps_per_traversal,
                "train_total_loss_running": sum(loss_sums.values())
                / max(loss_batches, 1),
                **{
                    f"train_{key}_running": loss_sums[key] / max(loss_batches, 1)
                    for key in LOSS_KEYS
                },
                **audit,
                "effective_b_weight": 1.0,
                "loss_weighting": "B0_unweighted",
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
                "variant": args.arm,
                "epoch": current_epoch,
                "batch_in_epoch": batch_position,
                "optimizer_step": global_step,
                "best_mask_ap": best_ap,
                "test_access": 0,
            },
        )

    atomic_json(
        args.output / "STATUS.json",
        {
            "status": "RUNNING",
            "pid": os.getpid(),
            "variant": args.arm,
            "epoch": epoch,
            "optimizer_step": global_step,
            "target_successful_optimizer_steps": target_updates,
            "test_access": 0,
        },
    )
    eval_step_set = set(eval_steps)
    try:
        while global_step < target_updates:
            loader = fixed_loader(
                train_ds, args.batch_size, epoch, args.seed, start_batch
            )
            model.train()
            for local_index, (images, targets) in enumerate(loader, start=start_batch):
                images = [image.to(device) for image in images]
                clean, codes = model_targets(targets, device)
                raw_batches += 1
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=True):
                    losses = model(images, clean)
                    loss = sum(losses.values())
                if not torch.isfinite(loss):
                    raise RuntimeError(
                        f"non-finite loss at successful_step={global_step}"
                    )
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                before_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= before_scale:
                    global_step += 1
                else:
                    amp_skipped += 1
                for key, value in losses.items():
                    loss_sums[key] += float(value.detach().cpu())
                loss_batches += 1
                next_batch = local_index + 1
                if global_step in eval_step_set and not any(
                    int(row["optimizer_step"]) == global_step for row in rows
                ):
                    evaluate_current(f"step{global_step:05d}", epoch, next_batch)
                if global_step >= target_updates:
                    break
            if global_step >= target_updates:
                break
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

        if not any(int(row["optimizer_step"]) == target_updates for row in rows):
            evaluate_current(
                f"step{target_updates:05d}",
                epoch,
                min(start_batch, steps_per_traversal),
            )
        torch.save(model.state_dict(), args.output / "last.pth")
        atomic_json(
            args.output / "STATUS.json",
            {
                "status": "COMPLETE",
                "raw_batches": raw_batches,
                "amp_skipped_updates": amp_skipped,
                "wall_clock_seconds": time.monotonic() - run_started,
                "variant": args.arm,
                "optimizer_step": global_step,
                "target_successful_optimizer_steps": target_updates,
                "combined_set_traversals": global_step / steps_per_traversal,
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
                "variant": args.arm,
                "optimizer_step": global_step,
                "error": repr(exc),
                "test_access": 0,
            },
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
