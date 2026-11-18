"""Materialize B0, RECOVERED_C and FIXED_N from authorized training/validation records."""

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import random
import tempfile


MEMBERSHIP_SEED = 20260927


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_standard(root):
    if (root / "split/test.txt").exists():
        raise ValueError("C2 requires a training/validation-only dataset; the input has a test split")
    records = read_json(root / "meta/dataset_records.json")
    by_id = {r["id"]: r for r in records}
    train, val = [
        (root / "split" / f"{split}.txt").read_text(encoding="utf-8-sig").split()
        for split in ("train", "val")
    ]
    if len(by_id) != len(records) or len(set(train + val)) != len(train) + len(val):
        raise ValueError("Duplicate member IDs")
    if set(train + val) != set(by_id):
        raise ValueError("Supply only the training and validation inventory")
    for split, ids in (("train", train), ("val", val)):
        if any(by_id[k]["source_disjoint_v3_split"] != split for k in ids):
            raise ValueError(f"Records disagree with the {split} member list")
    return records, [by_id[k] for k in train], [by_id[k] for k in val]


def source(record):
    return str(record["source_disjoint_v3_source_id"])


def fixed_members(standard, recovered):
    """Original fixed-size removal rule; use only source and member IDs."""
    grouped = defaultdict(list)
    for row in standard:
        if row["class_type"] == "B":
            grouped[source(row)].append(row["id"])
    counts = Counter(source(row) for row in recovered)
    sources = sorted(grouped.keys() | counts.keys())
    local = {s: min(len(grouped[s]), counts[s]) for s in sources}
    remaining = {s: len(grouped[s]) - local[s] for s in sources}
    shortage = len(recovered) - sum(local.values())
    available = sum(remaining.values())
    if shortage > available:
        raise ValueError("Insufficient Final-B members for fixed-size replacement")
    fallback = {s: 0 for s in sources}
    if shortage:
        fallback = {s: shortage * remaining[s] // available for s in sources}
        extra = shortage - sum(fallback.values())
        order = sorted(sources, key=lambda s: (-(shortage * remaining[s] % available), s))
        for s in order[:extra]:
            fallback[s] += 1
    rng = random.Random(MEMBERSHIP_SEED)
    removed = set()
    for s in sources:
        candidates = sorted(grouped[s])
        rng.shuffle(candidates)
        removed.update(candidates[:local[s] + fallback[s]])
    return [r for r in standard if r["id"] not in removed] + recovered, sum(local.values()), shortage


def recovered_from_expanded(root, standard, val):
    _, expanded_train, expanded_val = load_standard(root)
    def scientific_fields(row):
        return {k: v for k, v in row.items() if k not in {"image_path", "mask_path"}}
    for expected, actual in [(standard, expanded_train[:len(standard)]), (val, expanded_val)]:
        if list(map(scientific_fields, expected)) != list(map(scientific_fields, actual)):
            raise ValueError("Expanded and standard shared records differ or are reordered")
    return expanded_train[len(standard):]


def assemble(standard_root, recovered, output, base_assets, recovery_assets, fixed_order=None):
    records, train, val = load_standard(standard_root)
    train_sources = {source(r) for r in train}
    val_sources = {source(r) for r in val}
    if train_sources & val_sources or any(r["class_type"] not in {"A", "B"} for r in records):
        raise ValueError("Invalid A/B standard inventory or source leakage")
    if any(r["class_type"] != "C" or r["source_disjoint_v3_split"] != "train"
           or source(r) not in train_sources for r in recovered):
        raise ValueError("Recovered records must be training-side C from standard training sources")
    recovered_ids = [r["id"] for r in recovered]
    if len(set(recovered_ids)) != len(recovered_ids) or set(recovered_ids) & {r["id"] for r in records}:
        raise ValueError("Recovered IDs overlap or are duplicated")
    if recovered_ids != sorted(recovered_ids):
        raise ValueError("Recovered records must retain the original sorted instance order")
    fixed, local, fallback = fixed_members(train, recovered)
    fixed_ids = [r["id"] for r in fixed]
    if fixed_order is not None and fixed_order.read_text(encoding="utf-8-sig").split() != fixed_ids:
        raise ValueError("Recomputed fixed-size order differs from the supplied frozen list")
    if {source(r) for r in fixed} != train_sources:
        raise ValueError("Fixed-size substitution would lose a training source")
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as temp:
        staged = Path(temp) / "prepared"
        configurations = [
            ("B0", records, train),
            ("RECOVERED_C", train + recovered + val, train + recovered),
            ("FIXED_N", fixed + val, fixed),
        ]
        cache = {}
        for arm, inventory, training in configurations:
            root = staged / arm
            (root / "meta").mkdir(parents=True)
            (root / "split").mkdir()
            rewritten = []
            for row in inventory:
                item = dict(row)
                base = recovery_assets if row["class_type"] == "C" else base_assets
                for key in ["image_path", "mask_path"]:
                    path = base / row[key]
                    if path not in cache:
                        if not path.is_file():
                            raise FileNotFoundError(path)
                        cache[path] = path.resolve()
                    try:
                        item[key] = Path(os.path.relpath(cache[path], output / arm)).as_posix()
                    except ValueError:  # A different Windows volume has no relative path.
                        item[key] = cache[path].as_posix()
                rewritten.append(item)
            (root / "meta/dataset_records.json").write_text(json.dumps(rewritten, ensure_ascii=False),
                                                          encoding="utf-8")
            (root / "split/train.txt").write_text("".join(r["id"] + "\n" for r in training), encoding="utf-8")
            (root / "split/val.txt").write_bytes((standard_root / "split/val.txt").read_bytes())
        staged.rename(output)
    return {"standard": len(train), "relaxed": len(train) + len(recovered), "fixed": len(fixed),
            "validation": len(val), "within_source": local, "proportional": fallback}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--standard", type=Path, required=True)
    delta = parser.add_mutually_exclusive_group(required=True)
    delta.add_argument("--expanded", type=Path, help="Frozen expanded dataset with split/ and meta/")
    delta.add_argument("--recovered-records", type=Path, help="Ordered JSON list of successful training recoveries")
    parser.add_argument("--base-assets", required=True, type=Path)
    parser.add_argument("--recovery-assets", required=True, type=Path)
    parser.add_argument("--fixed-order", type=Path, help="Optional authoritative fixed-size split/train.txt")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.expanded:
        _, train, val = load_standard(args.standard)
        recovered = recovered_from_expanded(args.expanded, train, val)
    else:
        recovered = read_json(args.recovered_records)
    print(json.dumps(assemble(args.standard, recovered, args.output, args.base_assets,
                              args.recovery_assets, args.fixed_order)))


if __name__ == "__main__":
    main()
