"""Paired source-cluster bootstrap for the validation contrasts in Supplement S11."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


SEEDS = (42, 43, 44)
RANDOM_SEED = 20261020
N_DRAWS = 2000


def paired_intervals(rows, comparisons):
    """Return left-minus-right intervals, keeping each source's patches together."""
    methods = sorted({method for pair in comparisons for method in pair})
    if not methods:
        raise ValueError("Specify at least one pair of methods")
    membership = defaultdict(dict)
    totals = defaultdict(lambda: np.zeros(5, dtype=np.int64))
    for row in rows:
        method = row["method"]
        if method not in methods:
            continue
        seed = int(row["seed"])
        image, source = str(row["image_id"]).strip(), str(row["source"]).strip()
        if seed not in SEEDS or float(row["threshold"]) != 0.50:
            raise ValueError("S11 requires seeds 42, 43, 44 and a score threshold of 0.50")
        if not image or not source:
            raise ValueError("Every patch needs an image_id and original source identifier")
        patches = membership[method, seed]
        if image in patches:
            raise ValueError(f"Duplicate patch {image} in {method}, seed {seed}")
        tp, fp, fn, error = (
            int(row[key]) for key in ("tp", "fp", "fn", "count_abs_error")
        )
        if min(tp, fp, fn, error) < 0 or error != abs(fp - fn):
            raise ValueError(f"Inconsistent inventory counts for patch {image}")
        patches[image] = (source, tp + fn)
        totals[method, seed, source] += (tp, fp, fn, error, 1)

    reference = membership[methods[0], SEEDS[0]]
    if not reference:
        raise ValueError(f"Missing per-image data for {methods[0]}, seed {SEEDS[0]}")
    for method in methods:
        for seed in SEEDS:
            if membership[method, seed] != reference:
                raise ValueError(
                    f"Patch IDs, source IDs and reference counts must match for {method}, seed {seed}"
                )
    sources = sorted({source for source, _ in reference.values()})
    # The same matrix of draws is reused across every seed, method and contrast.
    rng = np.random.default_rng(RANDOM_SEED)
    draws = rng.integers(0, len(sources), size=(N_DRAWS, len(sources)))
    by_method = {}
    for method in methods:
        counts = np.array(
            [[totals[method, seed, source] for source in sources] for seed in SEEDS]
        )
        scores = []
        for selected in draws:
            tp, fp, fn, abs_error, patches = counts[:, selected, :].sum(axis=1).T
            denominator = 2 * tp + fp + fn
            if np.any(denominator == 0):
                raise ValueError(f"Undefined F1 for a resampled run in {method}")
            scores.append(((abs_error / patches).mean(), (2 * tp / denominator).mean() * 100))
        by_method[method] = np.array(scores)

    results = []
    for left, right in comparisons:
        low, high = np.quantile(by_method[left] - by_method[right], [0.025, 0.975], axis=0)
        for column, metric in enumerate(("count_mae_delta", "f1_percentage_point_delta")):
            results.append({
                "left": left, "right": right, "metric": metric,
                "ci_low": low[column], "ci_high": high[column],
                "source_clusters": len(sources), "bootstrap_draws": N_DRAWS,
                "scope": "conditional_selected_models_existing_validation_not_external_generalization",
            })
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-image", required=True, type=Path, help="Table 6 per_image.csv")
    parser.add_argument("--compare", required=True, action="append", nargs=2,
                        metavar=("LEFT", "RIGHT"), help="Method names; repeat for further contrasts")
    parser.add_argument("--output", type=Path, required=True, help="New CSV for the intervals")
    args = parser.parse_args()
    with args.per_image.open(encoding="utf-8-sig", newline="") as stream:
        rows = paired_intervals(csv.DictReader(stream), args.compare)
    with args.output.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
