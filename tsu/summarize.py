"""Summarize the four prespecified TSU endpoints from Supplementary Data S1."""

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

STEPS = json.loads(
    (Path(__file__).resolve().parents[1] / "student" / "eval_steps_47.json").read_text()
)
SEEDS = (42, 43, 44)
SIZES = {"CrownRoute": 6857, "Boundary-relaxed": 8009, "Fixed-N": 6857}
METRICS = (
    "peak_mask_ap",
    "top5_mean_mask_ap",
    "normalized_auc_observed_span",
    "late_window_mean_mask_ap",
)


def endpoints(values):
    if len(values) != len(STEPS) or not all(
        math.isfinite(y) and 0 <= y <= 1 for y in values
    ):
        raise ValueError("Expected 47 observed Mask AP fractions, each between 0 and 1")
    progress = [step / 27422 for step in STEPS]
    area = sum(
        (end - start) * (left + right) / 2
        for start, end, left, right in zip(progress, progress[1:], values, values[1:])
    )
    late = [y for x, y in zip(progress, values) if x >= 0.75]
    return dict(
        zip(
            METRICS,
            (
                max(values),
                statistics.mean(sorted(values, reverse=True)[:5]),
                area / (progress[-1] - progress[0]),
                statistics.mean(late),
            ),
        )
    )


def summarize(trajectory):
    groups = defaultdict(dict)
    with trajectory.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            arm, seed = row["configuration"], int(row["seed"])
            if arm not in SIZES or seed not in SEEDS or int(row["N"]) != SIZES[arm]:
                raise ValueError("Unexpected configuration, membership count or seed")
            step = int(row["optimizer_step"])
            if step in groups[arm, seed]:
                raise ValueError(f"Duplicate observation: {arm}, {seed}, {step}")
            if not math.isclose(
                float(row["successful_update_progress"]),
                step / 27422,
                abs_tol=1e-12,
                rel_tol=0,
            ):
                raise ValueError(
                    "Progress disagrees with the successful update coordinate"
                )
            groups[arm, seed][step] = float(row["mask_ap"])
    rows = []
    for arm, count in SIZES.items():
        metrics_by_seed = {}
        for seed in SEEDS:
            observations = groups[arm, seed]
            if sorted(observations) != STEPS:
                raise ValueError(f"Observation grid differs for {arm}, seed {seed}")
            metrics_by_seed[seed] = endpoints([observations[step] for step in STEPS])
        for metric in METRICS:
            values = [metrics_by_seed[seed][metric] for seed in SEEDS]
            mean, sd = statistics.mean(values), statistics.stdev(values)
            rows.append(
                {
                    "configuration": arm,
                    "N": count,
                    "metric": metric,
                    **{f"seed{seed}_raw": value for seed, value in zip(SEEDS, values)},
                    "mean_raw": mean,
                    "sample_sd_raw": sd,
                    "mean_AP_points": 100 * mean,
                    "sample_sd_AP_points": 100 * sd,
                }
            )
    return rows


def compare(rows, reference):
    with reference.open(encoding="utf-8-sig", newline="") as handle:
        records = list(csv.DictReader(handle))
    by_key = {(row["configuration"], row["metric"]): row for row in records}
    if len(records) != len(rows) or len(by_key) != len(rows):
        raise ValueError("Expected one S2 row per configuration and metric")
    maximum = 0.0
    for row in rows:
        target = by_key[row["configuration"], row["metric"]]
        if int(target["N"]) != row["N"]:
            raise ValueError("Different membership count in S2")
        for field, value in row.items():
            if field in ("configuration", "N", "metric"):
                continue
            expected = float(target[field])
            if not math.isclose(value, expected, abs_tol=1e-12, rel_tol=0):
                raise ValueError(
                    f"{row['configuration']} {row['metric']} {field}: {value} vs {expected}"
                )
            maximum = max(maximum, abs(value - expected))
    print(
        f"Verified 12 endpoint rows and 84 numeric fields. Max absolute difference {maximum:.3g}.",
        file=sys.stderr,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", type=Path, help="Supplementary Data S1 (CSV)")
    parser.add_argument(
        "--verify", type=Path, help="Compare to published-format Supplementary Data S2"
    )
    args = parser.parse_args()
    rows = summarize(args.trajectory)
    if args.verify:
        compare(rows, args.verify)
    writer = csv.DictWriter(sys.stdout, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)


if __name__ == "__main__":
    main()
