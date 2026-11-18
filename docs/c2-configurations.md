# C2 supervision configurations

Each configuration names a fixed training population. The three training seeds (42, 43 and 44) share the same IDs and order. The student uses 1,459 common pseudo-supervision validation records; the 235-patch, 888-instance manual reference for performance evaluation is a separate input.

| Configuration | Final-A | Final-B | Recovered | Training records |
| --- | ---: | ---: | ---: | ---: |
| Standard (`B0`) | 4,778 | 2,079 | 0 | 6,857 |
| Boundary-relaxed (`RECOVERED_C`) | 4,778 | 2,079 | 1,152 | 8,009 |
| Fixed-size substitution (`FIXED_N`) | 4,778 | 927 | 1,152 | 6,857 |

## Recovery population

The source population was the training partition's **original intermediate C**, as classified before the route II admission gate. Thus the recovery inventory omits intermediate B records rejected later by that gate. The original builder sorted these C records by instance ID and used the frozen RGB/box recovery and normalization functions. It accepted every nonempty successful result. Of 1,153 eligible training-side records, 1,152 yielded masks. One had no initial component. Neither the 60-record geometry probe nor a validation IoU threshold selected these members.

The portable `crownroute/recovery.py` preserves the original `run_frozen_recovery` operator from `build_ab_recovered_c_dataset.py`. It calls the same functions from `step10_5_hole_recovery_full.py` and `step10_7_ab_holefill_preview_300.py` as the historical delta. The optional entry below requires an authorized intermediate-routing JSON list, a source-split CSV (`source_id,split`) and a directory of RGB patches. Image names follow the original `<source>_patch_<x>_<y>` convention. Input records contain the pre-gate `class_type`, `patch_image`, zero-based `gt_index`, `gt_box` and saved association features. Only training-side original C images are opened.

```bash
python -m scripts.recover_training_delta --classified data/classified_instances.json --source-split data/master_source_split.csv --images data/patches --output recovered_delta
```

The entry writes ordered `records.json`, per-attempt `outcomes.json`, and the images and masks needed by those records. It is also possible to start from an authorized frozen recovery inventory.

## Data assembly and reuse

The original `build_ab_recovered_c_dataset.py` appended successful recoveries to the ordered standard training list. Validation records were carried over. For fixed-size substitution, `E2_FIXED_N_COMPOSITION_CONTROL/build_fixed_n.py` removed one Final-B per added recovery. Within each source the removal quota is the smaller of its B and recovered counts. Remaining removals are apportioned to the remaining B pool by Hamilton's largest-remainder method, breaking equal remainders by lexical source ID. Within sorted source groups, a single `random.Random(20260927)` shuffles the sorted B IDs. Retained standard IDs retain their original order, followed by recovered IDs in their original order. The archived freeze has 1,026 within-source and 126 proportional substitutions.

Use either the generated recovery list or the frozen expanded dataset:

```bash
python -m scripts.build_c2_configs --standard data/B0 --recovered-records recovered_delta/records.json --base-assets data/B0 --recovery-assets recovered_delta --output prepared
python -m scripts.build_c2_configs --standard data/B0 --expanded data/RECOVERED_C --base-assets base_package --recovery-assets delta_package --fixed-order frozen_fixed/split/train.txt --output prepared_from_archive
```

The standard and expanded inputs each contain `meta/dataset_records.json`, `split/train.txt` and `split/val.txt`. The asset arguments identify the base directories for the relative image/mask filenames in these manifests. A `split/test.txt` file makes the input ineligible, even if the file is empty. `--fixed-order` checks the exact recomputed IDs against a previously frozen training list. No training metric enters membership selection. The three output directories contain metadata views pointing to existing image/mask files. Keep the supplied assets in place. Original image and mask bytes are shared across configurations.

The formal training entry verifies the archived identities of all split orders and scientific record fields. A synthetic or newly assembled dataset may illustrate the construction without satisfying those historical identities.

## Verification

```bash
python -m unittest discover -s tests -p "test_c2_configs.py" -v
```

The CPU test covers a successful recovery, an empty reconstruction, skipping a validation image, within-source and proportional substitution, and reading the emitted masks. On the original private inventories, all three materialized datasets matched the formal student's pre-existing checks for training order, validation order and scientific record fields/order. Replaying the first successful archived recovery and the original failed case reproduced their statuses; the successful case also reproduced every final mask pixel and its scientific metadata. The repository contains the procedure and synthetic test. Historical membership lists, masks, images, score logs and manual annotations remain outside it.
