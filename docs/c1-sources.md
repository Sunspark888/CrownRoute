# C1 supervision construction

FastSAM-s generates a candidate pool once per RGB patch. Each weak crown box selects a candidate independently from this pool. The student receives one record per accepted box. The implementation follows the original OpenCV BGR decoding and color calculations.

| Stage | Original project source | Repository entry |
| --- | --- | --- |
| Generation, association and refinement | `step10_2_fastsam_refiner_full.py`, including `get_candidate_masks_from_fastsam`, `choose_best_mask_with_multiscore` and `refine_mask_first_version` | `crownroute/association.py` and `pipeline.generate_candidates` |
| Route I priority and route II eligibility | `step10_3_classify_instances_full.py::classify_instance` | `crownroute/eligibility.py`. The archived association summaries have no `gt_coverage` or `area_score`; the two legacy conditions using those fields are inactive. |
| Reconstruction from RGB and box | `step10_5_hole_recovery_full.py`: `build_initial_mask`, `contrast_aware_hole_recovery`, then the crop-to-patch transformation | `crownroute/reconstruction.py` |
| Branch mask selection and common normalization | `step10_6_build_final_dataset_full.py` selected the refined mask for A and the reconstructed mask for B. `step10_8_build_final_dataset_v2_ab_holefill.py` applied `step10_7_ab_holefill_preview_300.py::repair_mask` to both. | `crownroute/normalization.py` and `pipeline.construct_record` |
| Fixed admission and assembly | `build_proposed_final_variants_20260628.py::enrich/frozen_rules/materialize`, `frozen_rule_manifest.json`, and the `proposed_v3_balanced` training inventory | `pipeline.admission_score`, `configs/c1.json` and `scripts/build_supervision.py` |

Route I requires s ≥ 0.75, u ≥ 0.60 and r ≥ 0.75. Among remaining records, 0.45 ≤ s < 0.80 and g ≥ 0.50 qualifies for route II. The route II admission score uses features saved at candidate association: q = 0.40s + 0.30u + 0.20r + 0.10g. The fixed cutoff is 0.5401658842336331. The standard inventory consists of normalized refined A masks and accepted normalized B masks. The historical pipeline computed a refinement intermediate for every matched candidate before routing; the portable entry computes that intermediate for the records that use it.

The frozen cutoff used an 80% global coverage target and the distribution of validation candidate scores. The global routing counts included counts from the test partition. The assembly report records reconstructed masks for all 3,718 eligible records. When new inputs have no valid foreground component in route II, the historical reconstruction function returns no mask. The portable entry records this separately as `unconstructed`, without adding a training record.

## Saved-candidate example

From the repository root with Python 3.10+, NumPy and OpenCV:

```bash
python -m scripts.make_c1_demo demo_inputs
python -m scripts.build_supervision --manifest demo_inputs/patches.json --output demo_standard
python -m unittest discover -s tests -p "test_c1.py" -v
```

The five synthetic cases cover refinement, accepted reconstruction, rejection by the fixed admission rule, ineligibility and absence of a valid candidate. The demo supplies ordered binary candidate PNGs directly. It uses neither model weights nor research images.

## FastSAM-s input

Actual generation additionally needs the optional dependency listed in `requirements-fastsam.txt`, a compatible PyTorch environment and separately obtained FastSAM-s weights. The package version listed there was used for the integration check. The default generator settings are in `configs/c1.json`.

```bash
python -m scripts.build_supervision --manifest input/patches.json --output standard --weights /path/to/FastSAM-s.pt --device cpu
```

Example input manifest:

```json
{"patches":[{"id":"sample","image":"sample.png","split":"train","source_id":"source_1","boxes":[[20,20,76,76]]}]}
```

Image and candidate paths are relative to the manifest. Boxes are integer xyxy coordinates with exclusive right/bottom boundaries. Preserve source splits, the order of boxes within each patch and the order of candidates. To use saved masks, add `candidate_masks` (a list of same-size binary PNG paths) to each patch and omit `--weights`.

The output includes `meta/dataset_records.json`, `split/train.txt`, `split/val.txt`, patch images, single-instance masks and `meta/route_outcomes.json`. Asset paths are relative to the dataset root. Whenever the input contains a test source, `split/test.txt` is also written, even if empty because all test instances were rejected. C2 accepts training/validation-only inputs.

## Checks

The historical training inventory contains 6,857 unique IDs: 4,778 A and 2,079 B. All admitted B records meet the frozen cutoff. Replaying the first four A and first four B in training-list order, starting from their saved associated candidates, reproduced all eight routes and every mask pixel. The original candidate association features also matched. Research data and weights are held separately.

A prior real FastSAM-s CPU check used one synthetic patch with two boxes. A single image-level inference produced a candidate pool for both associations and both records were rejected at admission. The saved-candidate tests cover the remaining outcomes. The integration environment was Python 3.12.14, NumPy 2.4.4, OpenCV 4.13.0 and Ultralytics 8.4.75.
