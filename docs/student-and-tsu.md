# Student training, validation and temporal supervision utility

## How the entry points correspond to the paper

| Paper procedure | Entry point | Historical implementation and recorded output |
| --- | --- | --- |
| C1 student training with fixed members | `student/train_c1.py` | The R2_IDFIX release used `07_LAUNCH_SCRIPTS/run_one.py` to call the ordinary B0 loss in `02_CODE/p06_runner.py`. Each run used eight epochs, batch size 2 and seed 42, 43 or 44. The run recorded every validation observation in `checkpoint_metrics.csv` and updated the selected checkpoint only when validation Mask AP strictly exceeded its previous best. |
| C2 standard and boundary-relaxed configurations | `student/train_c2.py` | The six T31 runs used the T28 `fresh_panel_runner.py`. The standard configuration contained 6,857 training records and the boundary-relaxed configuration contained 8,009. Each run's `checkpoint_metrics.csv` contributed to Supplementary Data S1. |
| C2 fixed-size configuration | `student/train_fixed_n.py` | The final E2 archive used `COMMON_PROTOCOL/RUNNERS/e2_fixed_n_runner.py`. It retained 4,778 Final-A records and 927 original Final-B records, and added 1,152 prespecified recovered records. Each seed's `RAW_E2` log contributed 47 Fixed-N observations to S1. |
| Student model, loss and manual validation | `student/model.py`, `student/loss.py`, `student/validation.py`, `student/evaluate.py` | The model comes from the branch of `model_training_snapshot.build_model` without additional attention. The loss preserves the tensor operations of the B0 branch in `p06_loss`. Validation follows the original final4/final5/final5b routines and `P06Evaluator`. Use `evaluate.py` for training observations and `evaluate_table4.py` for the replay of the selected models in Table 4. |
| Four TSU measures | `tsu/summarize.py` | The per-run and aggregate calculations come from `analyze` in `SOURCE_METHODS/analyze_phase_a.py` and from `aggregate_e2.py` in the final E2 archive. For each of the three configurations, compute the measures separately for each seed's 47 observations, then report the mean and sample standard deviation across the three seeds, as in Supplementary Data S2. |

C2 uses the shared observation schedule in `student/eval_steps_47.json`. Each run stops at 27,422 successful optimizer updates. Each training record supplies an image and one target instance. The two-class Mask R-CNN student uses a ResNet-50-FPN backbone, COCO_V1 initialization, input size 512, batch size 2 and CUDA AMP. SGD uses learning rate 0.005, momentum 0.9 and weight decay 0.0005. MultiStepLR uses milestones 20 and 35 and γ = 0.1. The corrected validation reference contains 235 patches and 888 manually annotated instances. C1 fixed-member comparisons use an eight-epoch budget. Other historical comparisons follow their own training and checkpoint-selection protocols.

## Environments and installation

| Use | Recorded or verified environment | Dependencies |
| --- | --- | --- |
| C1 fixed-member, eight-epoch student training | Six Linux runs on an RTX 4090. Python 3.12.3, PyTorch 2.12.1+cu126, torchvision 0.27.1+cu126, NumPy 2.4.4, Pillow 12.2.0 and pycocotools 2.0.10. | [C1 GPU requirements](../configs/requirements-student-c1-gpu.txt) |
| C2 fixed-update training for three configurations | Linux, RTX 4090, Python 3.12.13, PyTorch 2.5.1+cu124, torchvision 0.20.1+cu124 and pycocotools 2.0.10. The T29 environment specification also pins NumPy 2.5.2 and Pillow 12.3.0. | [C2 GPU requirements](../configs/requirements-student-c2-gpu.txt) |
| Table 4 and Table 6 replay and synthetic checks | Windows, Python 3.12.14, NumPy 2.4.4, Pillow 12.2.0, pycocotools 2.0.11 and SciPy 1.18.0. Table 4 COCOeval requires NumPy and pycocotools. Table 6 also requires SciPy for Hungarian assignment. Matching comparisons, model initialization and synthetic inference were checked on CPU using PyTorch 2.12.1+cu126 and torchvision 0.27.1+cu126. | [CPU tool requirements](../configs/requirements-student-tools.txt) and optional torch installation below |
| TSU from S1 to S2 | Python standard library. | `python tsu/summarize.py --help` |

Use a separate environment with the indicated Python version for each use. GPU training requires compatible CUDA builds of PyTorch and torchvision and a matching NVIDIA driver. With a package index that supplies the recorded versions, install the relevant requirements file, for example `python -m pip install -r configs/requirements-student-tools.txt`. For offline installation, set `WHEELHOUSE` to a wheel directory for the target platform and Python version. Supply wheels for both direct and transitive dependencies. The requirements files list direct dependencies. The original archived environments provide additional version information for resolving historical dependencies.

```sh
# Run each section in its corresponding activated environment, from the repository root.
python -m pip install --no-index --find-links "$WHEELHOUSE" -r configs/requirements-student-c1-gpu.txt

python -m pip install --no-index --find-links "$WHEELHOUSE" -r configs/requirements-student-c2-gpu.txt

python -m pip install --no-index --find-links "$WHEELHOUSE" -r configs/requirements-student-tools.txt
python student/evaluate_table4.py --help

# Optional for model inference, final5b and the matching comparison on CPU.
python -m pip install --no-index --find-links "$WHEELHOUSE" "torch==2.12.1+cu126" "torchvision==0.27.1+cu126"
python student/smoke_matching.py
python student/smoke_cpu.py
```

The torch wheels used in the CPU checks included CUDA support. Other platforms and CPU-only builds require suitable wheels and have not been verified here. The COCO_V1 weight cache is required for historical student training and the optional initialization check. Table 4, Table 6, TSU and the ordinary synthetic CPU checks can run without pretrained weights.

## Inputs and training commands

Run commands from the repository root. The paths under `data`, `evaluation` and `runs` are examples. Supply authorized images, masks and manual validation annotations. Historical student training also requires the original torchvision COCO_V1 weights.

- `data/standard/meta/dataset_records.json` contains an array of records with `id`, `image_path`, `mask_path`, `patch_image` and `class_type`. Image and single-instance binary-mask paths may be relative to the dataset root or absolute on the local machine. Training boxes bound the foreground pixels, with 1 added to their right and lower limits.
- `data/standard/split/train.txt` and `data/standard/split/val.txt` are ordered lists of record IDs, one per line. Each C2 configuration has its own frozen lists. Both fixed-update training entry points check record identity and ordering. Paths may be relocated. Other archived record fields, the corrected manual reference and the observation schedule must correspond to the formal run. C1 can also train on the user's own data in this format.
- `evaluation/manual_gt_assets_final5b_repaired/normalized_gt_val_manual_coco.json` is the corrected manual COCO reference. Image objects contain `id`, `patch_stem`, `file_name` and image dimensions. Instance segmentations are RLEs. The adjacent `evaluation/manual_gt_assets/gt_val_samples.json` lists each sample's `id`, `patch_stem` and `file_name`. The optional `patch_image_map.json` maps patch stems to RGB paths, which can be relative to that directory. Include patches with zero instances in whole-set validation.
- Cache the original torchvision weights at `<TORCH_HOME>/hub/checkpoints/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth`. C1 training requires CUDA, a new output directory and verified COCO_V1 weights. C1 and C2 both check the weight file and stop if it is absent or does not match. The student loads the backbone weights and then initializes the two-class box and mask heads under the specified seed. The original fixed-member comparison used two 6,857-record lists with the same ordering. Each epoch shuffles using the training seed plus the epoch number.
- C1 can consume the output of the C1 supervision builder. Supply the manual reference separately, or use the synthetic example to test the interface. Historical C2 additionally requires all three prespecified configurations, complete records and ordered train/validation lists, 1,459 pseudo-validation records, the corrected manual reference and the original observation schedule. Supply each configuration in its own train/validation directory. The C1 writer omits `split/test.txt` when the input contains only training and validation sources. If the input contains a test source, it writes that file even when every test mask is rejected. Both C2 training entry points reject any package containing `split/test.txt`, including an empty file. Prepare C2 inputs from the original train/validation-only sources and preserve any test marker in historical or mixed packages. The C2 checks cover member order, scientific record fields, the reference and observation coordinates. Supply the original image and mask versions separately. Synthetic examples and a newly generated C1 set cannot substitute for the three archived C2 configurations.

```sh
export TORCH_HOME=/path/to/your/torch-cache
WEIGHT="$TORCH_HOME/hub/checkpoints/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth"
MANUAL=evaluation/manual_gt_assets_final5b_repaired
python student/train_c1.py --dataset data/standard --manual-root "$MANUAL" --output runs/c1_standard_42 --seed 42
python student/train_c2.py --arm B0 --dataset data/standard --manual-root "$MANUAL" --init-weight "$WEIGHT" --output runs/c2_standard_42 --seed 42
python student/train_c2.py --arm RECOVERED_C --dataset data/boundary_relaxed --manual-root "$MANUAL" --init-weight "$WEIGHT" --output runs/c2_relaxed_42 --seed 42
python student/train_fixed_n.py --arm FIXED_N --dataset data/fixed_n --manual-root "$MANUAL" --init-weight "$WEIGHT" --output runs/c2_fixed_42 --seed 42
python student/evaluate.py --gt "$MANUAL/normalized_gt_val_manual_coco.json" --predictions saved_segm_predictions.json
python student/evaluate_table4.py --gt "$MANUAL/normalized_gt_val_manual_coco.json" --predictions saved_segm_predictions.json --boxes saved_bbox_predictions.json --inventory
python tsu/summarize.py Supplementary_Data_S1_Trajectories.csv --verify Supplementary_Data_S2_Endpoints.csv
```

## Evaluation during training and for Tables 4 and 6

**Training observations.** The final5b evaluator matches masks and integrates precision using NumPy. At each IoU threshold, it processes predictions in descending score order, assigning the highest-IoU unmatched ground truth in the same patch. A duplicate with no remaining qualifying match is a false positive. Evaluation uses ten IoU thresholds from 0.50 through 0.95 and 101 recall thresholds. It retains up to 100 predictions per image with score ≥0.05 and binarizes predicted masks at 0.5. Training reads separate inputs for pseudo-validation loss and manual-reference checkpoint selection. `student/evaluate.py` replays saved segmentation predictions and reports final5b mask AP on a 0–1 scale.

**Table 4, replay of selected models.** `student/evaluate_table4.py` derives from `ap` in the original October 20 `evaluate_saved_predictions.py`. It evaluates selected predictions on the corrected 235-patch, 888-instance reference using official pycocotools COCOeval, original scores, per-image and per-category grouping and `maxDets=[1,10,100]`. For mask AP, it removes the predicted bounding box before calling `COCO.loadRes`, so the mask supplies the area. Results are AP points on a 0–100 scale. Differences from final5b include equal-IoU matches, crowd/ignore handling and area-specific evaluation. All 888 instances in this reference share one category and have no crowd/ignore flags.

**Table 6, patch-level inventory.** Pass `--inventory` to `student/evaluate_table4.py`. At the primary confidence threshold of 0.50 (inclusive), SciPy's `linear_sum_assignment` maximizes total mask IoU between predictions and manual instances within each patch. Then retain assigned pairs with IoU ≥0.50. Precision, matching coverage and F1 use TP, FP and FN pooled over patches. Count MAE is the mean absolute difference between predicted and annotated counts per patch. Mean relative area error uses the absolute difference in mask areas divided by the ground-truth area (in pixels, denominator at least 1), averaged over retained matches. The historical export writes zero when there are no matches and also reports `matched_area_n`. Training's `threshold_metrics.py` uses confidence-ordered greedy matching and records a separate diagnostic based on total area per patch.

To export one run or reproduce the three-seed inventory summary, use:

```sh
python student/evaluate_table4.py --gt validation.json --predictions seed42_masks.json --inventory --method CrownRoute --seed 42 --per-image-csv crownroute_seed42.csv
python student/agricultural_metrics.py --gt validation.json --runs runs.csv --output-dir inventory-report
```

The run manifest is a CSV with columns `method,seed,predictions`. Relative prediction paths are resolved from the manifest directory. Provide seeds 42, 43 and 44 for each method and choose a new output directory. The batch command writes `per_image.csv`, `per_seed.csv` and `three_seed_summary.csv`. Its patch records carry the image ID and source, ground-truth and predicted counts, TP/FP/FN, absolute and signed count error, and the sum and count used to aggregate matched relative area error. Patches with zero annotations or zero predictions are retained. The aggregate reports each method's mean and sample standard deviation across seeds. To reproduce confidence sensitivity, pass `--threshold 0.30` or `--threshold 0.70`. The mask IoU cutoff remains 0.50.

Table 6 and Supplementary Table S5 use the primary threshold. The sensitivity results at 0.30 and 0.70 are reported in the accompanying Supplement S11 text. The nine archived runs were replayed at all three confidence thresholds. At 0.50, image IDs and sources matched the historical export for all 2,115 patch records, and the largest absolute difference across the 19,035 patch-level numeric entries checked was 2.22 × 10⁻¹⁶. The single-run and batch interfaces produced identical results for the 235 patches in the single-run comparison.

Run `python student/smoke_matching.py` for two synthetic regression cases. With two ground-truth crowns and a duplicate prediction for the first, both evaluators return 83.498350 AP50 points. When the first prediction has exactly 0.50 IoU with both crowns, official COCOeval gives 100.000000 AP50 and final5b gives 50.495050. In the same two-prediction inventory example, Hungarian assignment yields two valid matches and the training diagnostic's greedy matcher yields one.

## TSU calculation and verification

Peak is the highest observed Mask AP in a run. Top-5 averages its five highest observations. nAUC integrates the actual observation coordinates using the trapezoidal rule and divides by the span from the first to the last observation. Late averages observations at successful-update progress ≥0.75. Input `mask_ap` uses the 0–1 scale. S2 reports both proportions and AP points. Recomputing the 423 observations from nine runs reproduced all 36 per-seed endpoint values, all 24 means and sample standard deviations on the proportion scale, and all 24 on the AP-point scale.

The CPU smoke check reads synthetic records using relative paths, constructs training boxes and loads final5b validation inputs. Perfect and empty predictions yielded Mask AP of 1 and 0. The student completed one inference pass without pretrained weights, and the ROI mask loss matched the archived implementation element by element. No new scientific training runs were needed for these checks. The initial check used Python 3.12.14, torch 2.12.0+cu126, torchvision 0.27.0+cu126 and pycocotools 2.0.11. The subsequent full matching comparison, COCO_V1 initialization and CPU checks used the CPU environment in the table above.

For Tables 4 and 6, the nine archived prediction sets for three selected methods and three seeds were replayed on the corrected reference. Across nine sets of 15 historical per-seed quantities, the maximum absolute difference was 1.4 × 10⁻¹⁷. For Final-A-only, CrownRoute and BoxInst, respectively, Table 4 Mask AP was 42.70 ± 0.92, 43.57 ± 0.33 and 44.57 ± 0.20 points. For standard CrownRoute, Table 6 F1 was 70.0170% and Count MAE was 1.42837.

The repository retains the ordinary loss and the model and evaluation paths used for the paper. It omits unused experimental options, machine paths and historical entry points. Some historical C1 logs recorded a B weight of 0.7, although B0 did not use that field. The current C1 output records the effective weight, 1.0. Model initialization, loss, sampling, optimization and evaluation follow the historical calculations. The C2 checks compare scientific record fields separately from relocatable machine paths. With the original COCO_V1 cache available, run `python student/smoke_cpu.py --check-initialization`. A checked backbone parameter matched the reference weights exactly. The two-class box and mask heads initialized identically on repeated runs with seed 42 and both changed with seed 43.

The original images, labels, predictions, member lists, weights and dependencies are managed by their respective providers. Third-party software and weights are subject to their own terms. Rights holders must confirm authorization for release of the project source. See [distribution and data access](distribution.md).
