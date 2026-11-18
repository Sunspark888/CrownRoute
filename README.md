# CrownRoute

[简体中文](README_zh.md)

CrownRoute constructs instance masks for training a tree-crown segmenter from UAV RGB patches and weak bounding boxes. FastSAM-s generates one candidate pool per patch. Each box is associated with a candidate independently. Route I refines a reliable candidate (Final-A). Route II reconstructs the foreground from pixels inside the box and accepts eligible records using the original association features (Final-B). The standard training set contains accepted A and B records. C2 compares supervision configurations using the validation trajectories of separately trained students.

## Build supervision on a CPU

From the repository root, use a Python 3.12 environment and two fresh directory names. This example generates synthetic image patches, boxes and candidate masks locally.

```sh
python -m pip install -r configs/requirements-c1.txt
python -m scripts.make_c1_demo demo-input
python -m scripts.build_supervision --manifest demo-input/patches.json --output demo-output
python -m unittest discover -s tests -p test_c1.py
```

`demo-output/meta/route_outcomes.json` gives the outcome for each of five boxes. Two records enter the training split, one through each accepted route. To use your own images and weak boxes, supply a manifest and either a local FastSAM-s checkpoint or saved candidate masks. See the [input format, commands and construction sources](docs/c1-sources.md).

## Train and evaluate the student

| Paper analysis | Command to inspect from the repository root | Data or output |
| --- | --- | --- |
| C1 fixed-membership comparison | `python student/train_c1.py --help` | Eight epochs, original COCO initialization and a fixed validation reference |
| C2 standard and expanded configurations | `python student/train_c2.py --help` | Frozen membership and 47 validation observations over 27,422 successful updates |
| C2 fixed-size replacement | `python student/train_fixed_n.py --help` | Matched record count and the same successful-update budget |
| Table 4 model comparison | `python student/evaluate_table4.py --help` | Official COCOeval on the saved predictions of each previously selected model |
| Table 6 orchard inventory | `python student/agricultural_metrics.py --help` for three-seed reports, or add `--inventory` to the Table 4 command for one run | Maximize total mask IoU per patch, then retain pairs with IoU ≥ 0.50. The primary confidence threshold is 0.50 |
| Training-time validation | `python student/evaluate.py --help` | Replay the project's final5b validation metric |

Exact C2 reproduction requires authorized access to the original images, frozen record fields and ordered membership, the corrected manual reference and the specified initialization. See [C2 configuration assembly](docs/c2-configurations.md) and [student environments, input formats and evaluation](docs/student-and-tsu.md). The latter distinguishes the two historical GPU environments from the CPU tools used for code checks.

With the student tools installed, `python student/smoke_cpu.py` checks data loading, a synthetic forward pass and perfect/empty predictions. `python student/smoke_matching.py` checks duplicate detections and an ambiguous two-crown example. These CPU checks exercise components; the paper's training results come from the archived runs.

## Recompute the TSU endpoints

Place the paper's supplementary CSVs in the current directory and run:

```sh
python tsu/summarize.py Supplementary_Data_S1_Trajectories.csv --verify Supplementary_Data_S2_Endpoints.csv
```

This command uses only the Python standard library. It calculates Peak, Top-5, normalized area under the trajectory, and Late for each run before summarizing the three seeds.

## Availability

This repository is a private review version. The research images, original annotations, complete supervision sets and model weights are held separately. The synthetic example can be used without access to those assets. See the [distribution scope and third-party terms](docs/distribution.md).
