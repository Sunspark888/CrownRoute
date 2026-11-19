# Supplement S11: paired source-cluster bootstrap

Use the primary-threshold `per_image.csv` produced by [the Table 6 inventory evaluator](student-and-tsu.md). Each patch must carry the identifier of its original UAV image in the `source` column. Each compared method must cover the same validation patches with training seeds 42, 43 and 44. Supply the method names as they appear in your input table.

```bash
python student/bootstrap_sources.py --per-image inventory/per_image.csv --compare CrownRoute Final-A-only --compare CrownRoute BoxInst --output s11_intervals.csv
```

The command uses NumPy and writes one interval per metric and contrast. It draws as many original images as there are distinct sources, with replacement, 2,000 times (random seed 20261020). Every occurrence contributes all patches from that source. Each method and training seed uses the same draw. The paper's validation set had 235 patches from 77 sources.

Within each draw, count MAE is the mean absolute count error per patch. F1 is computed from pooled patch-level TP, FP and FN counts. Each metric is computed per training seed and averaged over the three seeds before taking the left-minus-right difference. The CSV reports the 2.5th and 97.5th percentiles. Count MAE differences are in crowns per patch and F1 differences are in percentage points. The intervals describe validation-source sampling uncertainty conditional on the selected trained models.
