# PMP-DACIS

Does the choice of channels matter when a ResNet-18 is pruned for few-shot Earth observation? This repository compares
eight channel scores and seven training schedules on aerial scenes (NWPU-RESISC45), 13-band Sentinel-2 patches
(EuroSAT-MS) and crop photographs (PlantVillage, PlantDoc), and ships a one-click notebook that reruns every
experiment and rebuilds the paper tables, figures and PDFs from the saved results.

[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/Mudassiruddin7/PMP-DACIS-CV4EO/blob/main/notebooks/pmp_dacis_cv4eo.py)

## Reproduce in one click

1. Open `notebooks/pmp_dacis_cv4eo.py` in [molab](https://molab.marimo.io) (button above) on a GPU machine.
2. Press **Run all**. Nothing has to be selected or typed.
3. When the last cell finishes, press **Results**. You get `results.zip` with every run (JSON and per-episode arrays),
   all tables and figures, the logs, and the compiled PDFs (`paper_review.pdf`, anonymous, `paper_final.pdf`, with
   authors, and `paper_supp.pdf`).

The notebook downloads the four datasets from the Hugging Face Hub (about 5 GB) and trains about 130 models per class
partition. On an RTX 6000-class GPU the whole run takes about two hours. Finished runs are written to disk, so *Run all*
can be repeated after an interruption and continues where it stopped. Without a GPU it falls back to a tiny `smoke`
profile that only checks the code path; those numbers are not results. It also runs as a plain script:
`python notebooks/pmp_dacis_cv4eo.py`.

The library lives in `pmp_dacis/` (readable sources; the notebook embeds a copy so that one file is enough).
`python tests/test_core.py` checks the pruning surgery, the budget solver, the episode sampler, the run cache and the
dataset variants.

## What is compared

* **Where to cut.** Output channels of the first convolution of each ResNet-18 basic block (1,920 channels). Block widths
  of the residual stream never change. The budget is the share of the 11.18 M trunk parameters that remain.
* **Which channels to keep.** Random, filter l1 norm, BatchNorm scale, first-order Taylor, the three DACIS ingredients
  alone, and DACIS (gradient norm, activation variance and Fisher ratio, each as a within-block rank, weighted
  0.3 / 0.2 / 0.5).
* **When to prune.** P (prune once, then recover), P-M, M-P, P-G-P (gradient probe), P-M-P (the PMP schedule),
  P-M-P-M, and P-M with twice the episodes; budgets of 50, 30, 20 and 10 % of the parameters.
* **Baselines.** Dense ResNet-18 with and without the episodic stage; MobileNetV3-Small, MobileNetV2 and EfficientNet-B0
  with the same training recipe.
* **Where to test.** 5-way 1/5/10-shot episodes on classes held out from training: 15 of 45 RESISC45 scene classes,
  5 of 10 EuroSAT land-cover classes with all 13 Sentinel-2 bands (and a three-band control), and 8 of 38 PlantVillage
  classes; PlantDoc field photographs are a domain-shift test. Five random class partitions, and the same 2,000 test
  episodes for every method inside a partition.

## Results

<!-- RESULTS:BEGIN -->
Run on NVIDIA RTX PRO 6000 Blackwell Server Edition (unknown); 5 random class partitions; episodic learning rate 0.0001; mean ± standard deviation over partitions, 5-way 5-shot test episodes, identical for every row.

| model | params kept | RESISC45 | EuroSAT-MS | PlantVillage | PlantDoc |
|---|---|---|---|---|---|
| ResNet-18 dense (S0) | 100% | 86.9 ± 3.4 | 62.1 ± 5.3 | 91.4 ± 5.6 | 55.4 ± 1.3 |
| ResNet-18 dense (S0+R+M+R) | 100% | 86.7 ± 3.3 | 61.7 ± 5.1 | 91.3 ± 5.6 | 55.5 ± 1.1 |
| MobileNetV3-Small | 8% | 88.1 ± 3.3 | 72.4 ± 4.9 | 91.1 ± 5.9 | 50.0 ± 0.8 |
| MobileNetV2 | 20% | 86.9 ± 3.1 | 65.4 ± 3.5 | 91.2 ± 6.3 | 55.1 ± 1.0 |
| EfficientNet-B0 | 36% | 90.5 ± 2.4 | 69.9 ± 5.1 | 93.8 ± 5.2 | 58.5 ± 0.4 |
| 30%, random, prune+recover (P) | 30% | 79.7 ± 4.9 | 55.6 ± 6.4 | 85.6 ± 8.1 | 46.6 ± 1.3 |
| 30%, l1, prune+recover (P) | 30% | 80.5 ± 4.5 | 56.2 ± 7.9 | 85.8 ± 8.3 | 48.7 ± 0.9 |
| 30%, bn_gamma, prune+recover (P) | 30% | -- | -- | 85.1 ± 8.9 | 44.4 ± 1.7 |
| 30%, taylor, prune+recover (P) | 30% | 82.2 ± 3.9 | 58.1 ± 6.6 | 88.3 ± 7.1 | 47.9 ± 1.2 |
| 30%, G, prune+recover (P) | 30% | -- | -- | 87.3 ± 7.2 | 47.6 ± 1.5 |
| 30%, V, prune+recover (P) | 30% | -- | -- | 86.5 ± 6.9 | 47.2 ± 1.4 |
| 30%, D, prune+recover (P) | 30% | 79.3 ± 4.1 | 55.8 ± 6.9 | 85.2 ± 8.9 | 46.8 ± 0.8 |
| 30%, dacis, prune+recover (P) | 30% | 79.7 ± 4.1 | 56.3 ± 6.7 | 86.4 ± 7.7 | 47.3 ± 1.4 |
| 30%, random, PMP | 30% | 81.0 ± 4.2 | 57.6 ± 6.0 | 86.8 ± 7.7 | 46.7 ± 1.7 |
| 30%, l1, PMP | 30% | 81.0 ± 4.2 | 57.0 ± 6.0 | 86.4 ± 7.7 | 48.1 ± 0.8 |
| 30%, bn_gamma, PMP | 30% | 79.7 ± 4.2 | 55.0 ± 6.1 | 85.1 ± 8.7 | 45.5 ± 1.0 |
| 30%, taylor, PMP | 30% | 82.2 ± 3.9 | 57.0 ± 8.0 | 87.1 ± 7.1 | 48.5 ± 1.0 |
| 30%, G, PMP | 30% | 81.0 ± 4.5 | 56.4 ± 6.5 | 86.7 ± 7.9 | 46.9 ± 2.5 |
| 30%, V, PMP | 30% | 81.0 ± 4.0 | 56.5 ± 5.2 | 86.5 ± 7.3 | 47.3 ± 1.3 |
| 30%, D, PMP | 30% | 79.5 ± 3.9 | 56.0 ± 7.5 | 85.3 ± 8.0 | 46.4 ± 1.3 |
| 30%, dacis, PMP | 30% | 80.5 ± 4.2 | 57.1 ± 4.9 | 86.6 ± 7.5 | 47.5 ± 1.6 |
| 50%, l1, PMP | 50% | 84.1 ± 3.7 | 60.1 ± 4.4 | 89.0 ± 7.1 | 51.3 ± 1.1 |
| 50%, dacis, PMP | 50% | 83.8 ± 3.7 | 58.6 ± 5.7 | 88.6 ± 7.2 | 51.6 ± 1.4 |
| 20%, l1, PMP | 20% | 79.4 ± 4.4 | 54.6 ± 8.7 | 84.2 ± 8.3 | 45.3 ± 1.0 |
| 20%, dacis, PMP | 20% | 77.4 ± 4.9 | 54.5 ± 8.1 | 83.8 ± 8.2 | 44.1 ± 2.0 |
| 10%, l1, PMP | 10% | 70.8 ± 5.5 | 50.9 ± 10.4 | 79.3 ± 9.9 | 39.1 ± 1.1 |
| 10%, dacis, PMP | 10% | 71.9 ± 4.8 | 51.4 ± 7.5 | 78.2 ± 10.2 | 40.5 ± 1.1 |
<!-- RESULTS:END -->

## About earlier numbers

Earlier versions of this README, and the first arXiv versions of the paper, reported different results. The scripts that
produced them had no working dataset loader (they fell back to random tensors), so those numbers cannot be regenerated
and should not be cited. The scripts were removed from the working tree; they remain in the git history before commit
`eba0a99`. Everything here was rebuilt with real data loading, a shared pruning budget solver and paired test episodes.

## Layout

```
pmp_dacis/        library: data, ResNet-18 surgery, scores, training stages, evaluation, report, paper build
notebooks/        pmp_dacis_cv4eo.py, the single-file marimo notebook
paper/            LaTeX sources (WACV 2027 style) and the built PDFs
results/          saved runs of the reported experiment, generated tables and figures
tests/            unit tests for the parts where a silent bug would change the science
tools/            notebook builder, result finaliser, anonymity checker
```

## License

MIT, see `LICENSE`.
