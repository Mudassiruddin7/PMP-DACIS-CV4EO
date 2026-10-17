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
The numbers of the reported run are added here, generated from `results/` by `pmp_dacis/report.py` (no number in this
repository is typed by hand).
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
