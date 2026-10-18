# Channel importance scores for few-shot Earth observation

Which channels do you keep when a pretrained network is pruned to recognise classes it never trained on, from five labelled examples?
This repository compares eight channel scores (random, filter L1, BatchNorm scale, Taylor, three single statistics and DACIS) and seven
training schedules on a ResNet-18, and five of the scores on a MobileNetV2, on classes held out from aerial scenes (NWPU-RESISC45) and
13-band Sentinel-2 patches (EuroSAT-MS). PlantVillage and PlantDoc leaf photographs are a ground-level control. Every method answers the
same test episodes on ten random class partitions (five for the leaf sets), so the comparisons are paired.

The paper is under double-blind review at the 4th CV4EO workshop at WACV 2027. Authors and the paper are added here after the decision.

**Project page:** <https://mudassiruddin7.github.io/benchmarking-channel-pruning-few-shot-eo/>

Random channel choice held up against nearly every clever score. Only one beat it, and a network nobody pruned beat them all.

| 5-shot accuracy on held-out classes, RESISC45 | |
|---|---|
| Taylor score against random choice, ResNet-18 kept at 30% of its weights | +1.4 points, ahead in 10 of 10 partitions |
| The same at 10% of the weights (5 partitions) | +6.0 points |
| The same on a MobileNetV2 kept at 25% | +6.5 points |
| BatchNorm-scale score against random choice, MobileNetV2 at 25% | -8.3 points |
| DACIS against random choice, ResNet-18 at 30% | 0.0 points (95% interval -0.8 to +0.7) |
| Multiply-accumulates removed, then CPU time removed, at 30% | 58%, then 39% |
| MobileNetV3-Small, never pruned | 88.1% (the best pruned network we built: 85.9%) |

The method name PMP-DACIS comes from an earlier preprint. The numbers of that preprint do not reproduce under this protocol (see
*About earlier numbers*).

## Reproduce in one click

1. Open `notebooks/pmp_dacis_cv4eo.py` with marimo (`marimo edit`) on a GPU machine.
2. Press **Run all**. Nothing has to be selected or typed.
3. When the last cell finishes, press **Results**. You get `results.zip` with every run (JSON and per-episode arrays),
   all tables and figures, the logs, and the compiled PDFs of the anonymous review version and its supplement.

The notebook downloads the four datasets from the Hugging Face Hub (about 5 GB) and trains about 130 models per class
partition. On an RTX 6000-class GPU the whole run takes about two hours. Finished runs are written to disk, so *Run all*
can be repeated after an interruption and continues where it stopped. Without a GPU it falls back to a tiny `smoke`
profile that only checks the code path; those numbers are not results. It also runs as a plain script:
`python notebooks/pmp_dacis_cv4eo.py`.

A second notebook, `notebooks/pmp_dacis_cv4eo_ext.py`, adds class partitions 5 to 9 for the RESISC45 and EuroSAT-MS
comparisons and prunes MobileNetV2 with the same scores (about 1.5 hours on an RTX 6000-class GPU). The reported numbers combine
both runs; `results/` holds the merged saved runs.

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
  classes; PlantDoc field photographs are a domain-shift test. Five random class partitions (ten for the two
  Earth-observation sets, see the extension notebook), and the same 2,000 test episodes for every method inside a partition.
* **A compact network.** MobileNetV2 pruned with the same scores (random, L1, BatchNorm scale, Taylor, DACIS) to 60, 40 and
  25 % of its parameters (`MP_*` runs).

## Results

Run on NVIDIA RTX PRO 6000 Blackwell Server Edition; 10 random class partitions for RESISC45 and EuroSAT-MS (5 for PlantVillage; rows for 50, 20 and 10 % of the parameters use 5); episodic learning rate 0.0001; mean ± standard deviation over partitions, 5-way 5-shot test episodes, identical for every row. Full tables: `paper/generated/summary.md`.

| model | params kept | RESISC45 | EuroSAT-MS | PlantVillage | PlantDoc |
|---|---|---|---|---|---|
| ResNet-18 dense | 100% | 87.2 ± 2.5 | 64.7 ± 4.9 | 91.4 ± 5.6 | 55.4 ± 1.3 |
| MobileNetV3-Small dense | 8% | 88.1 ± 2.5 | 74.7 ± 6.0 | 91.1 ± 5.9 | 50.0 ± 0.8 |
| MobileNetV2 dense | 20% | 87.4 ± 2.5 | 67.8 ± 6.4 | 91.2 ± 6.3 | 55.1 ± 1.0 |
| EfficientNet-B0 dense | 36% | 90.7 ± 2.0 | 73.6 ± 6.9 | 93.8 ± 5.2 | 58.5 ± 0.4 |
| ResNet-18, 30%, random, PMP | 30% | 81.1 ± 3.0 | 58.4 ± 4.6 | 86.8 ± 7.7 | 46.7 ± 1.7 |
| ResNet-18, 30%, filter L1, PMP | 30% | 81.5 ± 3.0 | 59.4 ± 5.7 | 86.4 ± 7.7 | 48.1 ± 0.8 |
| ResNet-18, 30%, BN scale, PMP | 30% | 80.0 ± 3.1 | 56.8 ± 4.9 | 85.1 ± 8.7 | 45.5 ± 1.0 |
| ResNet-18, 30%, Taylor, PMP | 30% | 82.4 ± 3.0 | 59.7 ± 6.3 | 87.1 ± 7.1 | 48.5 ± 1.0 |
| ResNet-18, 30%, gradient, PMP | 30% | 81.4 ± 3.3 | 58.9 ± 5.6 | 86.7 ± 7.9 | 46.9 ± 2.5 |
| ResNet-18, 30%, variance, PMP | 30% | 81.5 ± 3.0 | 57.3 ± 4.4 | 86.5 ± 7.3 | 47.3 ± 1.3 |
| ResNet-18, 30%, Fisher, PMP | 30% | 79.9 ± 2.9 | 58.4 ± 6.0 | 85.3 ± 8.0 | 46.4 ± 1.3 |
| ResNet-18, 30%, DACIS, PMP | 30% | 81.0 ± 3.3 | 59.3 ± 4.9 | 86.6 ± 7.5 | 47.5 ± 1.6 |
| ResNet-18, 50%, filter L1, PMP | 50% | 84.1 ± 3.7 | 60.1 ± 4.4 | 89.0 ± 7.1 | 51.3 ± 1.1 |
| ResNet-18, 50%, DACIS, PMP | 50% | 83.8 ± 3.7 | 58.6 ± 5.7 | 88.6 ± 7.2 | 51.6 ± 1.4 |
| ResNet-18, 20%, filter L1, PMP | 20% | 79.4 ± 4.4 | 54.6 ± 8.7 | 84.2 ± 8.3 | 45.3 ± 1.0 |
| ResNet-18, 20%, DACIS, PMP | 20% | 77.4 ± 4.9 | 54.5 ± 8.1 | 83.8 ± 8.2 | 44.1 ± 2.0 |
| ResNet-18, 10%, filter L1, PMP | 10% | 70.8 ± 5.5 | 50.9 ± 10.4 | 79.3 ± 9.9 | 39.1 ± 1.1 |
| ResNet-18, 10%, DACIS, PMP | 10% | 71.9 ± 4.8 | 51.4 ± 7.5 | 78.2 ± 10.2 | 40.5 ± 1.1 |
| MobileNetV2, 60%, random, P | 60% | 85.1 ± 2.8 | 67.3 ± 6.5 | -- | -- |
| MobileNetV2, 60%, filter L1, P | 60% | 84.6 ± 2.7 | 66.3 ± 7.9 | -- | -- |
| MobileNetV2, 60%, BN scale, P | 60% | 85.5 ± 2.7 | 63.7 ± 8.4 | -- | -- |
| MobileNetV2, 60%, Taylor, P | 60% | 85.9 ± 2.5 | 66.2 ± 7.6 | -- | -- |
| MobileNetV2, 60%, DACIS, P | 60% | 84.0 ± 2.7 | 65.4 ± 8.0 | -- | -- |
| MobileNetV2, 40%, random, P | 40% | 78.8 ± 3.9 | 62.1 ± 7.5 | -- | -- |
| MobileNetV2, 40%, filter L1, P | 40% | 78.4 ± 3.4 | 62.1 ± 8.2 | -- | -- |
| MobileNetV2, 40%, BN scale, P | 40% | 77.3 ± 3.7 | 57.4 ± 9.5 | -- | -- |
| MobileNetV2, 40%, Taylor, P | 40% | 83.0 ± 2.8 | 63.9 ± 7.1 | -- | -- |
| MobileNetV2, 40%, DACIS, P | 40% | 78.2 ± 3.1 | 62.1 ± 6.0 | -- | -- |
| MobileNetV2, 25%, random, P | 25% | 67.0 ± 4.5 | 55.3 ± 8.3 | -- | -- |
| MobileNetV2, 25%, filter L1, P | 25% | 66.8 ± 4.6 | 52.3 ± 9.8 | -- | -- |
| MobileNetV2, 25%, BN scale, P | 25% | 58.7 ± 5.5 | 51.7 ± 10.6 | -- | -- |
| MobileNetV2, 25%, Taylor, P | 25% | 73.4 ± 4.2 | 58.6 ± 7.6 | -- | -- |
| MobileNetV2, 25%, DACIS, P | 25% | 68.6 ± 2.8 | 55.2 ± 8.5 | -- | -- |

## About earlier numbers

Earlier versions of this README, and the first arXiv versions of the paper, reported different results. The scripts that
produced them had no working dataset loader (they fell back to random tensors), so those numbers cannot be regenerated
and should not be cited. The scripts were removed from the working tree; they remain in the git history before commit
`eba0a99`. Everything here was rebuilt with real data loading, a shared pruning budget solver and paired test episodes.

## Layout

```
docs/             the project page (static HTML, published with GitHub Pages)
pmp_dacis/        library: data, ResNet-18 surgery, scores, training stages, evaluation, report
notebooks/        pmp_dacis_cv4eo.py (main study) and pmp_dacis_cv4eo_ext.py (partitions 5-9, pruned MobileNetV2)
results/          saved runs of the reported experiments (JSON and per-episode arrays)
results_replication/  the same runs on a laptop GPU, and latency on a named CPU
results_tables/   the generated summary and figures of the reported run
tests/            unit tests for the parts where a silent bug would change the science
tools/            the notebook templates and builder, and the CPU latency script
```

The paper sources are not part of this repository while the paper is under review. The notebooks carry an anonymous copy of them
(gzip and base64) so that *Run all* can rebuild the review PDF.

## License

MIT, see `LICENSE`.
