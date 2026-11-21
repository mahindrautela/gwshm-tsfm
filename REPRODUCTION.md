# Reproduction report

Every result of the paper was recomputed from the public data on a different machine. The table
cells and the numbers quoted in Section 5 are listed below, paper value next to reproduced value.
Every difference is explained.

**Summary.**
- Table 4 (Task 2) and Table 6 (Task 3, network level) reproduce exactly.
- Every number quoted in Section 5 reproduces exactly at the printed precision, including the two
  1D-CNN checks that are only printed by `task1_cnn_check.py`.
- 7 of 192 cells in Tables 3–6 differ, each by 0.01–0.06. Three are the 1D-CNN, which is trained
  from scratch and is not bit-reproducible across CPU architectures. Two come from one test signal,
  or one damage case, that crossed a decision boundary. None of them changes a claim in the paper.
- One real reproducibility bug was found and fixed: the grouped k-fold splits of Task 1 and Task 3
  depended on the CPU architecture (see "Platform-dependent folds" below). Without the fix, Task 1
  reproduces only up to ±0.03 and with different per-fold numbers.

## Setup used

| | Paper | This reproduction |
|---|---|---|
| Hardware | 2-core CPU; x86 inferred from the fold tie order (see below) | Apple M3 Ultra, 28 cores, arm64, macOS 26.7 |
| Python | 3.12, `uv venv` | 3.12.16, conda env `gwtsfm` from conda-forge (the Anaconda channels need a ToS acceptance) |
| Packages | `requirements.txt` + `momentfm==0.1.4 --no-deps` | the same, installed with pip; `uv` is not installed here |
| Unpinned | – | `transformers` 5.23.0, `huggingface-hub` 2.4.0, `accelerate` 1.16.0, `tokenizers` 0.23.3 |
| Device | cpu | cpu (the new `--device` option was left at its default) |

The six checkpoints load without missing or unexpected weights. Measured parameter counts:
- Chronos-2: 119.5M
- Chronos-Bolt base: 205.3M
- TiRex: 35.3M
- MOMENT-1 base: 109.6M
- Mantis-8M: 8.1M
- TTM-R2: 3.0M

All six match Table 2. Embedding all four datasets took 55 CPU-minutes in total, 42 of them for
TiRex. The paper's machine needed several hours.

## How to rerun the comparison

```bash
bash reproduce.sh                          # or the steps one at a time, as in the script
python scripts/compare_results.py --rows 5 # every file in results/ vs paper_results/ on main
python paper/make_tables.py && mkdir -p results/paper_tables_reproduced \
  && cp paper/tables/*.tex results/paper_tables_reproduced/ && git checkout -- paper/tables
python scripts/reproduction_report.py      # the tables and claims below
```

`compare_results.py` reads the reference tables from git (branch `main`), because
`audit_datasets.py` and `export_results.py` overwrite `paper_results/` itself. It matches rows on
their identifying columns and ignores `seconds`.

## Platform-dependent folds (fixed)

The first Task 1 run on this machine differed in every fold, even for methods that use no embeddings
(Raw signal + logistic regression, |FFT|, MiniRocket). The partition itself was different.

The cause is in `sklearn.model_selection.GroupKFold`. Without shuffling, it hands out groups largest
first and orders them with `np.argsort(sizes)`. All our groups have the same size: 2 signals per
Task 1 configuration and 3 frequencies per Task 3 (position, path). So the order is decided by how
`argsort` breaks ties. NumPy 2.x uses different SIMD sort kernels on x86 and on ARM, and they break
ties differently. Same seed, same library versions, different folds.

The paper's folds are exactly those of a stable tie order. With it, the per-fold AUC of raw-signal
logistic regression matches the paper to 6 decimals in all five folds. With the default ARM order it
does not. The fix (commit `c909ad5`, `gwtsfm/splits.py`) is the same greedy algorithm with
`np.argsort(..., kind="stable")`:
- It is identical to `GroupKFold` whenever group sizes differ (tested on 200 random cases).
- It gives the paper's folds on every platform.
- It is used by `task1_gwtcw.py` (and through it `task1_cnn_check.py`) and by `task3_ogw1_paths.py`.
- No protocol, seed or reported result changes. It restores the paper's splits.

| Task 1, before → after the fix | AUC values that differ from the paper | of those, ≥ 0.005 | max \|diff\| |
|---|---|---|---|
| `cv_metrics.csv` (279 rows) | 261 → 56 | 242 → 15 | 0.181 → 0.032 |
| `fewshot.csv` (1,240 rows) | 1,079 → 256 | 1,035 → 24 | 0.240 → 0.015 |
| `task1_cnn_check.py`, 150 epochs, fold 0 | 0.77 → **0.62** (paper 0.62) | | |

## File-by-file comparison with `paper_results/`

| File | Rows | Values that differ | ≥ 0.005 | Max \|diff\| | Where the differences are |
|---|---|---|---|---|---|
| `audits.json` | – | 2 | 0 | 1e-15 | float rounding in two correlations |
| `task1_gwtcw/cv_metrics.csv` | 279 | 56 AUC, 20 bal. acc. | 15, 20 | 0.032, 0.048 | 1D-CNN (all 11 rows), TiRex (24 of 30, ≤ 0.016), single rank flips elsewhere (≤ 0.002) |
| `task1_gwtcw/data_summary.json` | – | 0 | 0 | 0 | |
| `task1_gwtcw/fewshot.csv` | 1,240 | 256 | 24 | 0.015 | TiRex (137 of 155, ≤ 0.015); other models ≤ 0.002; Raw and \|FFT\| exact |
| `task2_nasa/signal_auc.csv` | 624 | 231 | 0 | 0.0005 | TiRex only |
| `task2_nasa/measurement_auc.csv` | 624 | 0 | 0 | 0 | |
| `task2_nasa/growth_curve.csv` | 260 | 216 | 0 | 0.0009 | TiRex, and ratios that involve it |
| `task3_ogw1/metrics.csv` | 314 | 73 acc., 29 pooled acc. | 10, 29 | 0.062, 0.19 | 1D-CNN (up to 0.19 per fold); TiRex (≤ 0.05); one or two examples elsewhere (≤ 0.007) |
| `task3_ogw1/network_localisation.csv` | 56 | 0 | 0 | 0 | |
| `task3_ogw1/spearman.csv` | 21 | 4 | 0 | 0.0003 | TiRex only |

### Where the differences come from

1. **TiRex embeddings differ slightly on this hardware.** TiRex is the only model whose downstream
   numbers move in every task. Its xLSTM recurrence runs thousands of sequential steps in float32,
   so differences in summation order between x86 and ARM BLAS kernels build up. On MPS it differs
   from CPU by 1e-3 relative, while the other models differ by 1e-7. The paper's embeddings are not
   in the repository, so this can't be measured directly. But every TiRex difference is small:
   ≤ 0.016 AUC in Task 1, ≤ 0.0005 in Task 2, and ≤ 0.0003 in Spearman ρ.
2. **The 1D-CNN is trained from scratch, and that training isn't bit-reproducible across CPU
   architectures.** It is seeded, but PyTorch's CPU kernels sum in a different order on x86 and on ARM, and over
   many epochs those rounding differences change the trained weights. Per fold its accuracy moves by
   up to 0.19 in Task 3, which has only 21 cases per fold. Its means move by ≤ 0.06.
3. **Single decision flips from float rounding.** A few rows of models with near-identical embeddings
   (Chronos, MOMENT, Mantis, TTM, and even Raw signal) differ by exactly one ranked pair or one
   example. In Task 1 that is 1/(31·31) ≈ 0.001 AUC, and in Task 3 one example in ~600. These come
   from LBFGS / BLAS rounding in scikit-learn on a different CPU, and they average out in the tables.

MiniRocket, the polarity probe, |FFT| in Tasks 1 and 2, all of Task 2 at measurement level and all
of the network-level Task 3 match to the last digit.

## Tables 3–6 and Section 5
### Tables 3–6

#### Table 3 (Task 1, GW-TCW)

Each cell: paper / reproduced. Differences in bold.

| Features | Sign-sym. AUC | Bal. acc. | As rec. | k-NN head | One-class |
|---|---|---|---|---|---|
| MOMENT-1 base | 0.86 ± 0.04 / 0.86 ± 0.04 | 0.79 / 0.79 | 0.81 / 0.81 | 0.61 / 0.61 | 0.68 / 0.68 |
| TiRex | 0.86 ± 0.06 / 0.86 ± 0.06 | 0.78 / 0.78 | 0.80 / 0.80 | 0.61 / 0.61 | 0.61 / 0.61 |
| Chronos-Bolt base | 0.85 ± 0.03 / 0.85 ± 0.03 | **0.80 / 0.79** | 0.79 / 0.79 | 0.63 / 0.63 | 0.59 / 0.59 |
| Chronos-2 | 0.84 ± 0.05 / 0.84 ± 0.05 | 0.80 / 0.80 | 0.78 / 0.78 | 0.68 / 0.68 | 0.55 / 0.55 |
| Mantis-8M | 0.83 ± 0.06 / 0.83 ± 0.06 | 0.76 / 0.76 | 0.79 / 0.79 | 0.79 / 0.79 | 0.79 / 0.79 |
| TTM-R2 | 0.74 ± 0.05 / 0.74 ± 0.05 | 0.68 / 0.68 | 0.72 / 0.72 | 0.52 / 0.52 | 0.57 / 0.57 |
| MiniRocket + ridge | 0.90 ± 0.04 / 0.90 ± 0.04 | 0.81 / 0.81 | 0.91 / 0.91 | -- / -- | -- / -- |
| \|FFT\| + log. reg. | 0.69 ± 0.04 / 0.69 ± 0.04 | 0.63 / 0.63 | 0.70 / 0.70 | 0.59 / 0.59 | 0.44 / 0.44 |
| 1D-CNN | 0.69 ± 0.07 / 0.69 ± 0.07 | **0.62 / 0.63** | **0.75 / 0.74** | -- / -- | -- / -- |
| Polarity probe | 0.50 / 0.50 | -- / -- | 0.51 / 0.51 | -- / -- | -- / -- |

47 of 50 cells identical.

#### Table 4 (Task 2, NASA)

Each cell: paper / reproduced. Differences in bold.

| Features and detector | Same-path | k-NN | PCA + ocSVM | Detected | Cross: same-path | Cross: detected |
|---|---|---|---|---|---|---|
| Mantis-8M | 0.98 ± 0.06 / 0.98 ± 0.06 | 0.96 / 0.96 | 0.56 / 0.56 | 20/20 / 20/20 | 0.81 / 0.81 | 4/4 / 4/4 |
| Chronos-Bolt base | 0.98 ± 0.02 / 0.98 ± 0.02 | 0.94 / 0.94 | 0.60 / 0.60 | 20/20 / 20/20 | 0.67 / 0.67 | 4/4 / 4/4 |
| TiRex | 0.98 ± 0.02 / 0.98 ± 0.02 | 0.91 / 0.91 | 0.60 / 0.60 | 20/20 / 20/20 | 0.66 / 0.66 | 3/4 / 3/4 |
| Chronos-2 | 0.88 ± 0.04 / 0.88 ± 0.04 | 0.82 / 0.82 | 0.51 / 0.51 | 20/20 / 20/20 | 0.54 / 0.54 | 2/4 / 2/4 |
| TTM-R2 | 0.97 ± 0.03 / 0.97 ± 0.03 | 0.96 / 0.96 | 0.44 / 0.44 | 20/20 / 20/20 | 0.51 / 0.51 | 2/4 / 2/4 |
| MOMENT-1 base | 0.95 ± 0.05 / 0.95 ± 0.05 | 0.93 / 0.93 | 0.62 / 0.62 | 20/20 / 20/20 | 0.47 / 0.47 | 1/4 / 1/4 |
| Baseline subtraction, 1 − corr. | 0.97 ± 0.05 / 0.97 ± 0.05 | -- / -- | -- / -- | 20/20 / 20/20 | 0.57 / 0.57 | 3/4 / 3/4 |
| Baseline subtraction, residual energy | 0.91 ± 0.12 / 0.91 ± 0.12 | -- / -- | -- / -- | 19/20 / 19/20 | 0.19 / 0.19 | 0/4 / 0/4 |
| Raw signal, same-path distance | 0.91 ± 0.12 / 0.91 ± 0.12 | 0.82 / 0.82 | 0.23 / 0.23 | 19/20 / 19/20 | 0.24 / 0.24 | 0/4 / 0/4 |
| \|FFT\|, same-path distance | 0.72 ± 0.09 / 0.72 ± 0.09 | 0.49 / 0.49 | 0.47 / 0.47 | 19/20 / 19/20 | 0.56 / 0.56 | 2/4 / 2/4 |

60 of 60 cells identical.

#### Table 5 (Task 3, path level)

Each cell: paper / reproduced. Differences in bold.

| Features | Random | Grouped | Unseen | 36 paths, unseen |
|---|---|---|---|---|
| MOMENT-1 base | 0.27 / 0.27 | 0.25 / 0.25 | 0.29 / 0.29 | 0.81 / 0.81 |
| Chronos-Bolt base | 0.24 / 0.24 | 0.24 / 0.24 | 0.26 / 0.26 | 0.69 / 0.69 |
| Chronos-2 | 0.25 / 0.25 | 0.24 / 0.24 | 0.25 / 0.25 | 0.67 / 0.67 |
| Mantis-8M | 0.20 / 0.20 | 0.20 / 0.20 | 0.23 / 0.23 | 0.64 / 0.64 |
| TiRex | 0.21 / 0.21 | 0.21 / 0.21 | 0.22 / 0.22 | **0.54 / 0.52** |
| TTM-R2 | 0.19 / 0.19 | 0.19 / 0.19 | 0.22 / 0.22 | 0.52 / 0.52 |
| \|FFT\| spectrum | 0.23 / 0.23 | 0.21 / 0.21 | 0.27 / 0.27 | 0.70 / 0.70 |
| Raw residual | 0.19 / 0.19 | 0.17 / 0.17 | 0.21 / 0.21 | 0.54 / 0.54 |
| MiniRocket† | 0.25 / 0.25 | -- / -- | 0.22 / 0.22 | 0.50 / 0.50 |
| 1D-CNN† | **0.26 / 0.28** | -- / -- | **0.23 / 0.21** | **0.45 / 0.39** |

36 of 40 cells identical.

#### Table 6 (Task 3, network level)

Each cell: paper / reproduced. Differences in bold.

| Damage index | 40 kHz | 60 kHz | 80 kHz | All | Control mean | Control max |
|---|---|---|---|---|---|---|
| TiRex | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 0.20 / 0.20 | 0.39 / 0.39 |
| MOMENT-1 base | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 0.19 / 0.19 | 0.36 / 0.36 |
| Mantis-8M | 1.00 / 1.00 | 1.00 / 1.00 | 0.93 / 0.93 | 1.00 / 1.00 | 0.15 / 0.15 | 0.32 / 0.32 |
| Chronos-Bolt base | 0.96 / 0.96 | 1.00 / 1.00 | 0.93 / 0.93 | 1.00 / 1.00 | 0.16 / 0.16 | 0.36 / 0.36 |
| Chronos-2 | 1.00 / 1.00 | 0.89 / 0.89 | 0.93 / 0.93 | 1.00 / 1.00 | 0.16 / 0.16 | 0.36 / 0.36 |
| TTM-R2 | 0.96 / 0.96 | 0.82 / 0.82 | 0.86 / 0.86 | 1.00 / 1.00 | 0.16 / 0.16 | 0.36 / 0.36 |
| Residual energy | 0.96 / 0.96 | 0.61 / 0.61 | 0.25 / 0.25 | 0.89 / 0.89 | 0.16 / 0.16 | 0.43 / 0.43 |

42 of 42 cells identical.

### Numbers quoted in Section 5

Claims marked † are printed by scripts/task1_cnn_check.py and are not kept in
paper_results/; the reproduced value is read from logs_task1_cnn.txt.

| § | Claim | Paper text | From paper_results/ | Reproduced | Same |
|---|---|---|---|---|---|
| 5.1 | LR, sign-symmetric AUC, five of six models | 0.83–0.86 | 0.83–0.86 | 0.83–0.86 | yes |
| 5.1 | MiniRocket AUC | 0.90 | 0.90 | 0.90 | yes |
| 5.1 | 1D-CNN AUC | 0.69 | 0.69 | 0.69 | yes |
| 5.1 | \|FFT\| spectrum AUC ("does clearly worse") | (worse) | 0.69 | 0.69 | yes |
| 5.1 | TTM-R2 AUC | 0.74 | 0.74 | 0.74 | yes |
| 5.1 | Random split: k-NN AUC on MOMENT, Mantis, TTM | 1.00 | 1.00 | 1.00 | yes |
| 5.1 | Polarity probe AUC (sign-symmetric / as recorded) | chance (Table 3) | 0.50 / 0.51 | 0.50 / 0.51 | yes |
| 5.1 | AUC gain from sign symmetry, every frozen model | 0.02–0.07 | 0.02–0.07 | 0.02–0.07 | yes |
| 5.1 | Mantis one-class AUC (no damage labels) | 0.79 | 0.79 | 0.79 | yes |
| 5.1 | Best AUC of any method with up to 8 labelled configurations | ≤ 0.64 | 0.64 | 0.64 | yes |
| 5.1 | From 32 configurations on, every frozen model ahead of the spectrum | yes | yes | yes | yes |
| 5.2 | Within: same-path AUC, five models | 0.95–0.98 | 0.95–0.98 | 0.95–0.98 | yes |
| 5.2 | Within: same-path AUC, Chronos-2 | 0.88 | 0.88 | 0.88 | yes |
| 5.2 | Within: baseline subtraction, 1 − correlation | 0.97 | 0.97 | 0.97 | yes |
| 5.2 | Within: damaged measurements detected at zero false alarms, worst model | 20/20 | 20/20 | 20/20 | yes |
| 5.2 | Same-path beats k-NN for every feature set | yes | yes | yes | yes |
| 5.2 | Within: PCA + ocSVM on embeddings | 0.44–0.62 | 0.44–0.62 | 0.44–0.62 | yes |
| 5.2 | Within: PCA + ocSVM on raw signals | 0.23 | 0.23 | 0.23 | yes |
| 5.2 | Cross: raw-signal detectors (all five) | 0.15–0.57 | 0.15–0.57 | 0.15–0.57 | yes |
| 5.2 | Cross: raw-signal detectors shown in Table 4 | 0.15–0.57 | 0.19–0.57 | 0.19–0.57 | yes |
| 5.2 | Cross: raw k-NN distance / PCA + ocSVM (not in Table 4) | (in range) | 0.15 / 0.28 | 0.15 / 0.28 | yes |
| 5.2 | Cross: Mantis same-path AUC, coupons detected | 0.81, 4/4 | 0.81, 4/4 | 0.81, 4/4 | yes |
| 5.2 | Cross: Chronos-Bolt same-path AUC, coupons detected | 0.67, 4/4 | 0.67, 4/4 | 0.67, 4/4 | yes |
| 5.2 | Cross: TiRex same-path AUC, coupons detected | 3/4 | 0.66, 3/4 | 0.66, 3/4 | yes |
| 5.2 | Cross: MOMENT same-path AUC, coupons detected | 0.47 | 0.47, 1/4 | 0.47, 1/4 | yes |
| 5.2 | Within: MOMENT same-path AUC | 0.95 | 0.95 | 0.95 | yes |
| 5.2 | Growth: same-path ratio to healthy at 10k cycles, every model | 1.9–3.2 | 1.9–3.2 | 1.9–3.2 | yes |
| 5.2 | Growth: cycle count where each model's ratio peaks | 100k–150k | Chronos-2 150k, Chronos-Bolt 150k, TiRex 150k, MOMENT-1 150k, Mantis-8M 150k, TTM-R2 150k | Chronos-2 150k, Chronos-Bolt 150k, TiRex 150k, MOMENT-1 150k, Mantis-8M 150k, TTM-R2 150k | yes |
| 5.3 | Path level: single-path accuracy, every feature set and split (Table 5) | 0.17–0.29 | 0.17–0.29 | 0.17–0.29 | yes |
| 5.3 | Path level: best 36-path pooled accuracy | 0.81 (MOMENT) | 0.81 (MOMENT-1 (base)) | 0.81 (MOMENT-1 (base)) | yes |
| 5.3 | Network: models with 28/28 (frequencies combined) | 6 of 6 | 6 of 6 | 6 of 6 | yes |
| 5.3 | Clopper–Pearson 95% interval for 28/28 | 0.88–1.00 | 0.88–1.00 | 0.88–1.00 | yes |
| 5.3 | Network: models with 28/28 at every frequency alone | MOMENT, TiRex | TiRex, MOMENT-1 | TiRex, MOMENT-1 | yes |
| 5.3 | Network: residual energy at 40 / 60 / 80 kHz | 0.96 / 0.61 / 0.25 | 0.96 / 0.61 / 0.25 | 0.96 / 0.61 / 0.25 | yes |
| 5.3 | Control: mean accuracy over draws, frozen models | 0.15–0.20 | 0.15–0.20 | 0.15–0.20 | yes |
| 5.3 | Control: mean accuracy, residual energy | 0.15–0.20 | 0.16 | 0.16 | yes |
| 5.3 | Spearman, MOMENT and TiRex, all frequencies | 0.77–0.82 | 0.77–0.82 | 0.77–0.82 | yes |
| 5.3 | Spearman, residual energy at 40 / 80 kHz | 0.49 / 0.16 | 0.49 / 0.16 | 0.49 / 0.16 | yes |
| 5.3 | RAPID at 40 kHz: embeddings vs residual energy | 0.32–0.50 vs 0.75 | 0.32–0.50 vs 0.75 | 0.32–0.50 vs 0.75 | yes |
| 5.3 | RAPID at 80 kHz: embeddings vs residual energy | 0.29–0.61 vs 0.07 | 0.29–0.61 vs 0.07 | 0.29–0.61 vs 0.07 | yes |
| 5.1 | Random split, random early-stopping data: 1D-CNN AUC † | 1.00 | – | 1.00 | yes |
| 5.1 | Grouped split, fold 0, 150 epochs: 1D-CNN AUC † | 0.62 | – | 0.62 | yes |

### The cells that differ

| Table | Cell | Paper → here | Exact values | Explanation |
|---|---|---|---|---|
| 3 | Chronos-Bolt base, bal. acc. | **0.80** → 0.79 | 0.7982 → 0.7950 | One test signal of fold 4 (1 of 62) crossed the 0.5 threshold. Chronos-2, at 0.7986, stays at 0.80, so the bold mark for best balanced accuracy now goes to Chronos-2 alone instead of being shared. Not quoted in the text. |
| 3 | 1D-CNN, bal. acc. | 0.62 → 0.63 | 0.616 → 0.629 | CNN training (item 2 above). |
| 3 | 1D-CNN, as recorded | 0.75 → 0.74 | 0.752 → 0.742 | CNN training. |
| 5 | TiRex, 36 paths, unseen | 0.54 → 0.52 | 0.536 → 0.524 | One damage case of fold 3 (1 of 21) changed zone, from TiRex embeddings (item 1). |
| 5 | 1D-CNN, random / unseen / 36 paths | 0.26 / 0.23 / 0.45 → 0.28 / 0.21 / 0.39 | | CNN training. The text's claim that every single-path method stays within 0.17–0.29 still holds. |

`paper/tables/task3_path_lr.tex`, the appendix table with the logistic-regression head, also changes
in 4 cells by 0.01: Mantis grouped 0.15→0.16, TTM-R2 pooled 0.35→0.36, TiRex pooled 0.64→0.63 and
|FFT| random 0.16→0.17. These have the same causes as items 1 and 3. Appendix Table 9 (growth) and
Table 10 (RAPID) regenerate unchanged.

### Notes on the paper text (no change needed, recorded for completeness)

- *"the raw-signal detectors fall to AUC 0.15–0.57"* (§5.2). The lower end, 0.15, is the raw k-NN
  distance, which is not in Table 4. The raw rows of Table 4 span 0.19–0.57. Both values reproduce.
- *"keeps rising more slowly up to 100,000–150,000 cycles"* (§5.2). For every model the same-path
  ratio peaks at 150k cycles here, as in `paper_results/`.
- *"Removing polarity … raises the AUC of every frozen model by 0.02 to 0.07"* (§5.1). This is
  computed from unrounded means; it reproduces.

## Figures

`paper/figures/make_figures.py` uses `text.usetex`, and **no LaTeX installation is available on this
machine**, so the committed figure PDFs were not regenerated. As a check, the three figures were
rendered with `usetex` switched off into a scratch folder, outside the repository. They are identical
in content to the paper's: same curves, points, maps and ρ values. Only the fonts differ.
Recompiling the paper also needs LaTeX; see the questions in the Phase 2 plan.

## Generated files not committed

`paper/tables/*.tex` and `paper_results/audits.json` were regenerated during the run and then
restored to their committed versions. This keeps the paper's results unchanged, as the task rules
require. The reproduced result CSVs are in `results/`, and the regenerated LaTeX rows in
`results/paper_tables_reproduced/`. Both are ignored by git, and the comparison above
records every difference.

## Open questions left as they are (flagged)

- **Task 1 holds out (block, path) configurations, not provably unseen physical paths.** Two
  configurations in different blocks may share a physical sensor path. Grouping by configuration
  rules out the repeated-acquisition leak, but not path identity across blocks. This is unchanged
  here.
- **Task 3 groups the 28 positions into seven 2×2 zones (D1–4, …, D25–28).** Whether this matches the
  zones of the Ultrasonics 2021 paper is still to be confirmed by the author. The reproduction uses
  the zones as coded.
