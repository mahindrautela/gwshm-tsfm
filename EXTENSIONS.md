# Extended experiments (Phase 2)

Six extensions of *Compare Like with Like*, run on top of the reproduced study ([REPRODUCTION.md](REPRODUCTION.md)).
None of them changes the protocols, splits, seeds or results of Tasks 1–3. Each has its own scripts
(`scripts/ext_*.py`), its own outputs (`results/ext_*/`), and a table in the new appendix section
"Extended Experiments" of the paper (`paper/tables/ext_*.tex`, written by `paper/make_ext_tables.py`).

| | Question | Short answer |
|---|---|---|
| E1 | Does the same-path embedding distance survive temperature change? | Only with dense healthy references (≤ 2 °C apart), where it equals optimal baseline selection. With sparser references, classical OBS + BSS is clearly better. |
| E2 | Do other TSFMs and other read-outs change the picture? | Sundial and Moirai 2 transfer best across NASA coupons; TimesFM 2.5 is weak on Task 1. Middle layers often beat the final one; the last token is the worst read-out. |
| E3 | Does Task 2 generalise to other boundary conditions and layups? | Within a specimen, yes, everywhere. Across specimens only under load (Sundial, Moirai 2, Mantis 0.79–0.86 on all layups); clamped and traction-free do not transfer. Cross-layup references partly work. |
| E4 | Which differences in Tables 3–6 are significant? | Few: the top seven frozen models are indistinguishable on Task 1, and nothing in Task 2 can be significant with four specimens. |
| E5 | Can Task 3 give coordinates, unseen zones, or survive missing sensors? | Coordinates to 22 mm (MOMENT); unseen zones largely fail; missing sensors cost little. |
| E6 | Does light adaptation beat the linear probe? | see E6 below |

Two decisions taken during Phase 2, both confirmed with the author:
- `config.PAPER_MODELS` holds the paper's six models. The Task 1–3 scripts, `make_tables.py` and
  `make_figures.py` use only these, so adding models to `config.MODELS` cannot change the paper's results.
- The extension embeddings were computed on the Apple GPU (`mps`). Five of the six paper models give the
  same embeddings as on cpu (relative difference 1e-7, TiRex 1e-3), and the Task 2 AUCs agree to 1e-4.
  Mantis is wrong on mps and is refused there; Moirai 2 runs on cpu.

---

## MPS check (prerequisite for E1–E6)

`scripts/ext_mps_check.py` → `results/ext_mps/`

| Model | Relative difference mps vs cpu (median) | Speed-up | Task 2 AUC, cpu → mps (within / across) |
|---|---|---|---|
| Chronos-2 | 3–4e-7 | 2.6–3.6× | 0.8820 → 0.8820 / 0.5375 → 0.5375 |
| Chronos-Bolt base | 2–3e-7 | 4.7–4.8× | 0.9789 → 0.9789 / 0.6694 → 0.6694 |
| TiRex | 1.2–1.4e-3 | 14–16× | 0.9781 → 0.9782 / 0.6611 → 0.6611 |
| MOMENT-1 base | 7–8e-8 | 5–7× | 0.9541 → 0.9541 / 0.4723 → 0.4723 |
| Mantis-8M | **0.10–1.05** (wrong) | – | refused on mps |
| TTM-R2 | 2–3e-7 | 1.5–6× | 0.9709 → 0.9709 / 0.5062 → 0.5062 |

Why Mantis fails: its scalar encoders apply `LayerNorm(eps=1e-15)` to near-zero inputs, and the MPS
LayerNorm kernel is off by factors of thousands there (traced layer by layer). Training on mps works:
MOMENT's gradients agree with cpu (cosine 1.000). TiRex has a separate problem on every device: it pads
inputs to its 2,048-sample context with NaN, and 27% of its parameter gradients become NaN, all in the
input embedding and the first 3–4 blocks.

Two further defects were found and handled while adding models (E2):
- **Sundial under transformers 5:** the rotary-embedding tables are non-persistent buffers. They are
  left uninitialised when the model is built on the meta device, so positions are encoded with whatever
  was in that memory, sometimes NaN. They are recomputed after loading, and the result was validated
  against a load under transformers 4.52 (relative difference 1.7e-7).
- **NASA metadata:** two files whose name says boundary condition 2 (clamped) record "Loaded" inside
  (L3S11 at 10 cycles and L3S18 at 100k cycles). Both are later-cycle measurements and are not used for
  detection.

---

## E1 Temperature robustness

**Question.** The paper's OGW-1 data were recorded at constant temperature. Does the same-path
embedding distance still detect damage when healthy references and test signals were recorded at
different temperatures, and how does it compare with classical temperature compensation?

**Protocol** (`ext_fetch_ogw_temperature.py`, `ext_temperature_embed.py`, `ext_temperature.py`)
- **Data:** the OGW CFRP temperature dataset (figshare collection 10.6084/m9.figshare.c.4488089). Same
  plate as OGW-1, 40 kHz only, 36 crossing paths, 1,024 samples at 1 MHz. Only the 40 kHz members were
  read from the 27–54 GB archives, with HTTP range requests (`gwtsfm/remote.py`, standard library only),
  about 13.5 GB in total.
- **Sessions:** two undamaged temperature cycles (udam-1, udam-2), then one cycle each with the defect at
  D12, D16, D04 and D24. Each cycle runs 20.5 → 60 → 20.5 °C in 0.5 °C steps, 161 measurements.
- **References:** udam-1 measurements on a temperature grid with spacing s ∈ {0.5, 2, 5, 10, 20, 40} °C.
  For each grid point, the nearest measurement of the heating half and of the cooling half is taken
  (150, 40, 16, 8, 4 and 2 references). The healthy test is udam-2, a different session; the damaged test
  is the four damaged sessions.
- **Detectors:** each gives a per-path score, averaged over the 36 paths of a measurement.
  - Embedding, same-path distance (Eq. 1) to the mean reference, to the nearest-temperature reference (NT),
    or to the closest reference (OBS in embedding space).
  - Raw signal, residual energy against the NT reference, the best reference (OBS), the NT reference
    stretched by the best factor in 0.96–1.04 (BSS), or OBS + BSS over the three nearest references.
- **Thresholds:** for 1% and 5% false alarms, set from leave-one-out scores of udam-1 only. The
  realised false-alarm rate on udam-2 is reported.
- **Controls:**
  - Trigger lag of each session against udam-1.
  - Healthy drift: udam-2 against the leave-one-out udam-1 scores.
  - Damage locality: Spearman correlation between a path's score change and its proximity to the defect.

**Results** (`results/ext_temperature/summary.csv`; mean over paths; 3 of 9 embeddings shown, all in the file)

| Detector | AUC, s = 0.5 | 2 | 5 | 10 | 20 | 40 °C | Detected at 5% (realised false alarms), s = 2 / s = 5 |
|---|---|---|---|---|---|---|---|
| Raw, nearest temperature | 0.76 | 0.70 | 0.57 | 0.52 | 0.50 | 0.48 | 0.02 (0.00) / 0.00 (0.00) |
| Raw, OBS | 1.00 | 0.96 | 0.69 | 0.55 | 0.50 | 0.48 | 0.30 (0.00) / 0.00 (0.01) |
| Raw, BSS | 0.92 | 0.91 | 0.89 | 0.82 | 0.64 | 0.54 | 0.61 (0.09) / 0.54 (0.07) |
| **Raw, OBS + BSS** | 1.00 | 1.00 | **0.98** | **0.87** | 0.65 | 0.54 | **0.95 (0.01) / 0.73 (0.01)** |
| TiRex, mean reference | 0.55 | 0.55 | 0.55 | 0.55 | 0.56 | 0.50 | 0.00 / 0.01 |
| TiRex, NT | 0.95 | 0.93 | 0.79 | 0.63 | 0.53 | 0.50 | 0.41 (0.01) / 0.02 (0.01) |
| **TiRex, OBS** | **1.00** | **1.00** | 0.92 | 0.67 | 0.54 | 0.50 | 0.63 (0.00) / 0.01 (0.01) |
| MOMENT, OBS | 1.00 | 1.00 | 0.80 | 0.61 | 0.53 | 0.50 | 0.93 (0.00) / 0.00 (0.01) |
| Moirai 2 small, OBS | 0.96 | 0.94 | 0.87 | 0.70 | 0.56 | 0.54 | 0.27 (0.01) / 0.01 (0.01) |
| Sundial, OBS | 0.93 | 0.92 | 0.87 | 0.72 | 0.56 | 0.52 | 0.41 (0.01) / 0.03 (0.01) |
| Mantis, OBS | 0.89 | 0.89 | 0.83 | 0.74 | 0.60 | 0.53 | 0.28 (0.04) / 0.09 (0.01) |

Controls:
- **Timing:** the median lag of every session against udam-1 at matched temperature is −0.03 to +0.08
  samples (interquartile range about 0.5), so there is no trigger offset to exploit.
- **Healthy drift:** udam-2 against leave-one-out udam-1 gives AUC ≤ 0.53 for every detector, so the second
  healthy session does not score higher. On fine grids it scores *lower* (AUC down to 0.08), because
  udam-2's temperatures match the reference set points almost exactly, while leave-one-out scores always
  have a 0.5 °C gap. This makes the thresholds mostly conservative: the realised false-alarm rate at a nominal 5% has a
  median of 0.6% over the 186 detector–spacing pairs and exceeds 5% in 14 of them. Most of those are at
  s = 40 °C or raw BSS (6–9%); the worst are raw NT and raw OBS at s = 40 °C (13–14%).
- **Locality:** paths closer to the defect change more. The Spearman correlation is 0.56–0.64 for the raw
  detectors (s ≤ 5) and 0.3–0.6 for six of the nine embeddings with OBS (Moirai, TimesFM, Sundial, TTM,
  MOMENT, TiRex). It is weak for Chronos-2, Chronos-Bolt and Mantis, and negative for D16 and D24 with
  those models.

**Conclusion.**
- With dense references (≤ 2 °C apart), choosing the closest reference in embedding space works as well
  as classical compensation: AUC 0.98–1.00 for TiRex, MOMENT and both Chronos models.
- As references thin out, classical compensation is clearly better. OBS + BSS keeps AUC 0.98 at 5 °C and
  0.87 at 10 °C; the best embedding reaches 0.92 and 0.74. At a fixed false-alarm rate the gap is larger:
  73% against at most 17% detected at 5 °C spacing.
- The paper's mean-reference distance, which works in Task 2, does not handle temperature (AUC 0.49–0.69).
- So the "compare like with like" recipe extends to temperature only if "like" includes temperature. The
  embeddings do not supply the time-stretch invariance that BSS builds in.

**Limitations.**
- One frequency (40 kHz), one plate, four defect positions.
- Damage is confounded with recording time (the sessions span one month). The controls argue against a
  shortcut but cannot rule out slow drift completely.
- The thresholds are conservative because of how the healthy calibration set is formed (see the drift
  control), so the detection rates at a fixed false-alarm rate understate what a better calibration could
  reach, for every method alike.
- Temperatures are the recorded surface means.

---

## E2 More models and pooling

**Question.** Do other open TSFMs behave like the paper's six, and is mean-pooling the final layer the
right read-out?

**Models added** (`gwtsfm/embed.py`; APIs checked against the installed source; parameters counted in the loaded checkpoint)

| Model | Checkpoint (Hub revision) | Parameters measured | Read-out | Environment |
|---|---|---|---|---|
| TimesFM 2.5 | `google/timesfm-2.5-200m-pytorch` (`1d95242`) | 231.29M | As in `decode`: left-pad to a multiple of 32 with masked zeros, per-patch running normalisation, 20 layers once; mean of patch tokens | `timesfm` 3.0.2 |
| Sundial base | `thuml/sundial-base-128m` (`3212e42`) | 128.33M (backbone 115.80M) | As in `forward(revin=True)`: standardise, 16-sample tokens; mean of hidden states after the final norm; rotary buffers recomputed | remote code, transformers 5.23 |
| Moirai 2.0-R small | `Salesforce/moirai-2.0-R-small` (`30f43ff`) | 11.39M | `Moirai2Forecast._convert` builds the packed input; module run to the encoder output; mean of context tokens | `uni2ts` 2.0.0, separate env (torch 2.4, numpy 1.26), cpu only |
| Toto | `Datadog/Toto-Open-Base-1.0` (`0411ceb`) | – | **not added**, see limitations | `toto-ts` 0.2.0, separate env |

The new models' output dimensions are 1,280 (TimesFM 2.5), 768 (Sundial) and 384 (Moirai 2).

**Protocol** (`ext_models.py`): the Task 1–3 scripts are imported and run unchanged with a nine-model
dictionary. The paper's six reproduce `results/` exactly inside this run (`check_paper_rows.txt`).

**Results** (`results/ext_models/`)

| Model | T1 AUC | T2 within | T2 across | Coupons detected across | T3 acc. per freq. | T3 all | Spearman ρ |
|---|---|---|---|---|---|---|---|
| MOMENT-1 base | 0.86 | 0.95 | 0.47 | 1/4 | 1.00 | 1.00 | 0.80 |
| TiRex | 0.86 | 0.98 | 0.66 | 3/4 | 1.00 | 1.00 | 0.78 |
| **Sundial base** (new) | **0.86** | **0.99** | **0.83** | **4/4** | 1.00 | 1.00 | 0.70 |
| Chronos-Bolt base | 0.85 | 0.98 | 0.67 | 4/4 | 0.96 | 1.00 | 0.64 |
| Chronos-2 | 0.84 | 0.88 | 0.54 | 2/4 | 0.94 | 1.00 | 0.53 |
| **Moirai 2.0 small** (new) | 0.83 | **0.99** | **0.84** | **4/4** | 0.75 | 1.00 | 0.47 |
| Mantis-8M | 0.83 | 0.98 | 0.81 | 4/4 | 0.98 | 1.00 | 0.65 |
| TimesFM 2.5 (new) | 0.75 | 0.89 | 0.57 | 3/4 | 1.00 | 1.00 | 0.75 |
| TTM-R2 | 0.74 | 0.97 | 0.51 | 2/4 | 0.88 | 1.00 | 0.56 |

**Pooling ablation** (`gwtsfm/pooling.py`, `ext_pooling_embed.py`, `ext_pooling.py` → `results/ext_pooling/`)

Forward hooks record the residual stream after every block in the same forward pass. Seven read-outs are
kept for each model and scored with the same protocols: 63 feature sets. Each cell is the mean over the
nine models, with the number of models where the read-out beats the model's own embedding in brackets.

| Read-out | T1 AUC | T2 within | T2 across | T3 acc. (per freq.) | T3 ρ |
|---|---|---|---|---|---|
| Model's own embedding (paper) | 0.82 | 0.96 | 0.66 | 0.95 | 0.65 |
| Mean, block at ¼ depth | 0.81 (3) | 0.97 (6) | 0.67 (4) | 0.99 (4) | **0.75 (7)** |
| Mean, block at ½ depth | 0.82 (5) | **0.98 (6)** | **0.70 (7)** | **0.99 (5)** | 0.73 (6) |
| Mean, block at ¾ depth | 0.82 (3) | 0.97 (6) | 0.68 (5) | 0.97 (4) | 0.69 (4) |
| Mean, last block | 0.82 (3) | 0.96 (3) | 0.65 (2) | 0.94 (2) | 0.64 (2) |
| Max, last block | 0.79 (3) | 0.95 (3) | 0.65 (5) | 0.99 (4) | 0.71 (6) |
| Last token, last block | 0.79 (1) | 0.93 (2) | 0.57 (2) | 0.89 (2) | **0.42 (2)** |

Notable single cases:
- Chronos-2 at ¼ depth: T2 within 0.98 instead of 0.88; across 0.70 instead of 0.54; ρ 0.81 instead of 0.53.
- Sundial at ½ depth: T1 0.89, the best Task 1 feature set of the study.
- Mantis at ½ depth: T2 across 0.88.
- Moirai 2's own embedding localises poorly (0.75); its middle blocks reach 1.00.

**Conclusion.**
- The paper's recipe carries over to newer TSFMs. Sundial (decoder-only, flow matching) and Moirai 2 are
  the best detectors across specimens, with 4/4 coupons like Mantis and Chronos-Bolt. TimesFM 2.5 is the
  second-weakest on Task 1 but locates damage as well as the best models.
- On read-outs, mean-pooling is right and the last token is wrong. The last token of a causal model
  summarises for forecasting and loses the spatial damage signal (ρ 0.42).
- Intermediate blocks are a cheap improvement for detection and localisation. Mean-pooling the final
  block is a sound default, not the optimum.

**Limitations.**
- Layer and read-out were not chosen on held-out data; the best single cases are optimistic. A fair
  comparison would select the layer inside each training fold.
- **Toto was not added.** `toto-ts` pins an old `lightning` that imports `pkg_resources`, which the
  current `setuptools` no longer ships. The fix (`setuptools<81` in the Toto environment) was blocked by
  the session's permission policy, and the embedder was not written against an untested API.
- The new models were embedded on mps and the paper's six on cpu; the MPS check bounds the effect at 1e-4
  AUC.

---

## E3 NASA generalisation

**Question.** Task 2 uses one layup (2) under one boundary condition (loaded). Does detection with healthy
data only carry over to the other layups and boundary conditions, and can a healthy reference from one
layup serve another?

**Protocol** (`ext_nasa_prepare.py`, `ext_nasa_embed.py`, `ext_nasa.py` → `results/ext_nasa/`)
- **Data:** layups 1–3 × conditions 1 (loaded), 2 (clamped), 3 (traction-free), from the same 4.6 GB
  archive. `prepare_nasa.py` is reused unchanged (same seed and selection), with its file pattern widened.
  Layup 2 under the loaded condition is rebuilt and checked to be identical to the paper's array.
- **Data availability:** the clamped and traction-free conditions have 2–7 healthy measurements per
  specimen, and a 20k-cycle measurement for only 1–3 of the 4 specimens. Specimens without one are used
  only as references.
- **Scaling:** signals are scaled by the standard deviation of healthy signals only (the paper used all
  signals; the models normalise inputs anyway).
- **Settings:**
  - Within specimen: half of a specimen's healthy measurements as reference; every split if there are at
    most 5, otherwise 5 random ones.
  - Across specimens: the other coupons of the same layup and condition as reference.
  - Across layups (loaded only): every healthy measurement of another layup as reference, averaged over
    the two source layups.
- **Detectors:** the same-path distance for nine models; raw 1 − correlation and residual energy.

**Results** (signal AUC, mean over specimens; in brackets, specimens whose 20k measurement scores above
all their healthy test measurements). Rows show four models and the raw reference; all nine are in the
paper's table and in `summary.csv`.

| Layup | Features | Within, loaded | Within, clamped | Within, traction-free | Across specimens (loaded) | Across layups (loaded) |
|---|---|---|---|---|---|---|
| 1 | Sundial | 0.99 (4/4) | 0.95 (3/3) | 0.95 (3/3) | 0.84 (4/4) | 0.66 (4/4) |
| 1 | Moirai 2 | 0.98 (4/4) | 0.92 (3/3) | 0.93 (3/3) | 0.83 (4/4) | 0.72 (4/4) |
| 1 | Mantis | 0.99 (4/4) | 0.96 (3/3) | 0.95 (3/3) | 0.81 (4/4) | 0.70 (4/4) |
| 1 | MOMENT | 0.97 (4/4) | 0.94 (3/3) | 0.94 (3/3) | 0.67 (4/4) | 0.55 (2/4) |
| 1 | Raw 1 − corr. | 0.97 (4/4) | 0.91 (3/3) | 0.90 (3/3) | 0.46 (1/4) | 0.45 (0/4) |
| 2 | Sundial | 1.00 (4/4) | 0.95 (1/1) | 0.85 (1/1) | 0.83 (4/4) | 0.78 (4/4) |
| 2 | Moirai 2 | 0.99 (4/4) | 0.97 (1/1) | 0.87 (1/1) | 0.84 (4/4) | 0.80 (4/4) |
| 2 | Mantis | 0.98 (4/4) | 0.90 (1/1) | 0.74 (0/1) | 0.79 (4/4) | 0.62 (2/4) |
| 2 | MOMENT | 0.96 (4/4) | 0.94 (1/1) | 0.88 (1/1) | 0.47 (1/4) | 0.41 (1/4) |
| 2 | Raw 1 − corr. | 0.97 (4/4) | 0.97 (1/1) | 0.94 (1/1) | 0.57 (3/4) | 0.44 (0/4) |
| 3 | Sundial | 0.99 (4/4) | 0.97 (2/2) | 0.98 (2/2) | 0.79 (4/4) | 0.63 (3/4) |
| 3 | Moirai 2 | 0.99 (4/4) | 0.97 (2/2) | 0.98 (2/2) | 0.82 (4/4) | 0.66 (4/4) |
| 3 | Mantis | 0.99 (4/4) | 0.97 (2/2) | 0.98 (2/2) | 0.86 (4/4) | 0.68 (3/4) |
| 3 | MOMENT | 0.93 (4/4) | 0.96 (2/2) | 0.98 (2/2) | 0.70 (4/4) | 0.58 (2/4) |
| 3 | Raw 1 − corr. | 0.99 (4/4) | 0.99 (2/2) | 0.99 (2/2) | 0.45 (1/4) | 0.48 (0/4) |

Summary over all nine models:
- **Within specimen:** AUC 0.82–1.00 under load and 0.70–0.99 clamped or traction-free. The 20k
  measurement is detected at zero false alarms in 215 of 216 model × specimen cases; the miss is Mantis on
  the single layup-2 traction-free specimen.
- **Across specimens:**
  - Loaded: 0.46–0.86 for the embeddings, 0.24–0.33 for raw same-path, 0.45–0.57 for raw 1 − corr., 0.15–0.20 for raw residual energy.
  - Clamped and traction-free: 0.36–0.74 for the embeddings; most models near chance.
- **Across layups (loaded):** 0.41–0.80 for the embeddings, 0.44–0.48 for raw 1 − corr., 0.12–0.14 for raw
  residual energy.
- **Why clamped and traction-free look alike:** their signals correlate at a median of 0.95–0.98 for the
  same specimen and cycle count, against 0.38–0.69 with the loaded condition. Both are unloaded states,
  which explains their nearly identical AUCs. The files are not duplicates.

**Conclusion.**
- The paper's within-specimen result generalises: every layup and boundary condition, every model.
- Its cross-specimen result generalises across layups under the loaded condition. Sundial, Moirai 2 and
  Mantis reach 0.79–0.86 and flag every damaged coupon in layups 1–3, while raw-signal detectors stay near
  or below chance.
- Clamped and traction-free coupons do not transfer across specimens for any method.
- A healthy reference from a *different layup* partly works for Moirai 2 (0.66–0.80, all coupons of every
  target layup detected) and Sundial (0.63–0.78), but much less well than one from the same layup.

**Limitations.**
- Under the unloaded conditions only 1–3 specimens per layup have a 20k measurement (one for layup 2).
- The 20k state is labelled damaged without an independent damage measurement, as in the paper.
- The selection of healthy measurements takes up to 8 at ≤ 3 cycles; for conditions 2–3 these include
  every early repeat there is.
- Cross-layup transfer is reported for the loaded condition only.

---

## E4 Statistics

**Question.** How uncertain are the numbers in Tables 3–6, and which differences are significant?

**Protocol** (`ext_stats.py` → `results/ext_stats/`)
- Every interval resamples the physical unit: Task 1 configurations (156), Task 2 specimens (4), Task 3
  damage positions (28).
- **Task 1:** out-of-fold scores on the paper's folds (logistic regression, sign-symmetric); pooled AUC
  with a 2,000-sample bootstrap over configurations; paired bootstrap tests for all pairs, Holm-corrected;
  the CV repeated with 20 fold assignments; label efficiency with 20 instead of 5 draws.
- **Task 2:** all C(8, 4) = 70 reference splits per specimen instead of 5 random draws; exact sign-flip
  test over specimens.
- **Task 3:** per-position correctness of the network classifier; Clopper–Pearson intervals; exact
  McNemar tests, Holm-corrected.

**Results**

| Features | T1 pooled AUC [95% CI] | T1 over 20 fold sets | T2 AUC over 70 splits (worst split) | T2 detected | T3 correct [95% CI] |
|---|---|---|---|---|---|
| MOMENT-1 base | 0.86 [0.82, 0.90] | 0.84 ± 0.02 | 0.96 (0.78) | 100% | 28/28 [0.88, 1.00] |
| TiRex | 0.85 [0.82, 0.89] | 0.86 ± 0.01 | 0.98 (0.86) | 100% | 28/28 [0.88, 1.00] |
| Sundial base | 0.85 [0.81, 0.89] | 0.85 ± 0.01 | 1.00 (0.96) | 100% | 28/28 [0.88, 1.00] |
| Chronos-Bolt base | 0.84 [0.80, 0.88] | 0.83 ± 0.02 | 0.98 (0.91) | 100% | 28/28 [0.88, 1.00] |
| Chronos-2 | 0.84 [0.79, 0.88] | 0.83 ± 0.02 | 0.89 (0.71) | 100% | 28/28 [0.88, 1.00] |
| Moirai 2.0 small | 0.82 [0.79, 0.87] | 0.82 ± 0.02 | 0.99 (0.94) | 100% | 28/28 [0.88, 1.00] |
| Mantis-8M | 0.82 [0.78, 0.86] | 0.82 ± 0.01 | 0.98 (0.74) | 100% | 28/28 [0.88, 1.00] |
| TimesFM 2.5 | 0.75 [0.70, 0.79] | 0.76 ± 0.01 | 0.90 (0.76) | 100% | 28/28 [0.88, 1.00] |
| TTM-R2 | 0.74 [0.70, 0.78] | 0.76 ± 0.02 | 0.97 (0.84) | 100% | 28/28 [0.88, 1.00] |
| MiniRocket | 0.90 [0.87, 0.93] | – | – | – | – |
| Raw 1 − corr. / residual energy | – | – | 0.97 (0.81) | 100% | 25/28 [0.72, 0.98] |

**Which differences in Tables 3–6 are significant**

- **Table 3 (Task 1).**
  - The seven strongest frozen models (MOMENT, TiRex, Sundial, Chronos-Bolt, Chronos-2, Moirai, Mantis)
    are statistically indistinguishable: every pairwise difference is ≤ 0.04 AUC, uncorrected p ≥ 0.11.
  - The fold assignment alone moves a model by about ±0.02. Across 20 fold sets TiRex ranks first (0.865)
    and MOMENT third (0.845); on the paper's folds MOMENT is first.
  - Significant after Holm correction:
    - TTM-R2 and TimesFM 2.5 are below MOMENT, TiRex, Sundial and Chronos-Bolt.
    - The seven strongest models are above |FFT|.
    - MiniRocket is above Mantis and Moirai.
  - Not significant after correction: MiniRocket against MOMENT, TiRex, Sundial and the two Chronos
    models (+0.04 to +0.06; uncorrected p 0.009–0.08). The paper's "MiniRocket does better (0.90)" holds
    against the frozen models as a group, but not against the best of them individually.
- **Table 4 (Task 2).**
  - With four specimens, the smallest attainable two-sided p of an exact paired test is 0.125, so **no
    difference in Table 4 is significant at the specimen level.**
  - What the extra splits do show is robustness. Every model detects the 20k measurement at zero false
    alarms in all 280 within-specimen runs (4 specimens × 70 splits). Sundial and Moirai have the highest
    worst-case split (0.96, 0.94); Chronos-2 the lowest (0.71).
- **Tables 5–6 (Task 3).**
  - Three frequencies combined: 28/28 for every model against 25/28 for residual energy, not significant
    (McNemar p = 0.25). Every model's interval is [0.88, 1.00].
  - At 80 kHz, every embedding beats residual energy (7/28) significantly (Holm p ≤ 0.04).
  - At 60 kHz the advantage has raw p ≈ 0.001 but does not survive the correction over all 180 tests.
  - Differences between models are never significant.

**Label efficiency with 20 draws** (`task1_fewshot20.csv`): the paper's conclusion holds. Up to 8
configurations, no method exceeds AUC 0.62. From 32 on, every frozen model is above the spectrum (0.61 at
32, 0.65 at 64), and Sundial and Mantis are the most label-efficient (0.70–0.71 at 32).

**Conclusion.** The paper's qualitative claims survive. The fine ordering of the strongest models in
Tables 3 and 4 does not, and should be read as ties.

**Limitations.**
- The configuration bootstrap treats configurations as independent, but they share physical paths across
  blocks (open question 1 below).
- Four specimens cannot support any specimen-level test; more coupons would be needed.

---

## E5 Task 3: coordinates, unseen zones, missing sensors

**Question.** Can the 36-path damage-index map give a position in mm rather than one of seven zones,
extrapolate to a zone never seen in training, and survive missing sensors?

**Protocol** (`ext_task3.py` → `results/ext_task3/`), all from the paper's log-DI maps:
- **Coordinates:** ridge regression of the defect centre, with the penalty chosen by generalised
  cross-validation on training rows only, under the paper's leave-position-out folds. References: RAPID,
  and the centre of the zone predicted by the paper's classifier.
- **Leave one zone out:** 7 folds, each holding out all 4 positions of one zone; regression only.
  References: the mean training position, and the healthy-map control (10 draws).
- **Missing sensors:** at test time 1–4 transducers (with all their paths) or 10–50% of the paths are
  missing, 20 draws each. Either the classifier is retrained on the remaining paths, or the missing values
  are imputed with the training mean.

**Results** (three frequencies combined unless stated; zone accuracy retrained)

| Damage index | Ridge, median mm | RAPID 40 kHz, mm | Unseen zone, ridge mm | Unseen zone, control mm | 2 transducers missing | 4 missing | 50% of paths missing |
|---|---|---|---|---|---|---|---|
| MOMENT-1 base | **22** | 101 | **119** | 187 | 1.00 | 1.00 | 1.00 |
| TiRex | 31 | 89 | 154 | 180 | 1.00 | 1.00 | 1.00 |
| Chronos-Bolt base | 33 | 122 | 123 | 177 | 1.00 | 0.99 | 1.00 |
| TimesFM 2.5 | 35 | 79 | 189 | 185 | 1.00 | 1.00 | 1.00 |
| Chronos-2 | 42 | 126 | 143 | 180 | 1.00 | 0.98 | 0.99 |
| Sundial base | 43 | 83 | 207 | 180 | 1.00 | 0.99 | 1.00 |
| Moirai 2.0 small | 58 | 110 | 146 | 184 | 0.93 | 0.81 | 0.86 |
| Mantis-8M | 62 | 94 | 225 | 177 | 1.00 | 0.94 | 0.99 |
| Residual energy | 66 | 72 | 155 | 179 | 0.87 | 0.70 | 0.77 |
| TTM-R2 | 72 | 108 | 189 | 180 | 0.97 | 0.93 | 0.94 |

The mean training position gives 174 mm on unseen zones. Imputing instead of retraining costs 1–8 points
with 4 transducers missing (e.g. TiRex 0.97, MOMENT 0.92).

**Conclusion.**
- **Coordinates:** the embedding maps locate the defect to 22–43 mm (six models), against 66 mm for
  residual energy and 72–126 mm for training-free RAPID. But when the zone is right, the zone centre is
  only 11 mm from the defect, so on known zones the seven-zone classifier is more precise than regression.
  Regression's value is a continuous estimate.
- **Unseen zones** are a negative result. With a zone never seen in training, errors are 119–225 mm,
  hardly better than no information (174–187 mm). Only the zone in the middle of the plate (D9–D12) is
  found by every method (16–86 mm). With 21 training positions, the classifier learns which pattern
  belongs to which zone; it does not learn geometry it can extrapolate.
- **Missing sensors cost little.** Retrained on the remaining paths, TiRex, MOMENT, TimesFM 2.5 and Sundial
  keep 99–100% with 4 of 12 transducers missing, against 70% for residual energy. The redundancy of the
  network carries over to the embedding maps.

**Limitations.**
- 28 positions in 7 tight clusters; the regression is trained on 21 points.
- The zone definition is the open question 2 below.
- Dropout is simulated by removing columns, not by re-measuring.

---

## E6 Light adaptation

(to be completed)

---

## Open questions, left as they are

1. **Task 1 holds out (block, path) configurations, not provably unseen physical paths.** The E4
   bootstrap inherits this: it resamples configurations as if they were independent.
2. **Task 3's seven 2×2 zones** (D1–4, …, D25–28) are still to be confirmed against the Ultrasonics 2021
   paper. E5's coordinate regression does not depend on the zones; its leave-one-zone-out split does.

## Reproducing Phase 2

```bash
# environments: gwtsfm (main), gwtsfm-moirai (uni2ts), see README
python scripts/ext_mps_check.py
python scripts/embed.py gwtcw nasa ogw1_residuals ogw1_baselines --models timesfm sundial --device mps
~/miniconda3/envs/gwtsfm-moirai/bin/python scripts/embed.py gwtcw nasa ogw1_residuals ogw1_baselines --models moirai2
python scripts/ext_models.py                                  # E2 models
python scripts/ext_pooling_embed.py && ~/miniconda3/envs/gwtsfm-moirai/bin/python scripts/ext_pooling_embed.py --models moirai2
python scripts/ext_pooling.py                                 # E2 pooling
python scripts/ext_fetch_ogw_temperature.py --freq 40         # E1 data, ~13.5 GB read
python scripts/ext_temperature_embed.py && ~/miniconda3/envs/gwtsfm-moirai/bin/python scripts/ext_temperature_embed.py --models moirai2
python scripts/ext_temperature.py                             # E1
python scripts/ext_nasa_prepare.py && python scripts/ext_nasa_embed.py && ~/miniconda3/envs/gwtsfm-moirai/bin/python scripts/ext_nasa_embed.py --models moirai2
python scripts/ext_nasa.py                                    # E3
python scripts/ext_stats.py                                   # E4
python scripts/ext_task3.py                                   # E5
python scripts/ext_adapt.py                                   # E6 (mps)
python paper/make_ext_tables.py
```
