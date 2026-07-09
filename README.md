# Observer–Target Tracking Alignment
### Continuous Time-Series Cross-Correlation & Group-Level Statistics — Pipeline Demonstration

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mooceanstudio/observer-target-tracking-analysis/blob/main/tracking_crosscorr_analysis.ipynb)

A complete, reproducible Python pipeline for a Human–Computer Interaction study
design: **28 observers** (15 Test / 13 Control) continuously rate **6 target
video trials** (1–2 min, sampled every 1000 ms), and the analysis quantifies
how strongly — and with what personal time delay — each observer tracks each
target.

> **Note:** this demonstration runs on **simulated (sample) data** generated to
> match the study design exactly. No real study data is included. Because each
> simulated observer carries a hidden *true* lag, the notebook can verify that
> the pipeline recovers the injected ground truth — a correctness check real
> data cannot offer. The real tracking files drop into `data/raw/` with only
> the ingest cell adapted.

---

## What the pipeline computes

| Tier | Analysis | Output |
|---|---|---|
| **1 — Baseline alignment** | Zero-lag Pearson *r* per observer–target session | 168 baseline values |
| **2 — Temporal dynamics** | Time-lagged cross-correlation over a **−5 s … +10 s** window in **1000 ms steps** — exactly **16 lags per session**, 2,688 correlations total. Boundary rows created by each shift are **dynamically omitted, never zero-filled** (zero-filling artificially attenuates the coefficient — demonstrated in the notebook). | **Peak r** (maximum absolute correlation) and **Peak lag** (personalised latency delay) per session |
| **3 — Group inference** | Mixed-design ANOVA (Group × Target) **and** a linear mixed-effects model (random intercept per observer), on Fisher-z-transformed Peak r and on Peak lag | *F*-statistics, degrees of freedom, exact *p*-values, effect sizes — APA-formatted |

**Lag sign convention:** a positive lag means the observer *trails* the target.

## Repository contents

| File | Role |
|---|---|
| `tracking_crosscorr_analysis.ipynb` | Main notebook — every step explained, all cells executed with visible outputs and figures |
| `tracking_crosscorr_analysis.py` | Mirror production script, **auto-exported** from the notebook via `jupyter nbconvert --to script` (never hand-edited, so it cannot diverge) |
| `requirements.txt` | Pinned dependency manifest |
| `data/raw/T1.csv … T6.csv` | Simulated raw trial files (regenerated deterministically by the notebook) |
| `data/synthetic_ground_truth.csv` | The hidden per-observer parameters used for validation |
| `outputs/master_metrics_sheet.csv` | **Master Metrics Sheet — exactly 168 rows**: Observer ID, Group, Target ID, Trial Length, Baseline Pearson r, Peak r, Peak lag |
| `outputs/anova_apa_table.csv` | APA-formatted ANOVA results (*F*, *df*, *p*, partial η²) |
| `outputs/group_descriptives.csv` | Group means and SDs for all metrics |

## Run it

**In Google Colab (zero setup):** click the badge above, then *Runtime → Run all*.
The first cell installs the one dependency Colab lacks (`pingouin`).

**Locally:**

```bash
git clone https://github.com/mooceanstudio/observer-target-tracking-analysis.git
cd observer-target-tracking-analysis
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/jupyter nbconvert --to notebook --execute --inplace tracking_crosscorr_analysis.ipynb
```

## Reproducibility guarantee

A single fixed seed (`RNG_SEED = 42`) drives every random draw, so every rerun
produces **bit-identical data frames**. The final cell prints an MD5 checksum of
the Master Metrics Sheet (`eb1dc0dbb407555257cea84ddd4ebd7c`); a matching hash
proves an exact reproduction. Verified across environments: the notebook, the
auto-exported mirror script, and two different pandas versions (2.2.2 / 3.0.3)
all reproduce identical output hashes.

## Scientific grounding of the simulation

The simulated observers reproduce documented properties of continuous-rating
traces: reaction lags of 1–6 s (Mariooryad & Busso, 2015; Khorram et al.,
2019), and smooth, gain/bias-varying joystick dynamics as in the CASE dataset
(Sharma et al., 2019). Full citations are in the notebook's final section.
