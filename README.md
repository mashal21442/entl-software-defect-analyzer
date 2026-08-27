# ENTL Paper Reproduction Project

This folder upgrades the earlier ENTL-style prototype into a research-oriented
reproduction of:

**Heterogeneous Cross-Project Defect Prediction Using Encoder Networks and Transfer Learning**
(Haque et al., IEEE Access, 2024, DOI: 10.1109/ACCESS.2023.3343329)

## What this version fixes

- Binary defect-label conversion
- Z-score normalization
- Separate source and target encoders
- Same latent dimension for heterogeneous feature spaces
- Cost-sensitive source classifier `Ms`
- 0.5 pseudo-label threshold
- Augmented dataset `HA, YA`
- Secondary model `MT`
- **Repeats `Ms -> pseudo-labels -> MT -> XGBoost` for 20 iterations**
- Averages predictions from the stored XGBoost models per Algorithm 1
- Uses all 16 paper datasets once the raw files are placed correctly
- Aggregates PD, PF, F1, G-Mean and AUC as mean ± standard deviation by target project family
- Creates RQ1 comparison files for EGW, HDP_KS, CTKCCA and EMKCA
- Creates RQ2 comparison files for WPDP
- Includes a Streamlit research dashboard

## Important reproduction limits

The paper does **not** publish enough implementation detail to recreate every
numerical result exactly from the article alone.

The following are explicitly marked as reproduction assumptions:

1. **Encoder training objective and architecture**
   - The paper describes separate encoders and common output dimensionality.
   - It does not state an exact autoencoder/decoder objective or layer sizes.
   - This project uses autoencoder training as a transparent implementation choice.

2. **Neural-network architecture / optimizer / epoch count**
   - Not fully specified in the paper.
   - Fixed values are used in `src/paper_entl.py`.

3. **XGBoost hyperparameters**
   - XGBoost is specified, exact hyperparameters are not.
   - Fixed values are used in `src/paper_entl.py`.

4. **Full source-target mapping**
   - The paper explicitly gives `JIRA activemq5.0.0 -> AEEM EQ` as an example
     and says every source/target pair comes from different projects.
   - The complete 16-pair mapping is not enumerated in the accessible text.
   - `config/experiments.csv` therefore contains a transparent reproduction
     mapping. The first row is the explicit paper example; other rows are marked
     `reproduction_mapping`.

5. **Algorithm 1 vs surrounding ensemble prose**
   - Algorithm 1 averages predictions from stored XGBoost models.
   - The prose around Equation 3 also mentions `MT`.
   - Default mode follows Algorithm 1 (`--ensemble-mode algorithm1`).
   - `--ensemble-mode prose` is provided only as a diagnostic interpretation.

6. **Published baseline standard deviations**
   - The paper's tables are described as mean ± standard deviation.
   - Machine-readable text available to this project exposes the published means,
     not every table cell's standard deviation.
   - `published_baselines.py` stores means only.
   - Your locally reproduced ENTL results do calculate mean ± standard deviation.

## Paper dataset list

| Family | Dataset | Entries | Bugs | Metrics | Bugs % |
|---|---|---:|---:|---:|---:|
| AEEM | EQ | 324 | 129 | 61 | 39.8 |
| AEEM | JDT | 997 | 206 | 61 | 20.7 |
| AEEM | LC | 691 | 64 | 61 | 9.3 |
| AEEM | ML | 1862 | 245 | 61 | 13.2 |
| JIRA | activemq5.0.0 | 1884 | 293 | 65 | 15.6 |
| JIRA | Derby10.5.1.1 | 2705 | 383 | 65 | 14.2 |
| JIRA | Hbase0.94.0 | 1059 | 218 | 65 | 2.6 |
| JIRA | Hive0.9.0 | 1416 | 283 | 65 | 20.0 |
| NASA | KC1 | 2095 | 325 | 21 | 15.5 |
| NASA | PC1 | 735 | 61 | 37 | 8.3 |
| NASA | PC3 | 1099 | 138 | 37 | 12.6 |
| NASA | PC4 | 1379 | 178 | 37 | 12.9 |
| PROMISE | Lucene2.4 | 340 | 203 | 20 | 59.7 |
| PROMISE | Poi3.0 | 442 | 281 | 20 | 63.6 |
| PROMISE | Synapse1.2 | 256 | 86 | 20 | 33.6 |
| PROMISE | Velocity1.6 | 229 | 78 | 20 | 34.1 |

The Hbase percentage is kept exactly as printed in the paper table even though
218 / 1059 does not equal 2.6%; the project deliberately does not silently
correct source material.

## Put raw datasets here

```text
data/raw/
├── AEEM/
│   ├── EQ.csv
│   ├── JDT.csv
│   ├── LC.csv
│   └── ML.csv
├── JIRA/
│   ├── activemq5.0.0.csv
│   ├── Derby10.5.1.1.csv
│   ├── Hbase0.94.0.csv
│   └── Hive0.9.0.csv
├── NASA/
│   ├── KC1.csv
│   ├── PC1.csv
│   ├── PC3.csv
│   └── PC4.csv
└── PROMISE/
    ├── Lucene2.4.csv
    ├── Poi3.0.csv
    ├── Synapse1.2.csv
    └── Velocity1.6.csv
```

## Install

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Validate all 16 datasets

```powershell
python src/validate_datasets.py
```

The validator creates:

```text
results/dataset_validation.csv
```

and warns if your public dataset version does not match the paper's row/bug/metric counts.

## Test one paper example first

The paper explicitly gives ActiveMQ as source and EQ as target:

```powershell
python train.py --quick --only-target AEEM/EQ
```

Then run the full 20-iteration version:

```powershell
python train.py --only-target AEEM/EQ
```

## Run all 16 targets

Quick smoke test:

```powershell
python train.py --quick
```

Full experiment:

```powershell
python train.py
```

This can be computationally expensive because each target performs 20 ENTL
iterations and every iteration trains both `Ms` and `MT` plus an XGBoost model.

## Main results

```text
results/metrics/all_target_results.csv
results/metrics/project_summary_numeric.csv
results/metrics/project_summary_mean_std.csv

results/paper_comparison/published_baseline_means.csv
results/paper_comparison/RQ1_published_comparison.csv
results/paper_comparison/RQ2_published_comparison.csv
results/paper_comparison/local_ENTL_vs_published_ENTL.csv
```

## Published overall means used for RQ2

| Metric | ENTL | WPDP |
|---|---:|---:|
| PD | 0.711 | 0.411 |
| PF | 0.297 | 0.193 |
| F1 | 0.475 | 0.413 |
| G-Mean | 0.770 | 0.553 |
| AUC | 0.663 | 0.641 |

These reproduce the paper's reported improvement statements for PD, F1,
G-Mean and AUC (approximately 72.99%, 15.01%, 39.24% and 3.43%).

## Streamlit

```powershell
streamlit run app.py
```
