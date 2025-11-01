<div align="center">

# 🌞 Solar-Synchronous Circadian Rhythm Optimizer
**Using wearable-derived data to predict and realign human biological clocks**

</div>

---

## 🧠 Overview

This repository hosts a modular deep-learning pipeline that ingests multi-modal signals from the [MMASH dataset](https://physionet.org/content/mmash/), estimates an individual’s circadian phase, computes the ideal solar-synchronous rhythm for their location, and recommends actionable sleep/light-exposure adjustments. All reusable code lives inside `src/`, enabling both notebook exploration and reproducible scripted experiments.

---

### Core Capabilities

- **Preprocessing**: Clean and align sleep diaries, actigraphy, RR-interval, questionnaire, and saliva data.
- **Feature Engineering**: Extract HRV indices, sleep metrics, activity markers, and time-of-day encodings.
- **Solar Context**: Use pvlib to derive sunrise, sunset, solar noon, and solar midnight for any lat/long pair.
- **Deep Model**: CNN → BiGRU architecture predicting circadian phase encoded as $(\sin \phi, \cos \phi)$.
- **Recommendation Engine**: Translate phase differences into a 7-day realignment plan.
- **Visualization**: Plot circadian curves, light vs. sleep offsets, confusion matrices, and weekly schedules.

---

## 🧪 Dataset Primer

| Modality | File | Notes |
|----------|------|-------|
| Sleep | `sleep.csv` | Episode-level start/end times, efficiency, quality metrics |
| Actigraphy | `actigraph.csv` | 3-axis acceleration + heart rate at ~5s cadence |
| RR intervals | `RR.csv` | Beat-to-beat intervals for HRV computation |
| Questionnaires | `questionnaire.csv` | MEQ chronotype, stress, anxiety |
| Saliva | `saliva.csv` | Melatonin & cortisol assays across the day |

Download from PhysioNet and store unzipped CSVs inside `data/raw/`. Processed, time-aligned CSV artefacts are written to `data/processed/`.

---

## 🧱 Repository Layout

```
circadian-rhythm-optimizer/
├── data/
│   ├── raw/
│   └── processed/
├── model/
├── notebooks/
├── outputs/
│   └── plots/
├── src/
│   ├── __init__.py
│   ├── feature_extraction.py
│   ├── model_phase_estimator.py
│   ├── preprocess.py
│   ├── recommendation_engine.py
│   ├── solar_features.py
│   └── visualize.py
├── requirements.txt
└── README.md
```

> Large datasets stay outside version control; drop raw CSVs under `data/raw/`.

---

## ⚙️ Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The code targets Python 3.12+, runs on CPU or GPU (auto-detected), and is compatible with Google Colab (upload the repository or clone it via `git`).

---

## 🚀 Pipeline Walkthrough

1. **Preprocess** (`src/preprocess.py`)
	- Configure dataset layout with `MMASHLoaderConfig`.
	- Call `process_mmash_dataset` to load, clean, harmonise timezones, resample actigraphy, and merge modalities into a single chronological dataframe.

2. **Feature Engineering** (`src/feature_extraction.py`)
	- Use `compute_time_derived_features`, `compute_rrv_features`, and `compute_sleep_metrics` to derive covariates.
	- Generate sliding windows via `make_sliding_windows` and attach $(\sin \phi, \cos \phi)$ targets with `attach_targets`.

3. **Solar Features** (`src/solar_features.py`)
	- Instantiate `SolarContext` with latitude/longitude.
	- Compute sunrise/sunset/solar midnight using `compute_solar_events` and `ideal_solar_schedule`.
	- Convert between radians and clock hours with helper utilities.

4. **Model Training** (`src/model_phase_estimator.py`)
	- Wrap numpy arrays in `PhaseDataset` and split using `prepare_datasets`.
	- Instantiate `CircadianPhaseEstimator`. Train with `train_model` (circular MAE loss) and evaluate via `evaluate_model`.
	- Save weights using `save_model` to `model/circadian_phase_model.pt`.

5. **Recommendations** (`src/recommendation_engine.py`)
	- Combine predicted vs. ideal phase angles, compute aggregate shift, and call `create_weekly_realignment_plan` for personalised advice.

6. **Visual Analytics** (`src/visualize.py`)
	- Plot phase alignment, light/sleep scatter, confusion matrices, and weekly schedules. All functions return Matplotlib axes for notebook embedding.

---

## 📓 Notebooks

| Notebook | Purpose |
|----------|---------|
| `notebooks/01_data_overview.ipynb` | Exploratory analysis checklist, data quality audits, and pipeline entry point. |

Keep notebooks lightweight; move reusable logic into `src/` and import from notebooks.

---

## 📈 Metrics & Evaluation

- **Loss**: Circular Mean Absolute Error (cMAE).
- **Regression**: MAE in minutes, circular correlation.
- **Classification**: Morning/Intermediate/Evening chronotype accuracy + confusion matrix.
- **Visualization**: Phase synchrony plots, solar vs. observed curves, weekly recommendation charts.

---

## 🗂️ Project Timeline (6 Weeks)

| Week | Focus | Output |
|------|-------|--------|
| 1 | Literature review, dataset exploration | Annotated notebooks, data dictionary |
| 2 | Cleaning & preprocessing | Unified dataframe under `data/processed/` |
| 3 | Feature extraction & solar features | Feature matrix, solar schedule helpers |
| 4 | Deep learning modeling | Trained CNN+BiGRU phase estimator |
| 5 | Recommendation engine | Personalised weekly plans |
| 6 | Visualization & documentation | Report-ready plots, README updates |

---

## 🤝 Contributing

1. Create an issue describing the enhancement or bug.
2. Branch from `main` (`git checkout -b feature/...`).
3. Run linting/tests inside your environment.
4. Open a pull request summarising the change and attach relevant plots.

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for details.
