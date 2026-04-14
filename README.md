# ECG Monitor

An end-to-end ECG analysis pipeline that detects heartbeats, classifies arrhythmias, and tracks cardiac health trends over time. Designed for continuous monitoring of at-risk patients — the system simulates weeks of disease progression on real clinical ECG data and surfaces early warning alerts before conditions become critical.

## What it does

The pipeline processes raw ECG signals through four stages:

```
Raw ECG Signal
    │
    ▼
┌──────────────────────┐
│  1. Beat Detection   │  Pan-Tompkins algorithm with adaptive thresholding
│     (R-peak finding) │  and searchback for missed beats
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  2. Beat             │  34 features (timing, morphology, PCA) →
│     Classification   │  Hybrid CNN + Tabular model (3-class: N/S/V)
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  3. Session          │  PVC/SVE burden, heart rate, HRV (time +
│     Aggregation      │  frequency domain), QRS width, pauses
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  4. Longitudinal     │  EWMA smoothing, z-score deviation from
│     Trend Detection  │  patient baseline, clinical alerting
└──────────────────────┘
```

Steps 1-3 are preprocessing. Step 4 is the clinical output — detecting deteriorating health by comparing each session's metrics against the patient's own baseline.

## Disease simulation

The system applies physiologically grounded signal transformations to real patient ECG recordings to simulate disease progression over dozens of sessions. The full pipeline then runs blind on each transformed signal — transforms are applied to the raw ECG before any analysis, so the system must detect changes the same way it would on a real patient.

**Preset scenarios:**

| Scenario | What it simulates | Key transforms |
|---|---|---|
| HF Decompensation | Worsening heart failure | Rising HR, declining HRV, increasing PVCs |
| Developing AF | Progression toward atrial fibrillation | Rising SVE burden, RR irregularity, P-wave attenuation |
| Conduction Disease | Progressive conduction block | QRS widening, bradycardia, emerging pauses |
| Cardiomyopathy | PVC-induced cardiomyopathy | PVC burden escalation, amplitude reduction |
| Stable Patient | Control (no disease) | No transforms — pipeline noise only |

A custom scenario builder allows mixing any combination of 9 transform parameters (heart rate, HRV compression, PVC/SVE insertion, QRS widening, P-wave flattening, AF irregularity, pause insertion, amplitude scaling) with start/end values interpolated across sessions.

## Dashboard

The Streamlit dashboard lets you select a scenario, run the simulation, and explore:
- **Alert summary** — danger/warning/forecast counts with metric change table
- **Trend charts** — per-metric time series with baseline, warning/danger zones, and forecast projections
- **Clinical interpretation** — each alert includes a plain-language explanation of what it may indicate

## Classification results

Trained and evaluated on the [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/) (48 records, 360 Hz, expert-annotated) using the standard DS1/DS2 inter-patient split.

| Model | Accuracy | N F1 | S F1 | V F1 | Macro F1 |
|---|---|---|---|---|---|
| **Hybrid CNN + Tabular** (mean, 5 seeds) | 92.7% | 0.960 | **0.507** | 0.854 | **0.774** |
| Gradient Boosting (48 features) | 95.5% | 0.976 | 0.298 | 0.921 | 0.732 |
| Gradient Boosting (34 features) | 94.9% | 0.972 | 0.228 | 0.907 | 0.702 |
| MLP (34 features) | 87.6% | 0.929 | 0.389 | 0.737 | 0.685 |

The Hybrid CNN learns P-wave and QRS morphological deviation directly from raw waveform data, while the tabular branch provides RR timing context — together achieving the best supraventricular ectopic (S-class) detection without requiring label-derived templates.

Cross-lead validation on the [INCART database](https://physionet.org/content/incartdb/1.0.0/) (12-lead, 257 Hz) shows moderate transfer penalty (macro F1 0.702 to 0.660) with Lead I comparable to Lead II, confirming viability for wearable deployment.

## Project structure

```
src/ecg_monitor/
├── pipeline.py       # Signal processing, feature extraction, session metrics
├── models.py         # Hybrid CNN architecture (PyTorch)
├── transforms.py     # ECG signal transformations for disease simulation
├── scenarios.py      # Preset + custom disease progression scenarios
├── simulation.py     # Multi-session simulation engine
└── trends.py         # EWMA trend detection and clinical alerting

notebooks/
├── MIT-BIH_Arrythmia.ipynb          # Data exploration, Pan-Tompkins, feature engineering
├── Classification_Experiments.ipynb  # GB/RF models, P-wave features, leakage analysis
├── DL_Model_Experiments.ipynb        # MLP, CNN, LSTM, Hybrid CNN experiments
├── Session_Aggregation.ipynb         # Session metrics, error propagation analysis
└── Generalization.ipynb              # Cross-lead/cross-dataset validation (INCART)

models/                # Trained model artifacts (GB, Hybrid CNN, PCA, baselines)
streamlit_app.py       # Dashboard frontend
scripts/
├── download_data.py   # Download MIT-BIH from PhysioNet
└── entrypoint.sh      # Docker entrypoint (auto-downloads data)
```

## Quickstart

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager

### Run locally

```bash
git clone https://github.com/dominicktan/ecg-monitor.git
cd ecg-monitor
uv sync
make app
```

The MIT-BIH dataset (~100 MB) is downloaded automatically from PhysioNet on first run. The dashboard opens at `http://localhost:8501`.

### Run with Docker

```bash
git clone https://github.com/dominicktan/ecg-monitor.git
cd ecg-monitor
make run
```

This builds the image, downloads the dataset on first run, and starts the dashboard at `http://localhost:8501`.

### Makefile targets

| Command | Description |
|---|---|
| `make app` | Run Streamlit dashboard locally |
| `make data` | Download MIT-BIH database from PhysioNet |
| `make run` | Start dashboard via Docker Compose |
| `make stop` | Stop Docker services |
| `make lint` | Run ruff linter |
| `make format` | Auto-format code |
| `make test` | Run pytest suite |

## Datasets

This project uses publicly available ECG databases from [PhysioNet](https://physionet.org/):

- **MIT-BIH Arrhythmia Database** — 48 half-hour two-channel ECG recordings at 360 Hz with expert beat-level annotations. Used for training and primary evaluation.
- **INCART Database** — 75 twelve-lead ECG recordings at 257 Hz with beat-level annotations. Used for cross-lead generalization validation.

Datasets are not included in the repository. Run `make data` to download MIT-BIH, or download manually from PhysioNet.

## Tech stack

- **ML/Signal Processing:** NumPy, SciPy, scikit-learn, PyTorch, wfdb
- **Dashboard:** Streamlit, Plotly
- **Tooling:** uv, ruff, Docker, GitHub Actions CI
