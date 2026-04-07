"""Multi-session simulation engine for longitudinal trend analysis.

Runs a Scenario through the full ECG pipeline — for each session, applies that
session's transform parameters to the raw signal, then runs Pan-Tompkins
detection, feature extraction, beat classification, and session-level metric
aggregation.  Returns a DataFrame with one row per session containing all
session metrics.
"""

from __future__ import annotations

from typing import Callable

import joblib
import numpy as np
import pandas as pd
import torch
import wfdb

from ecg_monitor.models import HybridCNN
from ecg_monitor.pipeline import (
    FEATURE_COLS, FEATURE_COLS_48, DATA_DIR,
    bandpass_filter, compute_patient_baseline, compute_session_metrics,
    extract_beat_features, process_single_record,
)
from ecg_monitor.scenarios import DEFAULT_PARAMS, Scenario
from ecg_monitor.transforms import apply_transforms


MODELS_DIR = 'models'


class SimulationContext:
    """Holds loaded models, PCA, and record data for running simulations.

    Create once at app startup; reuse across multiple scenario runs.
    """

    def __init__(self):
        # Model artifacts were pickled when the module was named 'ecg_pipeline'.
        # Register an alias so unpickling finds the class definitions.
        import sys
        import ecg_monitor.pipeline as _pipeline_mod
        if 'ecg_pipeline' not in sys.modules:
            sys.modules['ecg_pipeline'] = _pipeline_mod

        self.clf = joblib.load(f'{MODELS_DIR}/gb_48feat.joblib')
        self.pca = joblib.load(f'{MODELS_DIR}/qrs_pca.joblib')
        self.pw_pca = joblib.load(f'{MODELS_DIR}/pw_pca.joblib')
        self.frozen_baselines: dict = joblib.load(
            f'{MODELS_DIR}/frozen_baselines.joblib')
        self.feature_scaler = joblib.load(f'{MODELS_DIR}/feature_scaler.joblib')

        cnn_metadata = joblib.load(f'{MODELS_DIR}/hybrid_cnn_metadata.joblib')
        self.hybrid_cnn = HybridCNN(
            seq_len=cnn_metadata['wf_len'],
            n_tabular=cnn_metadata['n_tabular'],
            num_classes=cnn_metadata['num_classes'],
        )
        self.hybrid_cnn.load_state_dict(
            torch.load(f'{MODELS_DIR}/hybrid_cnn.pt',
                       map_location='cpu', weights_only=True))
        self.hybrid_cnn.eval()
        self.cnn_classes: list[str] = cnn_metadata['classes']

        # Cache for loaded records: record_id -> dict
        self._record_cache: dict[str, dict] = {}

    def _load_record(self, record_id: str) -> dict:
        """Load and cache a MIT-BIH record (signal, peaks, labels, baseline)."""
        if record_id in self._record_cache:
            return self._record_cache[record_id]

        record = wfdb.rdrecord(f'{DATA_DIR}/{record_id}')
        ann = wfdb.rdann(f'{DATA_DIR}/{record_id}', 'atr')
        fs = record.fs
        raw_signal = record.p_signal[:, 0].copy()
        filtered = bandpass_filter(raw_signal, fs)

        r_peaks = ann.sample.copy()
        symbols = list(ann.symbol)

        # Map to clinical labels, keep only mapped beats
        from ecg_monitor.pipeline import LABEL_MAP
        mapped = [(r_peaks[i], LABEL_MAP[symbols[i]])
                  for i in range(len(symbols)) if symbols[i] in LABEL_MAP]
        if not mapped:
            raise ValueError(f"Record {record_id} has no mapped beat labels")
        peaks_arr = np.array([m[0] for m in mapped])
        labels = [m[1] for m in mapped]

        # Compute baseline from original signal
        df_base = extract_beat_features(filtered, fs, peaks_arr, labels)
        baseline = compute_patient_baseline(
            df_base, filtered, fs, peaks_arr, labels)

        self._record_cache[record_id] = {
            'signal': filtered,
            'fs': fs,
            'r_peaks': peaks_arr,
            'labels': labels,
            'baseline': baseline,
        }
        return self._record_cache[record_id]


def run_scenario(
    ctx: SimulationContext,
    scenario: Scenario,
    model_type: str = 'gb',
    progress_callback: Callable[[int, int], None] | None = None,
) -> pd.DataFrame:
    """Run a full multi-session simulation for a scenario.

    Args:
        ctx: SimulationContext with loaded models and data.
        scenario: The scenario to simulate.
        model_type: 'gb' for Gradient Boosting or 'hybrid_cnn' for Hybrid CNN.
        progress_callback: Optional callback(session_idx, total_sessions) for
            progress reporting.

    Returns:
        DataFrame with one row per session, columns are session metrics plus
        'session' (0-indexed) and 'transform_params' (dict).
    """
    rec = ctx._load_record(scenario.base_record)
    signal = rec['signal']
    fs = rec['fs']
    r_peaks = rec['r_peaks']
    labels = rec['labels']
    baseline = rec['baseline']

    rows = []
    for session_idx in range(scenario.n_sessions):
        params = scenario.get_params(session_idx)

        # Apply transforms to raw signal
        has_transforms = any(
            params.get(k) != DEFAULT_PARAMS.get(k) for k in params)
        if has_transforms:
            transformed = apply_transforms(signal, fs, r_peaks, labels, params)
        else:
            transformed = signal.copy()

        # Run full pipeline blind on the transformed signal
        keep_wf = (model_type == 'hybrid_cnn')
        df_beats = process_single_record(
            transformed, fs, ctx.pca,
            pw_pca=ctx.pw_pca, baseline=baseline,
            keep_waveforms=keep_wf,
        )

        if len(df_beats) == 0:
            if progress_callback:
                progress_callback(session_idx, scenario.n_sessions)
            continue

        # Classify beats
        df_beats = _classify(ctx, df_beats, model_type)

        # Aggregate to session metrics
        metrics = compute_session_metrics(df_beats, fs=fs,
                                          label_col='predicted_label')

        metrics['session'] = session_idx
        rows.append(metrics)

        if progress_callback:
            progress_callback(session_idx, scenario.n_sessions)

    return pd.DataFrame(rows)


def _classify(
    ctx: SimulationContext,
    df_beats: pd.DataFrame,
    model_type: str,
) -> pd.DataFrame:
    """Add predicted_label column to the beat DataFrame."""
    if model_type == 'hybrid_cnn' and 'beat_waveform' in df_beats.columns:
        wf = np.stack(df_beats['beat_waveform'].values)
        tab = ctx.feature_scaler.transform(df_beats[FEATURE_COLS].values)
        x = np.concatenate([wf, tab], axis=1)
        with torch.no_grad():
            logits = ctx.hybrid_cnn(torch.tensor(x, dtype=torch.float32))
            preds = logits.argmax(dim=1).numpy()
        df_beats['predicted_label'] = [ctx.cnn_classes[p] for p in preds]
    else:
        preds = ctx.clf.predict(df_beats[FEATURE_COLS_48].values)
        df_beats['predicted_label'] = preds

    return df_beats
