"""Real-time ECG Rhythm Transformation Dashboard (Dash).

Streams real patient ECG data continuously with beat-aligned looping and
deferred signal transformations.  Slider changes are spliced in after a
time-based delay using a composite signal: old signal is preserved before
the playhead, transformed signal starts after it — no position jump, no
visible history change.

The classifier runs in real-time: each beat is classified only as the
playhead reaches it, so morphology changes visibly affect classification
rates.  Clinical alerts derive solely from the model's running metrics.

Usage:
    uv run python dash_app.py
"""

import time
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
import wfdb
from dash import Dash, dcc, html, callback, Input, Output, State, no_update, ctx
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

from ecg_monitor.pipeline import (
    FEATURE_COLS, FEATURE_COLS_48, DS1_RECORDS, DS2_RECORDS, DATA_DIR, LABEL_MAP,
    bandpass_filter, process_single_record,
)
from ecg_monitor.transforms import apply_transforms


# =============================================================================
# Hybrid CNN model definition
# =============================================================================

class HybridCNN(nn.Module):
    """1D-CNN on waveform concatenated with tabular features before classifier."""
    def __init__(self, seq_len, n_tabular, num_classes):
        super().__init__()
        self.wf_len = seq_len
        self.waveform_branch = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=9, padding=4),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        self.tabular_branch = nn.Sequential(
            nn.Linear(n_tabular, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(128 + 64, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        wf = x[:, :self.wf_len].unsqueeze(1)
        tab = x[:, self.wf_len:]
        wf_feat = self.waveform_branch(wf)
        tab_feat = self.tabular_branch(tab)
        combined = torch.cat([wf_feat, tab_feat], dim=1)
        return self.classifier(combined)

# =============================================================================
# Constants
# =============================================================================

COLOR = {'N': '#27ae60', 'S': '#f39c12', 'V': '#e74c3c'}
LABEL_NAME = {'N': 'Normal', 'S': 'Supraventricular', 'V': 'Ventricular'}

INTERVAL_MS = 50          # tick callback fires every 50 ms
WINDOW_SECONDS = 8        # visible ECG window width
PLAYBACK_SPEED = 1.0      # 1.0 = real-time
SPLICE_DELAY_S = 5.0      # seconds after slider change before splice triggers

# =============================================================================
# Data loading (runs once at startup — loads pre-trained artifacts)
# =============================================================================

MODELS_DIR = 'models'

print("Loading pre-trained model artifacts...")
clf = joblib.load(f'{MODELS_DIR}/gb_48feat.joblib')
pca = joblib.load(f'{MODELS_DIR}/qrs_pca.joblib')
pw_pca = joblib.load(f'{MODELS_DIR}/pw_pca.joblib')
frozen_baselines = joblib.load(f'{MODELS_DIR}/frozen_baselines.joblib')
baselines = joblib.load(f'{MODELS_DIR}/baselines.joblib')
record_info = joblib.load(f'{MODELS_DIR}/record_info.joblib')

# Hybrid CNN artifacts
feature_scaler = joblib.load(f'{MODELS_DIR}/feature_scaler.joblib')
cnn_metadata = joblib.load(f'{MODELS_DIR}/hybrid_cnn_metadata.joblib')
hybrid_cnn = HybridCNN(
    seq_len=cnn_metadata['wf_len'],
    n_tabular=cnn_metadata['n_tabular'],
    num_classes=cnn_metadata['num_classes'],
)
hybrid_cnn.load_state_dict(
    torch.load(f'{MODELS_DIR}/hybrid_cnn.pt', map_location='cpu', weights_only=True))
hybrid_cnn.eval()
cnn_classes = cnn_metadata['classes']

# Cache raw signals and annotations from MIT-BIH
raw_signals = {}
raw_annotations = {}
for rid in sorted(record_info.keys()):
    record = wfdb.rdrecord(f'{DATA_DIR}/{rid}')
    ann = wfdb.rdann(f'{DATA_DIR}/{rid}', 'atr')
    raw_signals[rid] = {'signal': record.p_signal[:, 0].copy(), 'fs': record.fs}
    raw_annotations[rid] = {'peaks': ann.sample.copy(), 'symbols': list(ann.symbol)}

record_ids = sorted(record_info.keys())
print(f"Ready. {len(record_ids)} records loaded.")


# =============================================================================
# Server-side streaming state (single-user development mode)
# =============================================================================

_stream = {
    'record_id': None,
    'model_type': 'gb',  # 'gb' or 'hybrid_cnn'
    'fs': 360,

    # Original record data (immutable after record load)
    'original_filtered': None,     # bandpass-filtered signal
    'original_peaks': None,        # ground-truth R-peak indices
    'original_symbols': None,      # ground-truth beat labels

    # Original detected features (for extending when no transform applied)
    'original_features': None,     # DataFrame from Pan-Tompkins on original
    'original_det_peaks': None,    # detected R-peak indices
    'original_boundaries': None,   # beat midpoint boundaries

    # Active signal + pre-extracted features (classification deferred to tick)
    'active_signal': None,         # numpy array (grows with each cycle extension)
    'active_features': None,       # DataFrame with per-beat features (no labels)
    'active_peaks': None,          # np.array of R-peak sample indices
    'active_boundaries': [],       # beat midpoint boundaries

    # Pending transform (pre-computed, waiting for splice after SPLICE_DELAY_S)
    'pending_signal': None,        # full transformed signal
    'pending_features': None,
    'pending_peaks': None,
    'pending_boundaries': [],
    'pending_scheduled_at': None,  # wall-clock time when pending was scheduled

    # The full transformed signal for cycle extension (set after first splice)
    'loop_signal': None,           # None = use original; set on splice
    'loop_features': None,
    'loop_peaks': None,
    'loop_boundaries': [],
    'loop_start_sample': 0,        # sample in source signal where loop begins (~1s in)

    # Real-time classification state
    'next_classify_idx': 0,        # index into active_peaks of next beat to classify
    'classified_beats': [],        # beats classified in current signal pass (for markers)
    'all_classified': [],          # all beats ever classified (for running metrics)

    # Wall-clock playback tracking
    'is_playing': True,
    'anchor_time': None,           # wall-clock time when anchor_sample was playing
    'anchor_sample': 0,            # sample position at anchor_time

    # Loop
    'has_looped': False,           # whether at least one extension has occurred
}


# =============================================================================
# Streaming helpers
# =============================================================================

def _beat_boundaries_from_peaks(peaks, signal_len):
    """Compute beat segment boundaries at midpoints between R-peaks."""
    if len(peaks) < 2:
        return [0, signal_len]
    boundaries = [0]
    for i in range(1, len(peaks)):
        boundaries.append((int(peaks[i - 1]) + int(peaks[i])) // 2)
    boundaries.append(signal_len)
    return boundaries


def _prepare_features(signal, fs, baseline=None):
    """Run pipeline up to feature extraction (no classification).

    Args:
        signal: bandpass-filtered ECG signal
        fs: sampling frequency
        baseline: PatientBaseline for frozen normalization, or None

    Returns:
        df_features: DataFrame with extracted features per beat
        peak_indices: np.array of R-peak sample indices
        boundaries: beat midpoint boundaries
    """
    need_waveforms = _stream.get('model_type') == 'hybrid_cnn'
    df_result = process_single_record(signal, fs, pca, pw_pca=pw_pca,
                                       baseline=baseline,
                                       keep_waveforms=need_waveforms)
    if len(df_result) > 0:
        peak_indices = df_result['sample_idx'].values.astype(int)
    else:
        peak_indices = np.array([], dtype=int)
    boundaries = _beat_boundaries_from_peaks(peak_indices, len(signal))
    return df_result, peak_indices, boundaries


def _classify_beats_up_to(current_sample):
    """Classify unclassified beats whose R-peak is at or before current_sample.

    Appends to both classified_beats (current pass) and all_classified (cumulative).
    Returns number of newly classified beats.
    """
    peaks = _stream['active_peaks']
    features = _stream['active_features']
    idx = _stream['next_classify_idx']

    if features is None or peaks is None or len(peaks) == 0:
        return 0

    # Find end index — all beats with R-peak <= current_sample
    end_idx = idx
    while end_idx < len(peaks) and peaks[end_idx] <= current_sample:
        end_idx += 1

    if end_idx <= idx:
        return 0

    # Batch classify for efficiency
    batch = features.iloc[idx:end_idx]
    if _stream.get('model_type') == 'hybrid_cnn':
        wf = np.stack(batch['beat_waveform'].values)
        tab = feature_scaler.transform(batch[FEATURE_COLS].values)
        x = np.concatenate([wf, tab], axis=1)
        with torch.no_grad():
            logits = hybrid_cnn(torch.tensor(x, dtype=torch.float32))
            preds = logits.argmax(dim=1).numpy()
        labels = [cnn_classes[p] for p in preds]
    else:
        labels = clf.predict(batch[FEATURE_COLS_48].values)

    rr_vals = batch['rr_prev'].values
    qw_vals = batch['qrs_width_ms'].values
    ra_vals = batch['r_amplitude'].values

    new_beats = []
    for i in range(end_idx - idx):
        new_beats.append({
            's': int(peaks[idx + i]),
            'l': labels[i],
            'rr': float(rr_vals[i]),
            'qw': float(qw_vals[i]),
            'ra': float(ra_vals[i]),
        })

    _stream['classified_beats'].extend(new_beats)
    _stream['all_classified'].extend(new_beats)
    _stream['next_classify_idx'] = end_idx
    return len(new_beats)


def _init_record(record_id):
    """Initialize streaming state for a new record."""
    sig_data = raw_signals[record_id]
    fs = sig_data['fs']
    filtered = bandpass_filter(sig_data['signal'], fs)

    baseline = frozen_baselines.get(record_id)
    df_features, peak_indices, boundaries = _prepare_features(filtered, fs,
                                                               baseline=baseline)

    # Find loop-start sample: first boundary at or after 1 second
    loop_start_sample = 0
    for b in boundaries[:-1]:
        if b / fs >= 1.0:
            loop_start_sample = b
            break

    _stream.update({
        'record_id': record_id,
        'fs': fs,
        'original_filtered': filtered,
        'original_peaks': raw_annotations[record_id]['peaks'],
        'original_symbols': raw_annotations[record_id]['symbols'],

        'original_features': df_features,
        'original_det_peaks': peak_indices.copy(),
        'original_boundaries': list(boundaries),

        'active_signal': filtered,
        'active_features': df_features.copy(),
        'active_peaks': peak_indices.copy(),
        'active_boundaries': list(boundaries),

        'pending_signal': None,
        'pending_features': None,
        'pending_peaks': None,
        'pending_boundaries': [],
        'pending_scheduled_at': None,

        'loop_signal': None,
        'loop_features': None,
        'loop_peaks': None,
        'loop_boundaries': [],
        'loop_start_sample': loop_start_sample,

        'next_classify_idx': 0,
        'classified_beats': [],
        'all_classified': [],

        'is_playing': True,
        'anchor_time': time.time(),
        'anchor_sample': 0,
        'has_looped': False,
    })


def _get_current_sample():
    """Compute current sample position from wall-clock time."""
    if not _stream['is_playing'] or _stream['anchor_time'] is None:
        return _stream['anchor_sample']
    elapsed = time.time() - _stream['anchor_time']
    return _stream['anchor_sample'] + int(elapsed * _stream['fs'] * PLAYBACK_SPEED)


def _extend_one_cycle():
    """Append one cycle of the loop source to the active signal.

    Uses the transformed signal (loop_signal) if a transform has been spliced,
    otherwise uses the original detected signal.  Skips the first ~1s of each
    cycle (noisy lead-on) using loop_start_sample.
    """
    fs = _stream['fs']

    # Choose source
    if _stream['loop_signal'] is not None:
        src_sig = _stream['loop_signal']
        src_feat = _stream['loop_features']
        src_peaks = _stream['loop_peaks']
        src_bounds = _stream['loop_boundaries']
    else:
        src_sig = _stream['original_filtered']
        src_feat = _stream['original_features']
        src_peaks = _stream['original_det_peaks']
        src_bounds = _stream['original_boundaries']

    # Recompute loop start in source (in case source changed after splice)
    loop_start = 0
    for b in src_bounds[:-1]:
        if b / fs >= 1.0:
            loop_start = b
            break
    _stream['loop_start_sample'] = loop_start

    ext_offset = len(_stream['active_signal'])

    # --- Append signal ---
    _stream['active_signal'] = np.concatenate([
        _stream['active_signal'],
        src_sig[loop_start:],
    ])

    # --- Append peaks ---
    mask = src_peaks >= loop_start
    new_peaks = src_peaks[mask] - loop_start + ext_offset
    _stream['active_peaks'] = np.concatenate([_stream['active_peaks'], new_peaks])

    # --- Append features ---
    n_skip = int(np.sum(src_peaks < loop_start))
    new_feat = src_feat.iloc[n_skip:].copy()
    _stream['active_features'] = pd.concat(
        [_stream['active_features'], new_feat], ignore_index=True)

    # --- Append boundaries ---
    old_bounds = _stream['active_boundaries']
    # Drop the trailing endpoint of old signal
    if old_bounds and old_bounds[-1] == ext_offset:
        old_bounds = old_bounds[:-1]

    src_bounds_after = [b for b in src_bounds if b >= loop_start]
    new_bounds = [b - loop_start + ext_offset for b in src_bounds_after]
    _stream['active_boundaries'] = old_bounds + new_bounds

    # Ensure final boundary = new total length
    total_len = len(_stream['active_signal'])
    if _stream['active_boundaries'][-1] != total_len:
        _stream['active_boundaries'].append(total_len)

    _stream['has_looped'] = True


def _current_beat_idx(sample, boundaries):
    """Find which beat contains the given sample."""
    idx = int(np.searchsorted(boundaries, sample, side='right')) - 1
    return max(0, min(idx, len(boundaries) - 2))


def _schedule_pending(params):
    """Pre-compute transformed signal and extract features (classification deferred)."""
    filtered = _stream['original_filtered']
    fs = _stream['fs']
    peaks = _stream['original_peaks']
    symbols = _stream['original_symbols']

    is_identity = (
        abs(params['hr_factor'] - 1.0) < 0.01 and
        params['hrv_compression'] <= 0 and
        params['n_pauses'] <= 0 and
        params['af_irregularity'] <= 0 and
        abs(params['amplitude_factor'] - 1.0) < 0.01 and
        abs(params['qrs_width_factor'] - 1.0) < 0.01 and
        params['pwave_attenuation'] <= 0 and
        params['pvc_rate'] <= 0 and
        params['sve_rate'] <= 0
    )

    if is_identity:
        signal = filtered
    else:
        signal = apply_transforms(filtered, fs, peaks, symbols, params)

    baseline = frozen_baselines.get(_stream['record_id'])
    df_features, peak_indices, boundaries = _prepare_features(signal, fs,
                                                               baseline=baseline)

    _stream.update({
        'pending_signal': signal,
        'pending_features': df_features,
        'pending_peaks': peak_indices,
        'pending_boundaries': boundaries,
        'pending_scheduled_at': time.time(),
    })


def _tick():
    """Advance playhead, handle splice and loop.

    Splice uses a composite signal: old signal preserved before playhead,
    transformed signal appended after.  No position jump, no history change.
    On loop, the full transformed signal replays from the start.

    Also classifies newly revealed beats.  Returns current sample position.
    """
    if _stream['active_signal'] is None:
        return 0

    current_sample = _get_current_sample()
    signal_len = len(_stream['active_signal'])
    fs = _stream['fs']

    # --- Check splice (time-based delay) ---
    if (_stream['pending_signal'] is not None
            and _stream['pending_scheduled_at'] is not None):
        elapsed = time.time() - _stream['pending_scheduled_at']

        if elapsed >= SPLICE_DELAY_S or current_sample >= signal_len:
            splice_sample = min(current_sample, signal_len)
            old_signal = _stream['active_signal']
            pending_signal = _stream['pending_signal']
            pending_peaks = _stream['pending_peaks']
            pending_features = _stream['pending_features']
            pending_boundaries = _stream['pending_boundaries']

            # --- Find where to start in the pending signal ---
            # Map by beat index: find the beat in old signal at playhead,
            # then start from that beat index in the pending signal.
            old_boundaries = _stream['active_boundaries']
            current_beat = _current_beat_idx(splice_sample, old_boundaries)
            # Clamp to valid range in pending signal
            if current_beat >= len(pending_boundaries) - 1:
                pending_start_sample = 0
                pending_start_beat = 0
            else:
                pending_start_sample = pending_boundaries[current_beat]
                pending_start_beat = current_beat

            # --- Build composite signal ---
            # old[:splice] + pending[pending_start:]
            tail = pending_signal[pending_start_sample:]
            composite = np.concatenate([old_signal[:splice_sample], tail])

            # Offset to shift pending peaks/boundaries into composite coords
            offset = splice_sample - pending_start_sample

            # Composite peaks: keep already-classified peaks + shifted pending
            old_peaks = _stream['active_peaks']
            old_peaks_before = old_peaks[old_peaks <= splice_sample]
            new_peaks_after = pending_peaks[pending_start_beat:] + offset
            composite_peaks = np.concatenate([old_peaks_before, new_peaks_after])

            # Composite features: keep old features for beats before splice,
            # append pending features for beats after
            old_features = _stream['active_features']
            n_old = len(old_peaks_before)
            new_features = pending_features.iloc[pending_start_beat:].copy()
            if n_old > 0 and len(new_features) > 0:
                composite_features = pd.concat(
                    [old_features.iloc[:n_old], new_features],
                    ignore_index=True)
            elif len(new_features) > 0:
                composite_features = new_features.reset_index(drop=True)
            else:
                composite_features = old_features.iloc[:n_old].copy()

            # Composite boundaries
            old_bounds_before = [b for b in old_boundaries if b <= splice_sample]
            new_bounds_after = [b + offset
                                for b in pending_boundaries
                                if b >= pending_start_sample]
            # Avoid duplicate at junction
            if (old_bounds_before and new_bounds_after
                    and old_bounds_before[-1] == new_bounds_after[0]):
                new_bounds_after = new_bounds_after[1:]
            composite_boundaries = old_bounds_before + new_bounds_after
            # Ensure final boundary = signal length
            if not composite_boundaries or composite_boundaries[-1] != len(composite):
                composite_boundaries.append(len(composite))

            # --- Update active state ---
            _stream['active_signal'] = composite
            _stream['active_features'] = composite_features
            _stream['active_peaks'] = composite_peaks
            _stream['active_boundaries'] = composite_boundaries

            # Save full transformed signal for loop replay
            _stream['loop_signal'] = pending_signal
            _stream['loop_features'] = pending_features
            _stream['loop_peaks'] = pending_peaks
            _stream['loop_boundaries'] = pending_boundaries

            # Clear pending
            _stream['pending_signal'] = None
            _stream['pending_features'] = None
            _stream['pending_peaks'] = None
            _stream['pending_boundaries'] = []
            _stream['pending_scheduled_at'] = None

            # Classification: keep existing classified_beats (history intact),
            # just update next_classify_idx to point past already-classified
            _stream['next_classify_idx'] = n_old

            signal_len = len(composite)

    # --- Extend signal for seamless looping ---
    # When playhead approaches the end, append another cycle so the chart
    # scrolls continuously without resetting.
    buffer_ahead = 2 * int(WINDOW_SECONDS * fs)
    extensions = 0
    while (current_sample + buffer_ahead >= len(_stream['active_signal'])
           and extensions < 5):
        _extend_one_cycle()
        extensions += 1

    # --- Classify newly revealed beats ---
    _classify_beats_up_to(current_sample)

    return current_sample


# =============================================================================
# Display helpers
# =============================================================================

def build_transform_params(hr_change, hrv_compression, n_pauses, af_irregularity,
                           amplitude_pct, qrs_width_factor, pwave_attenuation,
                           pvc_rate, sve_rate):
    return {
        'hr_factor': 1.0 / (1.0 + hr_change / 100.0),
        'hrv_compression': hrv_compression / 100.0,
        'n_pauses': n_pauses,
        'af_irregularity': af_irregularity / 100.0,
        'amplitude_factor': amplitude_pct / 100.0,
        'qrs_width_factor': qrs_width_factor,
        'pwave_attenuation': pwave_attenuation / 100.0,
        'pvc_rate': pvc_rate / 100.0,
        'sve_rate': sve_rate / 100.0,
    }


def is_transformed(hr_change, hrv_comp, n_pauses, af_irreg,
                    amp_pct, qrs_factor, pwave_atten, pvc_rate, sve_rate):
    return any([
        hr_change != 0, hrv_comp > 0, n_pauses > 0, af_irreg > 0,
        amp_pct != 100, qrs_factor != 1.0, pwave_atten > 0,
        pvc_rate > 0, sve_rate > 0,
    ])


def get_clinical_alerts(metrics):
    alerts = []
    pvc = metrics.get('pvc_burden_pct', 0) or 0
    if pvc > 10:
        alerts.append(("danger",
            f"PVC burden {pvc:.1f}% exceeds 10% threshold "
            "\u2014 cardiomyopathy risk (reversible with treatment)"))
    elif pvc > 5:
        alerts.append(("warning", f"PVC burden {pvc:.1f}% elevated \u2014 monitor trend"))

    sdnn = metrics.get('sdnn_ms')
    if sdnn is not None and not np.isnan(sdnn):
        if sdnn < 50:
            alerts.append(("danger",
                f"SDNN {sdnn:.0f}ms below 50ms "
                "\u2014 significantly reduced HRV, increased mortality risk"))
        elif sdnn < 100:
            alerts.append(("warning", f"SDNN {sdnn:.0f}ms \u2014 borderline low HRV"))

    hr = metrics.get('mean_hr_bpm')
    if hr is not None and not np.isnan(hr):
        if hr > 100:
            alerts.append(("warning", f"Tachycardia \u2014 resting HR {hr:.0f} bpm"))
        elif hr < 50:
            alerts.append(("warning", f"Bradycardia \u2014 resting HR {hr:.0f} bpm"))

    pauses = metrics.get('pause_count_2s', 0) or 0
    if pauses > 0:
        alerts.append(("warning",
            f"{pauses} pauses >2s detected \u2014 possible conduction disease"))

    sve = metrics.get('sve_burden_pct', 0) or 0
    if sve > 5:
        alerts.append(("warning",
            f"SVE burden {sve:.1f}% elevated \u2014 monitor for AF development"))
    return alerts


def format_metric(val, fmt, unit):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    return f"{val:{fmt}} {unit}"


def format_delta(val, base, fmt, unit):
    if (val is None or base is None
            or (isinstance(val, float) and np.isnan(val))
            or (isinstance(base, float) and np.isnan(base))):
        return ""
    delta = val - base
    if abs(delta) < 0.01:
        return ""
    return f"{delta:+{fmt}} {unit}"


def delta_span(d_str, inverse=False):
    if not d_str:
        return ""
    is_positive = d_str.strip().startswith('+')
    color = ("red" if is_positive else "green") if inverse else (
            "green" if is_positive else "red")
    return html.Span(d_str, style={"color": color, "fontSize": "0.8rem"})


def _compute_running_metrics(beats):
    """Compute progressive metrics from all classified beats."""
    n_seen = len(beats)

    if n_seen < 2:
        return {}, n_seen

    labels = [b['l'] for b in beats]
    rr_vals = [b['rr'] for b in beats]

    n_v = labels.count('V')
    n_s = labels.count('S')

    # N-N intervals for HRV
    nn = []
    for i in range(1, n_seen):
        if labels[i] == 'N' and labels[i - 1] == 'N':
            rr = rr_vals[i]
            if 0.3 < rr < 2.0:
                nn.append(rr)

    hr = sdnn = rmssd = pnn50 = None
    if len(nn) >= 10:
        nn_arr = np.array(nn)
        hr = 60.0 / np.mean(nn_arr)
        sdnn = float(np.std(nn_arr, ddof=1) * 1000)
        diffs = np.diff(nn_arr)
        rmssd = float(np.sqrt(np.mean(diffs ** 2)) * 1000)
        pnn50 = float(100.0 * np.sum(np.abs(diffs) > 0.050) / len(diffs))

    # Morphology from N beats
    n_qw = [b['qw'] for b in beats if b['l'] == 'N']
    n_ra = [b['ra'] for b in beats if b['l'] == 'N']

    # Pauses
    pause_2 = sum(1 for b in beats if b['rr'] > 2.0)
    pause_3 = sum(1 for b in beats if b['rr'] > 3.0)
    max_rr = max(b['rr'] for b in beats)

    return {
        'mean_hr_bpm': hr,
        'sdnn_ms': sdnn,
        'rmssd_ms': rmssd,
        'pnn50_pct': pnn50,
        'pvc_burden_pct': 100.0 * n_v / n_seen,
        'sve_burden_pct': 100.0 * n_s / n_seen,
        'mean_qrs_width_ms': float(np.mean(n_qw)) if n_qw else None,
        'mean_r_amplitude': float(np.mean(n_ra)) if n_ra else None,
        'pause_count_2s': pause_2,
        'pause_count_3s': pause_3,
        'max_rr_interval_s': max_rr,
    }, n_seen


def _build_metric_card(label, val_str, delta_el):
    """Build a single metric card component."""
    return dbc.Col(
        dbc.Card(dbc.CardBody([
            html.P(label, className="text-muted small mb-1"),
            html.H4(val_str, className="mb-0"),
            html.Small(delta_el),
        ], className="py-2 px-3"), className="h-100"),
        xs=6, sm=6, md=3, className="mb-2",
    )


def _build_metrics_panel(running, baseline, n_seen):
    """Build the complete metrics panel as a component tree."""
    def mc(label, key, fmt, unit, inverse=False):
        val = running.get(key)
        v_str = format_metric(val, fmt, unit)
        d_el = delta_span(format_delta(val, baseline.get(key), fmt, unit), inverse)
        return _build_metric_card(label, v_str, d_el)

    return html.Div([
        html.P(f"Beats classified: {n_seen}",
               className="text-muted small mb-2"),
        dbc.Row([
            mc("Heart Rate", "mean_hr_bpm", ".0f", "bpm"),
            mc("SDNN", "sdnn_ms", ".1f", "ms"),
            mc("RMSSD", "rmssd_ms", ".1f", "ms"),
            mc("pNN50", "pnn50_pct", ".1f", "%"),
        ], className="g-2"),
        dbc.Row([
            mc("PVC Burden", "pvc_burden_pct", ".1f", "%", inverse=True),
            mc("SVE Burden", "sve_burden_pct", ".1f", "%", inverse=True),
            mc("QRS Width", "mean_qrs_width_ms", ".1f", "ms", inverse=True),
            mc("R-Amplitude", "mean_r_amplitude", ".3f", "mV"),
        ], className="g-2"),
        dbc.Row([
            mc("Pauses (>2s)", "pause_count_2s", ".0f", "", inverse=True),
            mc("Pauses (>3s)", "pause_count_3s", ".0f", "", inverse=True),
            mc("Max RR", "max_rr_interval_s", ".2f", "s", inverse=True),
        ], className="g-2"),
    ])


# =============================================================================
# Layout
# =============================================================================

def make_slider(id, label, min_val, max_val, value, step, marks=None):
    if marks is None:
        marks = {min_val: str(min_val), max_val: str(max_val)}
    return html.Div([
        html.Label(label, className="small fw-bold mt-2 mb-0"),
        html.Div(
            dcc.Slider(id=id, min=min_val, max=max_val, value=value, step=step,
                       marks=marks, tooltip={"placement": "bottom"}),
            className="px-1",
        ),
    ])


app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    title="ECG Rhythm Transformer",
)

sidebar = dbc.Card([
    dbc.CardBody([
        html.H5("Patient Selection", className="mb-3"),
        dcc.Dropdown(
            id="record-select",
            options=[{"label": f"Record {r}", "value": r} for r in record_ids],
            value="208",
            clearable=False,
        ),
        html.Div(id="record-info", className="text-muted small mt-1 mb-3"),

        html.H6("Classification Model", className="mt-2 mb-1"),
        dcc.Dropdown(
            id="model-select",
            options=[
                {"label": "Gradient Boosting (48 feat)", "value": "gb"},
                {"label": "Hybrid CNN + Tabular", "value": "hybrid_cnn"},
            ],
            value="gb",
            clearable=False,
        ),
        html.Div(id="model-info", className="text-muted small mt-1 mb-3"),

        html.H6("Playback", className="mt-3 mb-2"),
        dbc.ButtonGroup([
            dbc.Button("Play", id="btn-play", color="success", size="sm", n_clicks=0),
            dbc.Button("Pause", id="btn-pause", color="secondary", size="sm",
                       outline=True, n_clicks=0),
            dbc.Button("Reset", id="btn-reset", color="secondary", size="sm",
                       outline=True, n_clicks=0),
        ], className="mb-2 w-100"),
        html.Div([
            html.Label("Speed", className="small fw-bold me-2"),
            dbc.RadioItems(
                id="speed-select",
                options=[
                    {"label": "1x", "value": 1},
                    {"label": "2x", "value": 2},
                    {"label": "4x", "value": 4},
                    {"label": "8x", "value": 8},
                ],
                value=1,
                inline=True,
                className="small",
            ),
        ], className="d-flex align-items-center mb-1"),
        html.Div(id="playback-time", className="text-muted small mb-2"),

        html.Hr(),
        html.H5("Transformations", className="mb-2"),

        html.H6("Timing / Rhythm", className="mt-2"),
        make_slider("sl-hr", "Heart Rate Change (%)", -30, 50, 0, 5,
                    marks={-30: "-30", 0: "0", 50: "+50"}),
        make_slider("sl-hrv", "HRV Compression (%)", 0, 90, 0, 5,
                    marks={0: "0", 90: "90"}),
        make_slider("sl-pauses", "Inserted Pauses", 0, 10, 0, 1,
                    marks={0: "0", 10: "10"}),
        make_slider("sl-af", "AF Irregularity (%)", 0, 50, 0, 5,
                    marks={0: "0", 50: "50"}),

        html.H6("Morphology", className="mt-3"),
        make_slider("sl-amp", "R-Amplitude (%)", 20, 100, 100, 5,
                    marks={20: "20", 100: "100"}),
        make_slider("sl-qrs", "QRS Width Factor", 1.0, 2.5, 1.0, 0.1,
                    marks={1.0: "1.0", 2.5: "2.5"}),
        make_slider("sl-pwave", "P-wave Attenuation (%)", 0, 100, 0, 5,
                    marks={0: "0", 100: "100"}),

        html.H6("Ectopic Insertion", className="mt-3"),
        make_slider("sl-pvc", "Additional PVC (%)", 0, 40, 0, 1,
                    marks={0: "0", 40: "40"}),
        make_slider("sl-sve", "Additional SVE (%)", 0, 30, 0, 1,
                    marks={0: "0", 30: "30"}),
        html.Div(id="ectopic-info", className="text-muted small mt-1"),
    ]),
], className="h-100")

main_content = html.Div([
    html.H3("ECG Rhythm Transformation Dashboard", className="mb-1"),
    html.P(
        "Real-time ECG streaming with interactive signal transformations. "
        "Full ML pipeline: Pan-Tompkins \u2192 beat classification "
        "\u2192 session metrics \u2192 alerts.",
        className="text-muted small mb-3",
    ),

    dbc.Card([
        dbc.CardBody([
            dcc.Graph(id="ecg-graph", config={"displayModeBar": False},
                      style={"height": "320px"}),
        ], className="py-1"),
    ], className="mb-3"),

    html.H5("Session Metrics", className="mb-2"),
    html.Div(id="metrics-panel"),

    html.H5("Clinical Alerts", className="mb-2 mt-3"),
    html.Div(id="alerts-panel"),
])

app.layout = dbc.Container([
    dcc.Interval(id="interval", interval=INTERVAL_MS, n_intervals=0),
    dbc.Row([
        dbc.Col(sidebar, width=3, className="vh-100 overflow-auto py-3"),
        dbc.Col(main_content, width=9, className="py-3 overflow-auto",
                style={"height": "100vh", "overflowY": "auto"}),
    ], className="g-0"),
], fluid=True, className="px-0")


# =============================================================================
# Callbacks
# =============================================================================

@callback(
    Output("record-info", "children"),
    Output("ectopic-info", "children"),
    Output("model-info", "children"),
    Input("record-select", "value"),
    Input("model-select", "value"),
    Input("sl-hr", "value"),
    Input("sl-hrv", "value"),
    Input("sl-pauses", "value"),
    Input("sl-af", "value"),
    Input("sl-amp", "value"),
    Input("sl-qrs", "value"),
    Input("sl-pwave", "value"),
    Input("sl-pvc", "value"),
    Input("sl-sve", "value"),
    prevent_initial_call='initial_duplicate',
)
def on_config_change(record_id, model_type, hr_change, hrv_comp, n_pauses, af_irreg,
                     amp_pct, qrs_factor, pwave_atten, pvc_rate, sve_rate):
    if not record_id:
        return "", "", ""

    # Model type changed — re-init record with new feature extraction
    model_changed = _stream.get('model_type') != model_type
    _stream['model_type'] = model_type

    # Determine if record changed
    need_init = _stream['record_id'] != record_id or model_changed

    if need_init:
        _init_record(record_id)

    model_info = ("Hybrid CNN + Tabular (macro F1 0.774, S F1 0.507)"
                  if model_type == 'hybrid_cnn'
                  else "HistGradientBoosting 48-feat (macro F1 0.732)")

    # Record info
    ri = record_info.get(record_id, {})
    n_n = ri.get('n_N', 0)
    n_s = ri.get('n_S', 0)
    n_v = ri.get('n_V', 0)
    ds = "DS1/train" if record_id in DS1_RECORDS else "DS2/test"
    info = f"{ds} | N: {n_n}  S: {n_s}  V: {n_v} | Total: {ri.get('n_total', 0)}"

    ectopic_msg = []
    if n_v == 0:
        ectopic_msg.append("No V beats \u2014 PVC insertion disabled")
    if n_s == 0:
        ectopic_msg.append("No S beats \u2014 SVE insertion disabled")

    eff_pvc = pvc_rate if n_v > 0 else 0
    eff_sve = sve_rate if n_s > 0 else 0

    transformed = is_transformed(hr_change, hrv_comp, n_pauses, af_irreg,
                                 amp_pct, qrs_factor, pwave_atten, eff_pvc, eff_sve)

    if transformed:
        params = build_transform_params(
            hr_change, hrv_comp, n_pauses, af_irreg,
            amp_pct, qrs_factor, pwave_atten, eff_pvc, eff_sve,
        )
        # Schedule pending transform (runs pipeline, ~0.5-1s)
        _schedule_pending(params)
    elif not need_init:
        # Sliders returned to default — schedule revert to original
        _schedule_pending(build_transform_params(0, 0, 0, 0, 100, 1.0, 0, 0, 0))

    return info, " | ".join(ectopic_msg), model_info


@callback(
    Output("btn-play", "outline"),
    Output("btn-pause", "outline"),
    Input("btn-play", "n_clicks"),
    Input("btn-pause", "n_clicks"),
    Input("btn-reset", "n_clicks"),
    Input("speed-select", "value"),
    prevent_initial_call=True,
)
def playback_controls(play_clicks, pause_clicks, reset_clicks, speed):
    global PLAYBACK_SPEED
    trigger = ctx.triggered_id

    if trigger == "speed-select":
        # Re-anchor at current position with new speed
        _stream['anchor_sample'] = _get_current_sample()
        PLAYBACK_SPEED = speed
        if _stream['is_playing']:
            _stream['anchor_time'] = time.time()
    elif trigger == "btn-play":
        # Resume from current position
        _stream['anchor_time'] = time.time()
        # anchor_sample stays where we paused
        _stream['is_playing'] = True
    elif trigger == "btn-pause":
        # Freeze current position
        _stream['anchor_sample'] = _get_current_sample()
        _stream['anchor_time'] = None
        _stream['is_playing'] = False
    elif trigger == "btn-reset":
        _stream['anchor_time'] = time.time()
        _stream['anchor_sample'] = 0
        _stream['is_playing'] = True
        _stream['has_looped'] = False
        # Reset active signal to original (undo extensions and splices)
        _stream['active_signal'] = _stream['original_filtered']
        _stream['active_features'] = _stream['original_features'].copy()
        _stream['active_peaks'] = _stream['original_det_peaks'].copy()
        _stream['active_boundaries'] = list(_stream['original_boundaries'])
        # Reset classification state
        _stream['classified_beats'] = []
        _stream['all_classified'] = []
        _stream['next_classify_idx'] = 0
        # Clear loop/pending state
        _stream['loop_signal'] = None
        _stream['loop_features'] = None
        _stream['loop_peaks'] = None
        _stream['loop_boundaries'] = []

    playing = _stream['is_playing']
    return not playing, playing


@callback(
    Output("ecg-graph", "figure"),
    Output("playback-time", "children"),
    Output("metrics-panel", "children"),
    Output("alerts-panel", "children"),
    Input("interval", "n_intervals"),
)
def tick(n_intervals):
    # --- Not initialized yet ---
    if _stream['active_signal'] is None:
        fig = go.Figure()
        fig.update_layout(margin=dict(l=40, r=20, t=10, b=30),
                          xaxis_title="Time (s)", yaxis_title="mV")
        return (fig, "Loading...",
                html.Div("Initializing pipeline..."),
                [dbc.Alert("Loading...", color="info", className="py-2")])

    # --- Advance playhead, handle splice/loop, classify new beats ---
    current_sample = _tick()

    signal = _stream['active_signal']
    fs = _stream['fs']
    classified = _stream['classified_beats']
    all_beats = _stream['all_classified']
    signal_len = len(signal)
    record_id = _stream['record_id']
    baseline = baselines.get(record_id, {})

    window_samples = int(WINDOW_SECONDS * fs)

    # --- ECG trace (playhead is the right edge of the window) ---
    s1 = min(int(current_sample), signal_len)
    s0 = max(0, s1 - window_samples)
    if s1 <= s0:
        s0, s1 = 0, min(window_samples, signal_len)

    t_axis = np.arange(s0, s1) / fs

    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=t_axis.tolist(), y=signal[s0:s1].tolist(),
        mode='lines', line=dict(color='#333', width=1),
        showlegend=False, hoverinfo='skip',
    ))

    # Beat markers — only show classified beats in visible window
    for label in ['N', 'S', 'V']:
        bx, by = [], []
        for b in classified:
            si = b['s']
            if s0 <= si < s1 and b['l'] == label:
                bx.append(si / fs)
                by.append(float(signal[si]) if 0 <= si < signal_len else 0)
        if bx:
            fig.add_trace(go.Scattergl(
                x=bx, y=by, mode='markers',
                marker=dict(color=COLOR[label], size=7),
                name=f'{LABEL_NAME[label]} ({label})',
            ))

    fig.update_layout(
        margin=dict(l=40, r=20, t=10, b=30),
        xaxis_title="Time (s)", yaxis_title="mV",
        xaxis=dict(range=[s0 / fs, (s0 + window_samples) / fs]),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, x=0),
        hovermode='x unified',
        uirevision='ecg',
    )

    # --- Playback time (cumulative, no reset) ---
    total_seconds = current_sample / fs
    minutes = int(total_seconds) // 60
    secs = total_seconds % 60
    pending_str = " \u2502 transform pending..." if _stream['pending_signal'] is not None else ""
    time_str = f"{minutes}:{secs:04.1f}{pending_str}"

    # --- Running metrics (from all classified beats, cumulative) ---
    running, n_seen = _compute_running_metrics(all_beats)
    metrics_panel = _build_metrics_panel(running, baseline, n_seen)

    # --- Clinical alerts (derived solely from model's running metrics) ---
    if running:
        alerts = get_clinical_alerts(running)
        if alerts:
            alerts_children = [dbc.Alert(msg, color=sev, className="py-2 mb-1")
                               for sev, msg in alerts]
        else:
            alerts_children = [dbc.Alert("All metrics within normal clinical range",
                                         color="success", className="py-2")]
    else:
        alerts_children = [dbc.Alert("Calibrating...", color="info", className="py-2")]

    return fig, time_str, metrics_panel, alerts_children


# =============================================================================
# Run
# =============================================================================

if __name__ == '__main__':
    app.run(debug=True, port=8050)
