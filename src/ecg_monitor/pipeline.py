"""Shared ECG processing pipeline — constants, signal processing, and data loading.

Extracted from MIT-BIH_Arrythmia.ipynb to eliminate code duplication across notebooks.
"""

import dataclasses
import wfdb
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks, welch, resample as sig_resample
from scipy.interpolate import CubicSpline
from scipy.stats import kurtosis, skew
from sklearn.decomposition import PCA


# =============================================================================
# Constants
# =============================================================================

DATA_DIR = 'mit-bih-arrhythmia-database-1.0.0'

BEAT_SYMBOLS = {'N', 'L', 'R', 'A', 'V', 'a', 'J', 'S', 'e', 'j', 'F', '/', 'f', 'Q'}

PACED_RECORDS = {'102', '104', '107', '217'}

DS1_RECORDS = {'101', '106', '108', '109', '112', '114', '115', '116', '118', '119',
               '122', '124', '201', '203', '205', '207', '208', '209', '215', '220',
               '223', '230'}

DS2_RECORDS = {'100', '103', '105', '111', '113', '117', '121', '123', '200', '202',
               '210', '212', '213', '214', '219', '221', '222', '228', '231', '232',
               '233', '234'}

# MIT-BIH labels → clinical 3-class scheme
# N (Normal conduction):     N, L, R, e, j
# S (Supraventricular):      A, a, J, S
# V (Ventricular):           V, F
LABEL_MAP = {
    'N': 'N', 'L': 'N', 'R': 'N', 'e': 'N', 'j': 'N',
    'A': 'S', 'a': 'S', 'J': 'S', 'S': 'S',
    'V': 'V', 'F': 'V',
}

# Original 28-feature set (backward compatibility)
FEATURE_COLS_28 = [
    # Timing
    'rr_prev', 'rr_curr', 'rr_ratio', 'hr_inst',
    # RR context (local-mean-based)
    'rr_prev_ratio_mean', 'rr_post_ratio_mean', 'comp_pause_ratio',
    # Morphology
    'r_amplitude', 'qrs_width_ms', 'qrs_area',
    'qrs_max', 'qrs_min', 'qrs_range', 'qrs_skew', 'qrs_kurt',
    # Per-patient normalized morphology
    'r_amplitude_norm', 'qrs_width_ms_norm', 'qrs_area_norm',
    # PCA beat waveform components
    'pca_0', 'pca_1', 'pca_2', 'pca_3', 'pca_4',
    'pca_5', 'pca_6', 'pca_7', 'pca_8', 'pca_9',
]

# Robust RR features (resistant to S-beat-dominated local windows)
ROBUST_RR_FEATURES = [
    'rr_local_std',
    'rr_prev_ratio_median', 'rr_post_ratio_median', 'comp_pause_ratio_median',
    'rr_prev_ratio_max', 'rr_post_ratio_max',
]

# Full 34-feature set (28 original + 6 robust RR)
FEATURE_COLS = FEATURE_COLS_28 + ROBUST_RR_FEATURES

# RR-only features (7 original) — lead-independent, transfer across hardware
RR_ONLY_FEATURES_7 = [
    'rr_prev', 'rr_curr', 'rr_ratio', 'hr_inst',
    'rr_prev_ratio_mean', 'rr_post_ratio_mean', 'comp_pause_ratio',
]

# RR-only features (13) — original + robust variants
RR_ONLY_FEATURES = RR_ONLY_FEATURES_7 + ROBUST_RR_FEATURES

# Feature ablation variants for two-stage classifier Stage 2
FEATURE_VARIANTS = {
    'v1 (3)': ['comp_pause_ratio', 'qrs_width_ms', 'qrs_width_ms_norm'],
    'v2 (5)': ['comp_pause_ratio', 'qrs_width_ms', 'qrs_width_ms_norm',
               'qrs_area_norm', 'r_amplitude_norm'],
    'v3 (8)': ['comp_pause_ratio', 'qrs_width_ms', 'qrs_width_ms_norm',
               'qrs_area_norm', 'r_amplitude_norm',
               'qrs_skew', 'qrs_kurt', 'qrs_range'],
    'v4 (13)': ['comp_pause_ratio', 'qrs_width_ms', 'qrs_width_ms_norm',
                'qrs_area_norm', 'r_amplitude_norm',
                'qrs_skew', 'qrs_kurt', 'qrs_range',
                'pca_0', 'pca_1', 'pca_2', 'pca_3', 'pca_4'],
    'v5 (18)': ['comp_pause_ratio', 'qrs_width_ms', 'qrs_width_ms_norm',
                'qrs_area_norm', 'r_amplitude_norm',
                'qrs_skew', 'qrs_kurt', 'qrs_range',
                'pca_0', 'pca_1', 'pca_2', 'pca_3', 'pca_4',
                'pca_5', 'pca_6', 'pca_7', 'pca_8', 'pca_9'],
    'v6 (28)': FEATURE_COLS_28[:],  # original 28 features
    'v7 (34)': FEATURE_COLS[:],    # 28 + 6 robust RR features
}

# P-wave feature columns
PW_RESAMPLE_LEN = 44  # fixed resampled length (~120ms at 360 Hz)
N_PW_PCA = 5

PW2_RAW_COLS = [
    'pw2_peak_amplitude', 'pw2_energy', 'pw2_peak_prominence',
    'pw2_has_peak', 'pw2_baseline_dev', 'pw2_max', 'pw2_min',
    'pw2_range', 'pw2_area',
]
PW2_PNORM_COLS = [f'{c}_pnorm' for c in PW2_RAW_COLS]
PW_PCA_COLS = [f'pw_pca_{i}' for i in range(N_PW_PCA)]

# 11 raw amplitude-dependent features (dropped from classifier for robustness)
DROPPED_AMP_FEATURES = {
    'r_amplitude', 'qrs_max', 'qrs_min', 'qrs_range', 'qrs_area',
    'pw2_peak_amplitude', 'pw2_energy', 'pw2_max', 'pw2_min',
    'pw2_range', 'pw2_area',
}

# Full 59-feature hybrid set (for reference)
FEATURE_COLS_59 = (
    FEATURE_COLS + PW2_RAW_COLS + PW2_PNORM_COLS +
    ['pw_template_corr', 'pw2_template_corr'] + PW_PCA_COLS
)

# 48-feature hybrid no-amp set (default for classifier)
FEATURE_COLS_48 = [f for f in FEATURE_COLS_59 if f not in DROPPED_AMP_FEATURES]


@dataclasses.dataclass
class PatientBaseline:
    """Frozen per-patient stats computed from the original (untransformed) signal."""
    # Per-patient QRS normalization: {feat: (mean, std)} for r_amplitude, qrs_width_ms, qrs_area
    norm_stats: dict
    # Per-patient P-wave normalization (N-beat stats): {feat: (mean, std)} for PW2_RAW_COLS
    pw_norm_stats: dict
    # Fixed-window P-wave N-beat template (shape PW_RESAMPLE_LEN,) or None
    pw_template_fixed: object
    # Adaptive-window P-wave N-beat template (shape PW_RESAMPLE_LEN,) or None
    pw_template_adaptive: object


# =============================================================================
# Signal Processing Functions
# =============================================================================

def bandpass_filter(signal, fs, lowcut=0.5, highcut=40.0, order=4):
    """Bandpass filter using a zero-phase Butterworth filter."""
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, signal)


def pan_tompkins_detect(raw_signal, fs, pre_filtered=False):
    """Pan-Tompkins R-peak detection with searchback on a raw MLII signal.

    Includes bandpass filtering internally unless pre_filtered=True.
    Returns (detected_peaks, filtered_signal).
    """
    filtered = raw_signal if pre_filtered else bandpass_filter(raw_signal, fs)
    diff = np.diff(filtered)
    squared = diff ** 2
    win_len = int(0.150 * fs)
    integrated = np.convolve(squared, np.ones(win_len) / win_len, mode='same')
    min_distance = int(0.2 * fs)
    peaks, _ = find_peaks(integrated, distance=min_distance)

    if len(peaks) == 0:
        return np.array([], dtype=int), filtered

    spki = np.mean(sorted(integrated[peaks])[-5:])
    npki = np.mean(sorted(integrated[peaks])[:5])
    threshold_i1 = npki + 0.25 * (spki - npki)
    threshold_i2 = 0.5 * threshold_i1

    rr_history = []
    rr_average = fs
    rr_missed_limit = 1.66 * rr_average
    signal_peaks = []

    i = 0
    while i < len(peaks):
        p = peaks[i]
        if integrated[p] > threshold_i1:
            signal_peaks.append(p)
            spki = 0.875 * spki + 0.125 * integrated[p]
            if len(signal_peaks) >= 2:
                rr = signal_peaks[-1] - signal_peaks[-2]
                rr_history.append(rr)
                if len(rr_history) > 8:
                    rr_history.pop(0)
                rr_average = np.mean(rr_history)
                rr_missed_limit = 1.66 * rr_average
        else:
            npki = 0.875 * npki + 0.125 * integrated[p]

        if len(signal_peaks) >= 1:
            gap = p - signal_peaks[-1]
            if gap > rr_missed_limit:
                search_start = signal_peaks[-1] + int(0.2 * fs)
                search_end = p
                gap_candidates = [
                    pk for pk in peaks
                    if search_start < pk < search_end
                    and integrated[pk] > threshold_i2
                    and pk not in signal_peaks
                ]
                if gap_candidates:
                    best = max(gap_candidates, key=lambda pk: integrated[pk])
                    signal_peaks.append(best)
                    signal_peaks.sort()
                    spki = 0.75 * spki + 0.25 * integrated[best]
                    if len(signal_peaks) >= 2:
                        rr = signal_peaks[-1] - signal_peaks[-2]
                        rr_history.append(rr)
                        if len(rr_history) > 8:
                            rr_history.pop(0)
                        rr_average = np.mean(rr_history)
                        rr_missed_limit = 1.66 * rr_average

        threshold_i1 = npki + 0.25 * (spki - npki)
        threshold_i2 = 0.5 * threshold_i1
        i += 1

    detected = np.array(signal_peaks, dtype=int)
    search_window = int(0.075 * fs)
    corrected = []
    for p in detected:
        lo = max(0, p - search_window)
        hi = min(len(filtered), p + search_window)
        corrected.append(lo + np.argmax(np.abs(filtered[lo:hi])))
    corrected = np.unique(np.array(corrected, dtype=int))
    return corrected, filtered


def extract_beat_features(signal, fs, r_peaks, labels):
    """Extract per-beat features from a filtered MLII signal using ground truth R-peaks."""
    features = []
    qrs_half = int(0.100 * fs)
    beat_pre = int(0.250 * fs)
    beat_post = int(0.150 * fs)
    rr_intervals = np.diff(r_peaks) / fs

    for i in range(1, len(r_peaks) - 1):
        r = r_peaks[i]
        label = labels[i]
        if label not in BEAT_SYMBOLS:
            continue

        rr_prev = (r_peaks[i] - r_peaks[i - 1]) / fs
        rr_curr = (r_peaks[i + 1] - r_peaks[i]) / fs
        if rr_prev < 0.2 or rr_prev > 2.5 or rr_curr < 0.2 or rr_curr > 2.5:
            continue

        rr_ratio = rr_curr / rr_prev
        hr_inst = 60.0 / rr_prev
        start_idx = max(0, i - 10)
        local_rr = rr_intervals[start_idx:i]
        rr_local_mean = np.mean(local_rr) if len(local_rr) > 0 else rr_prev
        rr_prev_ratio_mean = rr_prev / rr_local_mean
        rr_post_ratio_mean = rr_curr / rr_local_mean
        comp_pause_ratio = (rr_prev + rr_curr) / (2 * rr_local_mean)

        # Robust local RR references (resistant to S-beat-dominated windows)
        rr_local_median = np.median(local_rr) if len(local_rr) > 0 else rr_prev
        rr_local_max = np.max(local_rr) if len(local_rr) > 0 else rr_prev
        rr_local_std_val = np.std(local_rr) if len(local_rr) > 1 else 0.0
        rr_prev_ratio_median = rr_prev / rr_local_median
        rr_post_ratio_median = rr_curr / rr_local_median
        comp_pause_ratio_median = (rr_prev + rr_curr) / (2 * rr_local_median)
        rr_prev_ratio_max = rr_prev / rr_local_max
        rr_post_ratio_max = rr_curr / rr_local_max

        lo = max(0, r - qrs_half)
        hi = min(len(signal), r + qrs_half)
        qrs_segment = signal[lo:hi]
        if len(qrs_segment) < qrs_half:
            continue

        r_amp = signal[r]
        threshold = 0.5 * np.max(np.abs(qrs_segment))
        qrs_width_samples = np.sum(np.abs(qrs_segment) > threshold)
        qrs_width_ms = (qrs_width_samples / fs) * 1000

        beat_lo = r - beat_pre
        beat_hi = r + beat_post
        if beat_lo < 0:
            beat_segment = np.concatenate([np.zeros(-beat_lo), signal[0:beat_hi]])
        elif beat_hi > len(signal):
            beat_segment = np.concatenate([signal[beat_lo:len(signal)], np.zeros(beat_hi - len(signal))])
        else:
            beat_segment = signal[beat_lo:beat_hi].copy()

        beat_std = np.std(beat_segment)
        if beat_std > 0:
            beat_segment = (beat_segment - np.mean(beat_segment)) / beat_std
        else:
            beat_segment = beat_segment - np.mean(beat_segment)

        features.append({
            'label': label, 'sample_idx': r,
            'rr_prev': rr_prev, 'rr_curr': rr_curr, 'rr_ratio': rr_ratio,
            'hr_inst': hr_inst,
            'rr_prev_ratio_mean': rr_prev_ratio_mean,
            'rr_post_ratio_mean': rr_post_ratio_mean,
            'comp_pause_ratio': comp_pause_ratio,
            'rr_local_std': rr_local_std_val,
            'rr_prev_ratio_median': rr_prev_ratio_median,
            'rr_post_ratio_median': rr_post_ratio_median,
            'comp_pause_ratio_median': comp_pause_ratio_median,
            'rr_prev_ratio_max': rr_prev_ratio_max,
            'rr_post_ratio_max': rr_post_ratio_max,
            'r_amplitude': r_amp, 'qrs_width_ms': qrs_width_ms,
            'qrs_area': np.trapezoid(np.abs(qrs_segment)) / fs,
            'qrs_max': np.max(qrs_segment), 'qrs_min': np.min(qrs_segment),
            'qrs_range': np.max(qrs_segment) - np.min(qrs_segment),
            'qrs_skew': skew(qrs_segment), 'qrs_kurt': kurtosis(qrs_segment),
            'beat_waveform': beat_segment,
        })

    return pd.DataFrame(features)


# =============================================================================
# P-wave Feature Extraction
# =============================================================================

def extract_pwave_fixed(signal, fs, r_peaks):
    """Extract fixed-window P-wave waveforms for template correlation.

    Window: 200ms to 80ms before each R-peak. Resampled to PW_RESAMPLE_LEN.

    Returns:
        list of np.ndarray (PW_RESAMPLE_LEN) or None per R-peak
    """
    pw_start_offset = int(0.200 * fs)
    pw_end_offset = int(0.080 * fs)
    waveforms = []

    for i in range(len(r_peaks)):
        r = r_peaks[i]
        pw_start = r - pw_start_offset
        pw_end = r - pw_end_offset

        if i > 0:
            prev_qrs_end = r_peaks[i - 1] + int(0.100 * fs)
            if pw_start < prev_qrs_end:
                pw_start = prev_qrs_end

        if pw_start < 0 or pw_end >= len(signal) or pw_end <= pw_start or (pw_end - pw_start) < 5:
            waveforms.append(None)
            continue

        pw_segment = signal[pw_start:pw_end]
        waveforms.append(sig_resample(pw_segment, PW_RESAMPLE_LEN))

    return waveforms


def extract_pwave_adaptive(signal, fs, r_peaks):
    """Extract adaptive-window P-wave features and waveforms.

    Window: 25% to 10% of preceding RR interval before R-peak.

    Returns:
        DataFrame with PW2_RAW_COLS + 'pw2_waveform', one row per R-peak
    """
    features = []
    nan_row = {k: np.nan for k in PW2_RAW_COLS}

    for i in range(len(r_peaks)):
        r = r_peaks[i]
        rr_samples = (r - r_peaks[i - 1]) if i > 0 else int(0.8 * fs)

        pw_start = r - int(0.25 * rr_samples)
        pw_end = r - int(0.10 * rr_samples)

        if i > 0:
            prev_qrs_end = r_peaks[i - 1] + int(0.100 * fs)
            if pw_start < prev_qrs_end:
                pw_start = prev_qrs_end

        min_window = int(0.015 * fs)
        if pw_start < 0 or pw_end >= len(signal) or pw_end <= pw_start or (pw_end - pw_start) < min_window:
            feat = nan_row.copy()
            feat['pw2_waveform'] = None
            features.append(feat)
            continue

        pw_segment = signal[pw_start:pw_end]
        n_edge = max(1, min(3, len(pw_segment) // 3))
        baseline = (np.mean(pw_segment[:n_edge]) + np.mean(pw_segment[-n_edge:])) / 2
        pw_detrended = pw_segment - baseline

        peaks_found, props = find_peaks(pw_detrended, prominence=0.01)
        has_peak = len(peaks_found) > 0
        if has_peak:
            best = np.argmax(props['prominences'])
            peak_amp = pw_detrended[peaks_found[best]]
            peak_prom = props['prominences'][best]
        else:
            peak_amp = float(np.max(pw_detrended))
            peak_prom = 0.0

        features.append({
            'pw2_peak_amplitude': peak_amp,
            'pw2_energy': float(np.sum(pw_detrended ** 2) / len(pw_detrended)),
            'pw2_peak_prominence': peak_prom,
            'pw2_has_peak': float(has_peak),
            'pw2_baseline_dev': float(np.mean(np.abs(pw_detrended))),
            'pw2_max': float(np.max(pw_segment)),
            'pw2_min': float(np.min(pw_segment)),
            'pw2_range': float(np.max(pw_segment) - np.min(pw_segment)),
            'pw2_area': float(np.trapezoid(pw_detrended) / fs),
            'pw2_waveform': sig_resample(pw_segment, PW_RESAMPLE_LEN),
        })

    return pd.DataFrame(features)


def compute_pwave_template_corr(waveforms, template):
    """Compute correlation of each P-wave waveform against a template.

    Args:
        waveforms: list of np.ndarray (PW_RESAMPLE_LEN) or None
        template: np.ndarray (PW_RESAMPLE_LEN)

    Returns:
        np.ndarray of correlation values (0.0 for invalid waveforms)
    """
    template_c = template - np.mean(template)
    template_std = np.std(template_c)
    if template_std == 0:
        return np.zeros(len(waveforms))

    result = np.zeros(len(waveforms))
    for i, wf in enumerate(waveforms):
        if wf is not None and hasattr(wf, '__len__') and len(wf) >= 5:
            if len(wf) != PW_RESAMPLE_LEN:
                wf = sig_resample(wf, PW_RESAMPLE_LEN)
            beat_c = wf - np.mean(wf)
            if np.std(beat_c) > 0:
                result[i] = np.corrcoef(template_c, beat_c)[0, 1]
    return result


def _build_pwave_template(waveforms, min_beats=10):
    """Build a mean P-wave template from a list of waveforms.

    Returns the template array or None if insufficient valid waveforms.
    """
    valid = []
    for wf in waveforms:
        if wf is not None and hasattr(wf, '__len__'):
            if len(wf) == PW_RESAMPLE_LEN:
                valid.append(wf)
            elif len(wf) >= 5:
                valid.append(sig_resample(wf, PW_RESAMPLE_LEN))
    if len(valid) < min_beats:
        return None
    return np.mean(valid, axis=0)


def select_on_time_beats(rr_prev, threshold=0.8):
    """Return boolean mask selecting beats with on-time RR intervals.

    Excludes premature beats (short rr_prev) without using labels.
    Uses per-patient median RR as reference — beats below threshold * median
    are considered premature (likely ectopic).

    Args:
        rr_prev: array of preceding RR intervals (seconds)
        threshold: fraction of median RR below which beats are excluded

    Returns:
        boolean mask (True = on-time beat, use for template)
    """
    rr = np.asarray(rr_prev, dtype=float)
    valid = ~np.isnan(rr) & (rr > 0)
    if valid.sum() == 0:
        return np.ones(len(rr), dtype=bool)
    median_rr = np.median(rr[valid])
    return valid & (rr >= threshold * median_rr)


def compute_patient_baseline(df_features, signal, fs, r_peaks, labels):
    """Compute frozen per-patient baseline from the original record.

    Args:
        df_features: DataFrame from extract_beat_features (needs r_amplitude,
            qrs_width_ms, qrs_area columns)
        signal: filtered ECG signal
        fs: sampling frequency
        r_peaks: R-peak sample indices
        labels: clinical labels per beat ('N', 'S', 'V'), same length as r_peaks

    Returns:
        PatientBaseline with frozen normalization stats and templates
    """
    # QRS normalization stats (all beats)
    norm_stats = {}
    for feat in ['r_amplitude', 'qrs_width_ms', 'qrs_area']:
        m = float(df_features[feat].mean())
        s = float(df_features[feat].std())
        if s == 0 or np.isnan(s):
            s = 1.0
        norm_stats[feat] = (m, s)

    # Extract P-wave waveforms from the original signal
    pw_fixed = extract_pwave_fixed(signal, fs, r_peaks)
    df_pw2 = extract_pwave_adaptive(signal, fs, r_peaks)

    # Identify N-beat indices (aligned to the feature DataFrame rows)
    # df_features may have fewer rows than r_peaks (skipped beats), so align by sample_idx
    feat_samples = set(df_features['sample_idx'].values.astype(int))
    n_labels = np.array(labels)

    # Build index mapping: position in r_peaks -> is valid N-beat
    n_mask_full = np.array([
        (n_labels[i] == 'N' if i < len(n_labels) else False)
        and (int(r_peaks[i]) in feat_samples)
        for i in range(len(r_peaks))
    ])

    # Fixed-window template from N-beats
    n_fixed_waveforms = [pw_fixed[i] for i in range(len(r_peaks)) if n_mask_full[i]]
    pw_template_fixed = _build_pwave_template(n_fixed_waveforms)

    # Adaptive-window template from N-beats
    pw2_waveforms = df_pw2['pw2_waveform'].tolist()
    n_adaptive_waveforms = [pw2_waveforms[i] for i in range(len(r_peaks)) if n_mask_full[i]]
    pw_template_adaptive = _build_pwave_template(n_adaptive_waveforms)

    # P-wave N-beat normalization stats
    # Filter df_pw2 to valid N-beats (rows where corresponding r_peak is N and in df_features)
    valid_n_rows = [i for i in range(len(r_peaks)) if n_mask_full[i] and i < len(df_pw2)]
    pw_norm_stats = {}
    for feat in PW2_RAW_COLS:
        vals = df_pw2[feat].iloc[valid_n_rows].dropna()
        if len(vals) > 1:
            m = float(vals.mean())
            s = float(vals.std())
            if s == 0:
                s = 1.0
        else:
            m, s = 0.0, 1.0
        pw_norm_stats[feat] = (m, s)

    return PatientBaseline(
        norm_stats=norm_stats,
        pw_norm_stats=pw_norm_stats,
        pw_template_fixed=pw_template_fixed,
        pw_template_adaptive=pw_template_adaptive,
    )


# =============================================================================
# Data Loading
# =============================================================================

def build_df_all(data_dir=DATA_DIR, paced_records=PACED_RECORDS,
                 n_pca_components=10, pca_fit_records=DS1_RECORDS,
                 verbose=True, keep_waveforms=False):
    """Process all MIT-BIH records into a single feature DataFrame.

    Runs the full pipeline: record loading → Pan-Tompkins → feature extraction →
    label mapping → per-patient normalization → PCA → P-wave features.

    Returns:
        df_all: DataFrame with 48+ features + clinical_label + record columns
        pca: fitted QRS PCA model
        pw_pca: fitted P-wave PCA model
    """
    all_records = open(f'{data_dir}/RECORDS').read().strip().split('\n')
    all_features = []
    # Store per-record P-wave data for post-concat processing
    pw_fixed_per_rec = {}   # rec_id -> list of waveforms
    pw_adaptive_per_rec = {}  # rec_id -> DataFrame

    for rec_id in all_records:
        if rec_id in paced_records:
            if verbose:
                print(f"Record {rec_id}: SKIPPED (paced)")
            continue
        try:
            record = wfdb.rdrecord(f'{data_dir}/{rec_id}')
            ann = wfdb.rdann(f'{data_dir}/{rec_id}', 'atr')
            _, filtered = pan_tompkins_detect(record.p_signal[:, 0], record.fs)
            df_rec = extract_beat_features(filtered, record.fs, ann.sample, ann.symbol)
            df_rec['record'] = rec_id

            # P-wave extraction (aligned to ann.sample, same length)
            pw_fixed = extract_pwave_fixed(filtered, record.fs, ann.sample)
            df_pw2 = extract_pwave_adaptive(filtered, record.fs, ann.sample)

            # Map P-wave data to the beats that survived extract_beat_features
            # df_rec has sample_idx column; pw_fixed/df_pw2 are indexed by r_peaks position
            sample_to_pw_idx = {int(ann.sample[i]): i for i in range(len(ann.sample))}
            pw_fixed_aligned = []
            pw2_rows = []
            for _, row in df_rec.iterrows():
                pw_i = sample_to_pw_idx.get(int(row['sample_idx']))
                if pw_i is not None and pw_i < len(pw_fixed):
                    pw_fixed_aligned.append(pw_fixed[pw_i])
                else:
                    pw_fixed_aligned.append(None)
                if pw_i is not None and pw_i < len(df_pw2):
                    pw2_rows.append(df_pw2.iloc[pw_i])
                else:
                    pw2_rows.append(pd.Series({k: np.nan for k in PW2_RAW_COLS} | {'pw2_waveform': None}))

            df_rec['_pw_fixed_waveform'] = pw_fixed_aligned
            df_pw2_aligned = pd.DataFrame(pw2_rows).reset_index(drop=True)
            for col in PW2_RAW_COLS:
                df_rec[col] = df_pw2_aligned[col].values
            df_rec['_pw2_waveform'] = df_pw2_aligned['pw2_waveform'].values

            all_features.append(df_rec)
            if verbose:
                print(f"Record {rec_id}: {len(df_rec)} beats")
        except Exception as e:
            if verbose:
                print(f"Record {rec_id}: FAILED — {e}")

    df_all = pd.concat(all_features, ignore_index=True)

    # Label mapping
    df_all['clinical_label'] = df_all['label'].map(LABEL_MAP)
    df_all = df_all[df_all['clinical_label'].notna()].copy()

    # Drop rows with NaN P-wave features
    valid_pw = df_all[PW2_RAW_COLS].notna().all(axis=1)
    df_all = df_all[valid_pw].copy()

    # Per-patient normalized morphology features (QRS)
    for feat in ['r_amplitude', 'qrs_width_ms', 'qrs_area']:
        group_mean = df_all.groupby('record')[feat].transform('mean')
        group_std = df_all.groupby('record')[feat].transform('std').replace(0, 1)
        df_all[f'{feat}_norm'] = (df_all[feat] - group_mean) / group_std

    # QRS PCA on beat waveforms (fit on pca_fit_records only)
    waveform_matrix = np.stack(df_all['beat_waveform'].values)
    fit_mask = df_all['record'].isin(pca_fit_records)
    pca = PCA(n_components=n_pca_components)
    pca.fit(waveform_matrix[fit_mask])
    pca_features = pca.transform(waveform_matrix)
    for i in range(n_pca_components):
        df_all[f'pca_{i}'] = pca_features[:, i]
    if not keep_waveforms:
        df_all = df_all.drop(columns=['beat_waveform'])

    # --- P-wave template correlation (per-patient, using ground truth labels) ---
    df_all['pw_template_corr'] = 0.0
    df_all['pw2_template_corr'] = 0.0

    for rec in df_all['record'].unique():
        rec_mask = df_all['record'] == rec
        n_mask = rec_mask & (df_all['clinical_label'] == 'N')

        # Fixed-window template
        n_fixed_wf = df_all.loc[n_mask, '_pw_fixed_waveform'].tolist()
        template_fixed = _build_pwave_template(n_fixed_wf)
        if template_fixed is not None:
            all_fixed_wf = df_all.loc[rec_mask, '_pw_fixed_waveform'].tolist()
            df_all.loc[rec_mask, 'pw_template_corr'] = compute_pwave_template_corr(
                all_fixed_wf, template_fixed)

        # Adaptive-window template
        n_adaptive_wf = df_all.loc[n_mask, '_pw2_waveform'].tolist()
        template_adaptive = _build_pwave_template(n_adaptive_wf)
        if template_adaptive is not None:
            all_adaptive_wf = df_all.loc[rec_mask, '_pw2_waveform'].tolist()
            df_all.loc[rec_mask, 'pw2_template_corr'] = compute_pwave_template_corr(
                all_adaptive_wf, template_adaptive)

    # --- P-wave per-patient normalization (N-beat z-score) ---
    n_beat_mask = df_all['clinical_label'] == 'N'
    for feat in PW2_RAW_COLS:
        n_stats = df_all[n_beat_mask].groupby('record')[feat].agg(['mean', 'std'])
        n_stats['std'] = n_stats['std'].replace(0, 1)

        df_all[f'{feat}_pnorm'] = 0.0
        for rec in df_all['record'].unique():
            if rec in n_stats.index:
                rec_mask = df_all['record'] == rec
                df_all.loc[rec_mask, f'{feat}_pnorm'] = (
                    (df_all.loc[rec_mask, feat] - n_stats.loc[rec, 'mean'])
                    / n_stats.loc[rec, 'std']
                )

    # --- P-wave PCA (fit on DS1 N-beats) ---
    ds1_n_mask = df_all['record'].isin(pca_fit_records) & (df_all['clinical_label'] == 'N')
    ds1_n_wf = []
    for wf in df_all.loc[ds1_n_mask, '_pw2_waveform']:
        if wf is not None and hasattr(wf, '__len__'):
            if len(wf) == PW_RESAMPLE_LEN:
                ds1_n_wf.append(wf)
            elif len(wf) >= 5:
                ds1_n_wf.append(sig_resample(wf, PW_RESAMPLE_LEN))

    pw_pca = PCA(n_components=N_PW_PCA, random_state=42)
    pw_pca.fit(np.array(ds1_n_wf))

    # Transform all beats
    all_pw_wf = []
    for wf in df_all['_pw2_waveform']:
        if wf is not None and hasattr(wf, '__len__') and len(wf) >= 5:
            if len(wf) != PW_RESAMPLE_LEN:
                wf = sig_resample(wf, PW_RESAMPLE_LEN)
            all_pw_wf.append(wf)
        else:
            all_pw_wf.append(np.zeros(PW_RESAMPLE_LEN))

    pw_pca_features = pw_pca.transform(np.array(all_pw_wf))
    for i in range(N_PW_PCA):
        df_all[f'pw_pca_{i}'] = pw_pca_features[:, i]

    # Fill any remaining NaN in pnorm/PCA with 0
    for col in PW2_PNORM_COLS + PW_PCA_COLS:
        df_all[col] = df_all[col].fillna(0.0)

    # Drop temporary waveform columns
    if not keep_waveforms:
        df_all = df_all.drop(columns=['_pw_fixed_waveform', '_pw2_waveform'], errors='ignore')

    if verbose:
        print(f"\nTotal: {len(df_all)} beats")
        print(f"Clinical label distribution:")
        print(df_all['clinical_label'].value_counts().to_string())
        print(f"QRS PCA variance: {sum(pca.explained_variance_ratio_)*100:.1f}%")
        print(f"P-wave PCA variance: {sum(pw_pca.explained_variance_ratio_)*100:.1f}%")

    return df_all, pca, pw_pca


def get_train_test_split(df_all):
    """Split df_all into DS1 (train) and DS2 (test) with cascade label columns.

    Returns:
        df_train: DS1 records with ectopic_label and v_label columns
        df_test: DS2 records with ectopic_label and v_label columns
    """
    train_mask = df_all['record'].isin(DS1_RECORDS)
    test_mask = df_all['record'].isin(DS2_RECORDS)
    df_train = df_all[train_mask].copy()
    df_test = df_all[test_mask].copy()

    # Cascade labels
    label_to_ectopic = {'N': 'Normal', 'S': 'Ectopic', 'V': 'Ectopic'}
    label_to_v = {'N': 'Non-V', 'S': 'Non-V', 'V': 'V'}

    df_train['ectopic_label'] = df_train['clinical_label'].map(label_to_ectopic)
    df_test['ectopic_label'] = df_test['clinical_label'].map(label_to_ectopic)
    df_train['v_label'] = df_train['clinical_label'].map(label_to_v)
    df_test['v_label'] = df_test['clinical_label'].map(label_to_v)

    return df_train, df_test


# =============================================================================
# Session-Level Aggregation
# =============================================================================

def compute_hrv_frequency(nn_intervals_s):
    """Compute HRV frequency-domain metrics from N-N intervals.

    Uses cubic spline interpolation to uniform 4 Hz, linear detrend,
    and Welch PSD estimation.

    Args:
        nn_intervals_s: array of N-N intervals in seconds

    Returns:
        dict with vlf_power_s2, lf_power_s2, hf_power_s2, lf_hf_ratio, total_power_s2
    """
    result = {
        'vlf_power_s2': np.nan, 'lf_power_s2': np.nan, 'hf_power_s2': np.nan,
        'lf_hf_ratio': np.nan, 'total_power_s2': np.nan,
    }

    if len(nn_intervals_s) < 30:
        return result

    t_nn = np.cumsum(nn_intervals_s)
    t_nn = t_nn - t_nn[0]

    fs_interp = 4.0
    t_uniform = np.arange(t_nn[0], t_nn[-1], 1.0 / fs_interp)

    if len(t_uniform) < 64:
        return result

    cs = CubicSpline(t_nn, nn_intervals_s)
    nn_uniform = cs(t_uniform)
    nn_uniform = nn_uniform - np.polyval(
        np.polyfit(t_uniform, nn_uniform, 1), t_uniform)

    nperseg = min(256, len(nn_uniform))
    freqs, psd = welch(nn_uniform, fs=fs_interp, nperseg=nperseg,
                       noverlap=nperseg // 2, window='hann')

    vlf_mask = (freqs >= 0.003) & (freqs < 0.04)
    lf_mask = (freqs >= 0.04) & (freqs < 0.15)
    hf_mask = (freqs >= 0.15) & (freqs < 0.4)

    if vlf_mask.sum() > 1:
        result['vlf_power_s2'] = np.trapezoid(psd[vlf_mask], freqs[vlf_mask])
    if lf_mask.sum() > 1:
        result['lf_power_s2'] = np.trapezoid(psd[lf_mask], freqs[lf_mask])
    if hf_mask.sum() > 1:
        result['hf_power_s2'] = np.trapezoid(psd[hf_mask], freqs[hf_mask])
    if result['hf_power_s2'] and result['hf_power_s2'] > 0:
        result['lf_hf_ratio'] = result['lf_power_s2'] / result['hf_power_s2']
    total_mask = (freqs >= 0.003) & (freqs < 0.4)
    if total_mask.sum() > 1:
        result['total_power_s2'] = np.trapezoid(psd[total_mask], freqs[total_mask])

    return result


def compute_session_metrics(df_session, fs=360, label_col='clinical_label'):
    """Compute all session-level metrics from a single record's beat DataFrame.

    Args:
        df_session: DataFrame with per-beat features for one record
        fs: sampling frequency
        label_col: column name for beat labels

    Returns:
        dict of session metrics
    """
    total = len(df_session)
    if total == 0:
        return {}

    labels = df_session[label_col]

    n_count = int((labels == 'N').sum())
    s_count = int((labels == 'S').sum())
    v_count = int((labels == 'V').sum())

    pvc_burden_pct = 100.0 * v_count / total
    sve_burden_pct = 100.0 * s_count / total
    ectopic_burden_pct = pvc_burden_pct + sve_burden_pct

    if 'sample_idx' in df_session.columns and total >= 2:
        duration_s = (df_session['sample_idx'].iloc[-1]
                      - df_session['sample_idx'].iloc[0]) / fs
    else:
        duration_s = total * df_session['rr_prev'].median()
    duration_h = duration_s / 3600.0

    pvc_per_hour = v_count / duration_h if duration_h > 0 else 0.0
    sve_per_hour = s_count / duration_h if duration_h > 0 else 0.0

    # N-N intervals for HRV
    prev_label = labels.shift(1)
    nn_mask = (labels == 'N') & (prev_label == 'N')
    nn_intervals = df_session.loc[nn_mask, 'rr_prev'].values
    nn_intervals = nn_intervals[(nn_intervals > 0.3) & (nn_intervals < 2.0)]

    if len(nn_intervals) >= 10:
        mean_nn_s = np.mean(nn_intervals)
        mean_hr_bpm = 60.0 / mean_nn_s
        sdnn_ms = np.std(nn_intervals, ddof=1) * 1000.0
        nn_diffs = np.diff(nn_intervals)
        rmssd_ms = np.sqrt(np.mean(nn_diffs ** 2)) * 1000.0
        pnn50_pct = 100.0 * np.sum(np.abs(nn_diffs) > 0.050) / len(nn_diffs)
    else:
        mean_hr_bpm = np.nan
        sdnn_ms = np.nan
        rmssd_ms = np.nan
        pnn50_pct = np.nan

    freq_metrics = compute_hrv_frequency(nn_intervals)

    n_beats = df_session[labels == 'N']
    if len(n_beats) > 0:
        mean_qrs_width_ms = n_beats['qrs_width_ms'].mean()
        std_qrs_width_ms = n_beats['qrs_width_ms'].std()
        mean_r_amplitude = n_beats['r_amplitude'].mean()
    else:
        mean_qrs_width_ms = np.nan
        std_qrs_width_ms = np.nan
        mean_r_amplitude = np.nan

    all_rr = df_session['rr_prev'].values
    pause_count_2s = int(np.sum(all_rr > 2.0))
    pause_count_3s = int(np.sum(all_rr > 3.0))
    max_rr_interval_s = float(np.max(all_rr)) if len(all_rr) > 0 else np.nan

    return {
        'total_beats': total,
        'n_count': n_count, 's_count': s_count, 'v_count': v_count,
        'duration_min': duration_s / 60.0,
        'pvc_burden_pct': pvc_burden_pct,
        'sve_burden_pct': sve_burden_pct,
        'ectopic_burden_pct': ectopic_burden_pct,
        'pvc_per_hour': pvc_per_hour,
        'sve_per_hour': sve_per_hour,
        'mean_hr_bpm': mean_hr_bpm,
        'sdnn_ms': sdnn_ms,
        'rmssd_ms': rmssd_ms,
        'pnn50_pct': pnn50_pct,
        'nn_count': len(nn_intervals),
        **freq_metrics,
        'mean_qrs_width_ms': mean_qrs_width_ms,
        'std_qrs_width_ms': std_qrs_width_ms,
        'mean_r_amplitude': mean_r_amplitude,
        'pause_count_2s': pause_count_2s,
        'pause_count_3s': pause_count_3s,
        'max_rr_interval_s': max_rr_interval_s,
    }


# =============================================================================
# Single-Record Pipeline (for dashboard / real-time use)
# =============================================================================

def process_single_record(signal, fs, pca, pw_pca=None, baseline=None,
                          keep_waveforms=False):
    """Run full pipeline on a single ECG signal.

    Detects R-peaks via Pan-Tompkins, extracts features, applies normalization
    and PCA. If pw_pca and baseline are provided, also extracts P-wave features
    using frozen baseline stats for normalization and template correlation.

    Args:
        signal: ECG signal (already bandpass filtered)
        fs: sampling frequency
        pca: fitted QRS PCA model
        pw_pca: fitted P-wave PCA model, or None (skip P-wave features)
        baseline: PatientBaseline with frozen stats, or None (use session-local)
        keep_waveforms: if True, retain beat_waveform column (for CNN models)

    Returns:
        DataFrame with features ready for classification
    """
    detected_peaks, _ = pan_tompkins_detect(signal, fs, pre_filtered=True)

    if len(detected_peaks) < 3:
        return pd.DataFrame()

    dummy_labels = ['N'] * len(detected_peaks)
    df = extract_beat_features(signal, fs, detected_peaks, dummy_labels)

    if len(df) == 0:
        return df

    # Per-patient QRS normalization (frozen baseline or session-local)
    if baseline is not None:
        for feat in ['r_amplitude', 'qrs_width_ms', 'qrs_area']:
            m, s = baseline.norm_stats[feat]
            df[f'{feat}_norm'] = (df[feat] - m) / s
    else:
        for feat in ['r_amplitude', 'qrs_width_ms', 'qrs_area']:
            m = df[feat].mean()
            s = df[feat].std()
            if s == 0 or np.isnan(s):
                s = 1.0
            df[f'{feat}_norm'] = (df[feat] - m) / s

    # QRS PCA on beat waveforms
    waveform_matrix = np.stack(df['beat_waveform'].values)
    pca_features = pca.transform(waveform_matrix)
    for i in range(pca.n_components):
        df[f'pca_{i}'] = pca_features[:, i]
    if not keep_waveforms:
        df = df.drop(columns=['beat_waveform'])

    # P-wave features (only if pw_pca provided)
    if pw_pca is not None:
        # Extract P-wave from detected peaks
        pw_fixed = extract_pwave_fixed(signal, fs, detected_peaks)
        df_pw2 = extract_pwave_adaptive(signal, fs, detected_peaks)

        # Map to surviving beats (extract_beat_features may skip some)
        sample_to_pw_idx = {int(detected_peaks[i]): i for i in range(len(detected_peaks))}
        pw_fixed_aligned = []
        pw2_rows = []
        for _, row in df.iterrows():
            pw_i = sample_to_pw_idx.get(int(row['sample_idx']))
            if pw_i is not None and pw_i < len(pw_fixed):
                pw_fixed_aligned.append(pw_fixed[pw_i])
            else:
                pw_fixed_aligned.append(None)
            if pw_i is not None and pw_i < len(df_pw2):
                pw2_rows.append(df_pw2.iloc[pw_i])
            else:
                pw2_rows.append(pd.Series(
                    {k: np.nan for k in PW2_RAW_COLS} | {'pw2_waveform': None}))

        df_pw2_aligned = pd.DataFrame(pw2_rows).reset_index(drop=True)

        # Merge raw P-wave features
        for col in PW2_RAW_COLS:
            df[col] = df_pw2_aligned[col].values
        # Fill NaN raw features with 0
        for col in PW2_RAW_COLS:
            df[col] = df[col].fillna(0.0)

        # Template correlation using frozen baseline templates
        if baseline is not None and baseline.pw_template_fixed is not None:
            df['pw_template_corr'] = compute_pwave_template_corr(
                pw_fixed_aligned, baseline.pw_template_fixed)
        else:
            df['pw_template_corr'] = 0.0

        if baseline is not None and baseline.pw_template_adaptive is not None:
            pw2_wf = df_pw2_aligned['pw2_waveform'].tolist()
            df['pw2_template_corr'] = compute_pwave_template_corr(
                pw2_wf, baseline.pw_template_adaptive)
        else:
            df['pw2_template_corr'] = 0.0

        # P-wave per-patient normalization (frozen baseline or session-local)
        if baseline is not None:
            for feat in PW2_RAW_COLS:
                m, s = baseline.pw_norm_stats[feat]
                df[f'{feat}_pnorm'] = (df[feat] - m) / s
        else:
            for feat in PW2_RAW_COLS:
                m = df[feat].mean()
                s = df[feat].std()
                if s == 0 or np.isnan(s):
                    s = 1.0
                df[f'{feat}_pnorm'] = (df[feat] - m) / s

        # P-wave PCA
        pw2_waveforms = df_pw2_aligned['pw2_waveform'].tolist()
        pw_wf_matrix = []
        for wf in pw2_waveforms:
            if wf is not None and hasattr(wf, '__len__') and len(wf) >= 5:
                if len(wf) != PW_RESAMPLE_LEN:
                    wf = sig_resample(wf, PW_RESAMPLE_LEN)
                pw_wf_matrix.append(wf)
            else:
                pw_wf_matrix.append(np.zeros(PW_RESAMPLE_LEN))
        pw_pca_features = pw_pca.transform(np.array(pw_wf_matrix))
        for i in range(pw_pca.n_components):
            df[f'pw_pca_{i}'] = pw_pca_features[:, i]

        # Fill any remaining NaN
        for col in PW2_PNORM_COLS + PW_PCA_COLS:
            if col in df.columns:
                df[col] = df[col].fillna(0.0)

    return df
