"""Signal transformations for simulating ECG disease progression.

Each transformation operates on beat segments extracted from the original signal.
Segments are cut at midpoints between R-peaks, transformed individually, then
reconstructed into a continuous signal for blind pipeline processing.

Transformations map to clinical conditions:
- PVC/SVE insertion: developing arrhythmia, atrial irritability
- QRS widening: progressive bundle branch block
- P-wave flattening: atrial disease, AF progression
- HR change: tachycardia/bradycardia from HF, infection, medication
- HRV compression: autonomic deterioration
- Pause insertion: AV conduction disease
- AF irregularity: atrial fibrillation onset
- Amplitude reduction: pericardial effusion, cardiomyopathy
"""

import numpy as np
from ecg_monitor.pipeline import LABEL_MAP


def segment_beats(signal, fs, r_peaks, labels):
    """Split signal into beat segments at midpoints between R-peaks.

    Args:
        signal: filtered ECG signal array
        fs: sampling frequency
        r_peaks: R-peak sample indices
        labels: beat labels (same length as r_peaks)

    Returns:
        list of dicts with 'waveform', 'label', 'r_offset' keys
    """
    segments = []
    for i in range(1, len(r_peaks) - 1):
        mid_left = max(0, (r_peaks[i - 1] + r_peaks[i]) // 2)
        mid_right = min(len(signal), (r_peaks[i] + r_peaks[i + 1]) // 2)

        if mid_right <= mid_left:
            continue

        segments.append({
            'waveform': signal[mid_left:mid_right].copy(),
            'label': labels[i] if i < len(labels) else 'N',
            'r_offset': r_peaks[i] - mid_left,
        })

    return segments


def reconstruct_signal(segments):
    """Concatenate beat segments into a continuous signal.

    Returns:
        (signal_array, r_peak_indices) tuple
    """
    if not segments:
        return np.array([]), np.array([], dtype=int)

    waveforms = []
    r_peaks = []
    offset = 0
    fade = 4  # crossfade samples at boundaries

    for i, s in enumerate(segments):
        wav = s['waveform'].copy()

        # Short crossfade to avoid discontinuities at segment boundaries
        if i > 0 and len(wav) > fade * 2:
            wav[:fade] *= np.linspace(0.5, 1.0, fade)
        if i < len(segments) - 1 and len(wav) > fade * 2:
            wav[-fade:] *= np.linspace(1.0, 0.5, fade)

        r_peaks.append(offset + s['r_offset'])
        waveforms.append(wav)
        offset += len(wav)

    return np.concatenate(waveforms), np.array(r_peaks, dtype=int)


def _clinical_label(raw_label):
    """Map raw beat label to clinical class (N/S/V)."""
    return LABEL_MAP.get(raw_label, raw_label)


def insert_ectopics(segments, beat_type, additional_rate, rng):
    """Replace random N beats with ectopic beat waveforms from the same patient.

    Template waveforms are resampled to match the target segment duration
    so timing structure is preserved.

    Args:
        segments: list of beat segment dicts
        beat_type: 'V' or 'S'
        additional_rate: fraction of total beats to add (e.g., 0.10 = 10%)
        rng: numpy random generator
    """
    if additional_rate <= 0:
        return segments

    templates = [s for s in segments if _clinical_label(s['label']) == beat_type]
    if not templates:
        return segments

    n_indices = [i for i, s in enumerate(segments)
                 if _clinical_label(s['label']) == 'N']

    n_to_replace = min(int(additional_rate * len(segments)), len(n_indices))
    if n_to_replace <= 0:
        return segments

    replace_set = set(rng.choice(n_indices, size=n_to_replace, replace=False))

    new_segments = []
    for i, s in enumerate(segments):
        if i in replace_set:
            template = templates[rng.integers(len(templates))]
            new_s = s.copy()
            old_len = len(template['waveform'])
            new_len = len(s['waveform'])
            new_s['waveform'] = np.interp(
                np.linspace(0, 1, new_len),
                np.linspace(0, 1, old_len),
                template['waveform'],
            )
            new_s['label'] = beat_type
            new_s['r_offset'] = int(template['r_offset'] * new_len / old_len)
            new_segments.append(new_s)
        else:
            new_segments.append(s)

    return new_segments


def widen_qrs(segments, fs, factor):
    """Stretch the QRS region (R-peak +/- 50ms) of each beat.

    Pre-QRS and post-QRS signal is preserved; the segment grows longer,
    naturally increasing QRS duration measured by the pipeline.

    Args:
        segments: list of beat segment dicts
        fs: sampling frequency
        factor: stretch factor (1.0 = no change, 2.0 = double width)
    """
    if factor <= 1.0:
        return segments

    qrs_half = int(0.050 * fs)

    new_segments = []
    for s in segments:
        seg = s['waveform']
        r = s['r_offset']

        qrs_start = max(0, r - qrs_half)
        qrs_end = min(len(seg), r + qrs_half)
        qrs = seg[qrs_start:qrs_end]

        if len(qrs) < 3:
            new_segments.append(s)
            continue

        new_qrs_len = int(len(qrs) * factor)
        stretched_qrs = np.interp(
            np.linspace(0, 1, new_qrs_len),
            np.linspace(0, 1, len(qrs)),
            qrs,
        )

        new_waveform = np.concatenate([
            seg[:qrs_start], stretched_qrs, seg[qrs_end:]
        ])
        new_r_offset = qrs_start + int((r - qrs_start) * factor)

        new_s = s.copy()
        new_s['waveform'] = new_waveform
        new_s['r_offset'] = new_r_offset
        new_segments.append(new_s)

    return new_segments


def flatten_pwave(segments, fs, attenuation):
    """Attenuate the P-wave region (200ms to 80ms before R-peak).

    Uses smooth cosine tapers at boundaries to avoid discontinuities.

    Args:
        segments: list of beat segment dicts
        fs: sampling frequency
        attenuation: 0.0 = no change, 1.0 = fully flatten
    """
    if attenuation <= 0:
        return segments

    pw_pre_samples = int(0.200 * fs)
    pw_post_samples = int(0.080 * fs)
    taper_len = max(2, int(0.015 * fs))
    scale = 1.0 - attenuation

    new_segments = []
    for s in segments:
        seg = s['waveform'].copy()
        r = s['r_offset']

        pw_start = max(0, r - pw_pre_samples)
        pw_end = max(0, r - pw_post_samples)

        if pw_end <= pw_start + taper_len:
            new_segments.append(s)
            continue

        mask = np.ones(len(seg))
        mask[pw_start:pw_end] = scale

        # Smooth taper in
        t_in_start = max(0, pw_start - taper_len)
        if pw_start > t_in_start:
            mask[t_in_start:pw_start] = np.linspace(1.0, scale,
                                                     pw_start - t_in_start)
        # Smooth taper out
        t_out_end = min(len(seg), pw_end + taper_len)
        if t_out_end > pw_end:
            mask[pw_end:t_out_end] = np.linspace(scale, 1.0,
                                                  t_out_end - pw_end)

        new_s = s.copy()
        new_s['waveform'] = seg * mask
        new_segments.append(new_s)

    return new_segments


def change_heart_rate(segments, factor):
    """Change heart rate by resampling all beat segments.

    factor < 1.0 = faster HR (shorter segments).
    factor > 1.0 = slower HR (longer segments).

    Args:
        segments: list of beat segment dicts
        factor: time-stretch factor
    """
    if abs(factor - 1.0) < 0.01:
        return segments

    new_segments = []
    for s in segments:
        old_len = len(s['waveform'])
        new_len = max(10, int(old_len * factor))

        new_s = s.copy()
        new_s['waveform'] = np.interp(
            np.linspace(0, 1, new_len),
            np.linspace(0, 1, old_len),
            s['waveform'],
        )
        new_s['r_offset'] = int(s['r_offset'] * factor)
        new_segments.append(new_s)

    return new_segments


def compress_hrv(segments, compression):
    """Compress RR interval variability toward the mean duration.

    compression = 0.0: no change (original variability).
    compression = 1.0: all segments same duration (zero variability).

    Args:
        segments: list of beat segment dicts
        compression: fraction of variability to remove (0-1)
    """
    if compression <= 0:
        return segments

    lengths = np.array([len(s['waveform']) for s in segments])
    mean_len = np.mean(lengths)

    new_segments = []
    for s in segments:
        old_len = len(s['waveform'])
        target_len = max(10, int(old_len + compression * (mean_len - old_len)))

        new_s = s.copy()
        new_s['waveform'] = np.interp(
            np.linspace(0, 1, target_len),
            np.linspace(0, 1, old_len),
            s['waveform'],
        )
        new_s['r_offset'] = int(s['r_offset'] * target_len / old_len)
        new_segments.append(new_s)

    return new_segments


def insert_pauses(segments, fs, n_pauses, rng):
    """Insert pause segments (baseline-level signal, no QRS) between beats.

    Simulates dropped beats / AV block. Pan-Tompkins won't detect a beat
    in the pause, creating a long RR interval that triggers pause detection.

    Args:
        segments: list of beat segment dicts
        fs: sampling frequency
        n_pauses: number of pauses to insert
        rng: numpy random generator
    """
    if n_pauses <= 0 or len(segments) < 5:
        return segments

    pause_duration = int(2.5 * fs)
    n_to_insert = min(n_pauses, len(segments) - 4)

    positions = sorted(
        rng.choice(range(2, len(segments) - 2), size=n_to_insert, replace=False),
        reverse=True,
    )

    new_segments = list(segments)
    for pos in positions:
        pause_waveform = rng.normal(0, 0.005, pause_duration).astype(np.float64)
        pause_seg = {
            'waveform': pause_waveform,
            'label': 'N',
            'r_offset': pause_duration // 2,
        }
        new_segments.insert(pos, pause_seg)

    return new_segments


def add_af_irregularity(segments, irregularity, rng):
    """Randomize RR intervals to simulate atrial fibrillation.

    Each segment is randomly stretched or compressed around its original
    duration. Higher irregularity = more variation = more "irregularly irregular".

    Args:
        segments: list of beat segment dicts
        irregularity: coefficient of variation (0.0 = regular, 0.5 = very irregular)
        rng: numpy random generator
    """
    if irregularity <= 0:
        return segments

    new_segments = []
    for s in segments:
        factor = np.clip(1.0 + rng.normal(0, irregularity), 0.5, 2.0)

        old_len = len(s['waveform'])
        new_len = max(10, int(old_len * factor))

        new_s = s.copy()
        new_s['waveform'] = np.interp(
            np.linspace(0, 1, new_len),
            np.linspace(0, 1, old_len),
            s['waveform'],
        )
        new_s['r_offset'] = int(s['r_offset'] * new_len / old_len)
        new_segments.append(new_s)

    return new_segments


def apply_transforms(signal, fs, r_peaks, labels, params):
    """Apply all transformations to an ECG signal.

    Transformations are applied in order:
    1. Ectopic insertion (modifies beat content)
    2. Morphology transforms (QRS widening, P-wave flattening)
    3. Timing transforms (HRV compression, pauses, AF, HR change)
    4. Reconstruction
    5. Global amplitude scaling

    Args:
        signal: filtered ECG signal
        fs: sampling frequency
        r_peaks: R-peak sample indices (ground truth)
        labels: beat labels (ground truth, same length as r_peaks)
        params: dict with keys:
            pvc_rate, sve_rate, qrs_width_factor, pwave_attenuation,
            hr_factor, hrv_compression, n_pauses, af_irregularity,
            amplitude_factor

    Returns:
        transformed signal array
    """
    rng = np.random.default_rng(42)

    segments = segment_beats(signal, fs, r_peaks, labels)
    if not segments:
        return signal.copy()

    # 1. Ectopic insertion
    if params.get('pvc_rate', 0) > 0:
        segments = insert_ectopics(segments, 'V', params['pvc_rate'], rng)
    if params.get('sve_rate', 0) > 0:
        segments = insert_ectopics(segments, 'S', params['sve_rate'], rng)

    # 2. Morphology transforms
    if params.get('qrs_width_factor', 1.0) > 1.0:
        segments = widen_qrs(segments, fs, params['qrs_width_factor'])
    if params.get('pwave_attenuation', 0) > 0:
        segments = flatten_pwave(segments, fs, params['pwave_attenuation'])

    # 3. Timing transforms
    if params.get('hrv_compression', 0) > 0:
        segments = compress_hrv(segments, params['hrv_compression'])
    if params.get('n_pauses', 0) > 0:
        segments = insert_pauses(segments, fs, int(params['n_pauses']), rng)
    if params.get('af_irregularity', 0) > 0:
        segments = add_af_irregularity(segments, params['af_irregularity'], rng)
    if abs(params.get('hr_factor', 1.0) - 1.0) >= 0.01:
        segments = change_heart_rate(segments, params['hr_factor'])

    # 4. Reconstruct continuous signal
    new_signal, _ = reconstruct_signal(segments)

    # 5. Global amplitude scaling
    amp = params.get('amplitude_factor', 1.0)
    if amp != 1.0:
        new_signal = new_signal * amp

    return new_signal
