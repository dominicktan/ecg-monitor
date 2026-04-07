"""Disease progression scenarios for longitudinal trend simulation.

Each scenario defines a sequence of transform parameters over N sessions,
simulating clinical disease progression. Scenarios map directly to the
transform parameters accepted by ``apply_transforms()`` in transforms.py.

Preset scenarios cover: HF decompensation, developing AF, conduction disease,
cardiomyopathy, and a stable control. Custom scenarios can be built by
specifying start/end values for any transform parameter — intermediate values
are linearly interpolated.
"""

from __future__ import annotations

import dataclasses
import numpy as np


# Transform parameter defaults (no transformation applied)
DEFAULT_PARAMS = {
    'pvc_rate': 0.0,
    'sve_rate': 0.0,
    'qrs_width_factor': 1.0,
    'pwave_attenuation': 0.0,
    'hr_factor': 1.0,
    'hrv_compression': 0.0,
    'n_pauses': 0,
    'af_irregularity': 0.0,
    'amplitude_factor': 1.0,
}


@dataclasses.dataclass
class Scenario:
    """A disease progression scenario for longitudinal simulation.

    Attributes:
        name: Human-readable scenario name.
        description: Clinical narrative explaining what this scenario simulates.
        n_sessions: Number of sessions (e.g. days) to simulate.
        base_record: MIT-BIH record ID to use as the patient's baseline ECG.
        param_schedule: Mapping of transform parameter name to a list of values,
            one per session.  Parameters not listed default to neutral values
            (no transformation).
    """

    name: str
    description: str
    n_sessions: int
    base_record: str
    param_schedule: dict[str, list[float]]

    def get_params(self, session_idx: int) -> dict:
        """Return the full transform param dict for a given session index."""
        params = dict(DEFAULT_PARAMS)
        for key, schedule in self.param_schedule.items():
            idx = min(session_idx, len(schedule) - 1)
            params[key] = schedule[idx]
        return params


# Valid ranges for each transform parameter (used for clipping noisy schedules)
PARAM_BOUNDS = {
    'pvc_rate': (0.0, 0.50),
    'sve_rate': (0.0, 0.50),
    'qrs_width_factor': (1.0, 2.5),
    'pwave_attenuation': (0.0, 1.0),
    'hr_factor': (0.5, 2.0),
    'hrv_compression': (0.0, 0.95),
    'n_pauses': (0.0, 15.0),
    'af_irregularity': (0.0, 0.50),
    'amplitude_factor': (0.2, 1.0),
}


def _noisy_schedule(
    start: float,
    end: float,
    n: int,
    noise_frac: float = 0.3,
    param_name: str | None = None,
    rng: np.random.Generator | None = None,
) -> list[float]:
    """Linearly interpolate with session-to-session noise.

    Adds Gaussian noise proportional to the total parameter range, so the
    overall trend is preserved but individual sessions vary realistically.

    Args:
        start: Starting value.
        end: Ending value.
        n: Number of sessions.
        noise_frac: Noise magnitude as a fraction of |end - start|.
            0.0 = perfectly linear, 0.3 = moderate day-to-day variation.
        param_name: If provided, clips to valid bounds from PARAM_BOUNDS.
        rng: NumPy random generator (for reproducibility).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    base = np.linspace(start, end, n)
    spread = abs(end - start)

    if spread > 0 and n > 1:
        noise = rng.normal(0, spread * noise_frac, size=n)
        # Keep first session clean (baseline) and last session near target
        noise[0] = 0.0
        noise[-1] *= 0.3  # dampen last session noise
        values = base + noise
    else:
        values = base

    # Clip to valid parameter bounds
    if param_name and param_name in PARAM_BOUNDS:
        lo, hi = PARAM_BOUNDS[param_name]
        values = np.clip(values, lo, hi)

    return values.tolist()


def build_custom_scenario(
    name: str,
    base_record: str,
    n_sessions: int,
    param_ranges: dict[str, tuple[float, float]],
    description: str = "Custom scenario",
) -> Scenario:
    """Build a custom scenario from start/end values for each parameter.

    Args:
        name: Scenario name.
        base_record: MIT-BIH record ID.
        n_sessions: Number of sessions.
        param_ranges: Dict of param_name -> (start_value, end_value).
            Values are linearly interpolated across sessions.
        description: Optional description.

    Returns:
        A Scenario with linearly interpolated param_schedule.
    """
    rng = np.random.default_rng(42)
    schedule = {}
    for param, (start, end) in param_ranges.items():
        schedule[param] = _noisy_schedule(
            start, end, n_sessions, param_name=param, rng=rng)
    return Scenario(
        name=name,
        description=description,
        n_sessions=n_sessions,
        base_record=base_record,
        param_schedule=schedule,
    )


# ============================================================================
# Preset scenarios
# ============================================================================

def _hf_decompensation(n: int = 50) -> Scenario:
    """Heart failure decompensation over ~2 weeks.

    Progressive tachycardia, declining HRV, and increasing PVC burden.
    Simulates a patient whose heart failure is worsening — fluid overload,
    neurohormonal activation, and increased ventricular irritability.
    """
    rng = np.random.default_rng(42)
    return Scenario(
        name="HF Decompensation",
        description=(
            "Simulates worsening heart failure: rising resting heart rate from "
            "neurohormonal activation, declining heart rate variability from "
            "autonomic deterioration, and increasing PVC burden from ventricular "
            "irritability.  Expect HR alerts first, followed by HRV and PVC alerts."
        ),
        n_sessions=n,
        base_record='119',  # clean sinus rhythm, some ectopy available
        param_schedule={
            'hr_factor': _noisy_schedule(1.0, 0.80, n, param_name='hr_factor', rng=rng),
            'hrv_compression': _noisy_schedule(0.0, 0.70, n, param_name='hrv_compression', rng=rng),
            'pvc_rate': _noisy_schedule(0.0, 0.12, n, param_name='pvc_rate', rng=rng),
        },
    )


def _developing_af(n: int = 50) -> Scenario:
    """Developing atrial fibrillation over ~2 weeks.

    Rising SVE burden, increasing RR irregularity, and P-wave flattening
    as atrial substrate deteriorates.
    """
    rng = np.random.default_rng(123)
    return Scenario(
        name="Developing AF",
        description=(
            "Simulates progression toward atrial fibrillation: increasing "
            "supraventricular ectopic burden (atrial irritability), progressive "
            "RR interval irregularity, and P-wave attenuation as organized atrial "
            "activity degrades.  Expect SVE burden alerts first, then HRV metrics "
            "becoming erratic as AF irregularity develops."
        ),
        n_sessions=n,
        base_record='100',
        param_schedule={
            'sve_rate': _noisy_schedule(0.0, 0.15, n, param_name='sve_rate', rng=rng),
            'af_irregularity': _noisy_schedule(0.0, 0.25, n, param_name='af_irregularity', rng=rng),
            'pwave_attenuation': _noisy_schedule(0.0, 0.80, n, param_name='pwave_attenuation', rng=rng),
        },
    )


def _conduction_disease(n: int = 50) -> Scenario:
    """Progressive conduction disease over ~2 weeks.

    QRS widening (bundle branch block), developing bradycardia, and
    emerging pauses (AV block).
    """
    rng = np.random.default_rng(456)
    return Scenario(
        name="Conduction Disease",
        description=(
            "Simulates progressive conduction system disease: QRS duration widens "
            "(developing bundle branch block), heart rate slows (AV node "
            "degeneration), and pauses emerge (intermittent AV block).  Expect "
            "QRS width alerts early, followed by HR and pause alerts."
        ),
        n_sessions=n,
        base_record='100',
        param_schedule={
            'qrs_width_factor': _noisy_schedule(1.0, 1.6, n, param_name='qrs_width_factor', rng=rng),
            'hr_factor': _noisy_schedule(1.0, 1.25, n, param_name='hr_factor', rng=rng),
            'n_pauses': ([0.0] * 5
                         + _noisy_schedule(0, 6, n - 5, param_name='n_pauses', rng=rng)),
        },
    )


def _cardiomyopathy(n: int = 50) -> Scenario:
    """PVC-induced cardiomyopathy over ~2 weeks.

    PVC burden rises steeply (isolated -> bigeminy pattern) while
    R-wave amplitude decreases (weakening contractility).
    """
    rng = np.random.default_rng(789)
    return Scenario(
        name="Cardiomyopathy",
        description=(
            "Simulates PVC-induced cardiomyopathy: PVC burden escalates from "
            "occasional isolated PVCs to near-bigeminy patterns, while R-wave "
            "amplitude decreases reflecting weakening ventricular contractility.  "
            "PVC burden is the primary alert driver; amplitude changes affect "
            "morphology metrics."
        ),
        n_sessions=n,
        base_record='119',
        param_schedule={
            'pvc_rate': _noisy_schedule(0.0, 0.40, n, param_name='pvc_rate', rng=rng),
            'amplitude_factor': _noisy_schedule(1.0, 0.65, n, param_name='amplitude_factor', rng=rng),
            'hrv_compression': _noisy_schedule(0.0, 0.30, n, param_name='hrv_compression', rng=rng),
        },
    )


def _stable_patient(n: int = 50) -> Scenario:
    """Stable patient control — no disease progression.

    No transforms applied. Session-to-session variation comes only from the
    pipeline's inherent variability (Pan-Tompkins detection differences, etc.).
    Z-scores should remain near zero; no alerts should fire.
    """
    return Scenario(
        name="Stable Patient",
        description=(
            "Control scenario: no disease progression applied.  The same baseline "
            "ECG is processed each session.  Any metric variation comes from "
            "pipeline noise only.  Expect near-zero z-scores and no alerts."
        ),
        n_sessions=n,
        base_record='100',
        param_schedule={},  # no transforms
    )


PRESET_SCENARIOS: dict[str, Scenario] = {
    'hf_decompensation': _hf_decompensation(),
    'developing_af': _developing_af(),
    'conduction_disease': _conduction_disease(),
    'cardiomyopathy': _cardiomyopathy(),
    'stable_patient': _stable_patient(),
}
