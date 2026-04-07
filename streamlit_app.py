"""Streamlit frontend for longitudinal ECG trend analysis.

Calls the ML pipeline modules directly — no API server needed.

Usage:
    uv run streamlit run streamlit_app.py
"""

from __future__ import annotations

import dataclasses

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from ecg_monitor.pipeline import DS1_RECORDS, DS2_RECORDS
from ecg_monitor.scenarios import (
    DEFAULT_PARAMS, PRESET_SCENARIOS, Scenario, build_custom_scenario,
)
from ecg_monitor.simulation import SimulationContext, run_scenario
from ecg_monitor.trends import (
    ALERT_RULES, DISPLAY_METRICS, TRACKED_METRICS,
    compute_trends, detect_alerts, detect_forecast_alerts, summarize_alerts,
)


# ============================================================================
# Page config
# ============================================================================

st.set_page_config(
    page_title="ECG Trend Analysis",
    page_icon=":anatomical_heart:",
    layout="wide",
)

st.title("Longitudinal ECG Trend Analysis")
st.caption("Simulate disease progression, detect metric trends, and review clinical alerts.")


# ============================================================================
# Model loading (cached — loads once per process)
# ============================================================================

@st.cache_resource(show_spinner="Loading models and data...")
def load_context() -> SimulationContext:
    return SimulationContext()


# ============================================================================
# Helpers
# ============================================================================

METRIC_DISPLAY = {
    'mean_hr_bpm': ('Heart Rate', 'bpm'),
    'sdnn_ms': ('SDNN', 'ms'),
    'rmssd_ms': ('RMSSD', 'ms'),
    'pnn50_pct': ('pNN50', '%'),
    'pvc_burden_pct': ('PVC Burden', '%'),
    'sve_burden_pct': ('SVE Burden', '%'),
    'mean_qrs_width_ms': ('QRS Width', 'ms'),
    'pause_count_2s': ('Pauses (>2s)', 'count'),
    'max_rr_interval_s': ('Max RR Interval', 's'),
    'mean_r_amplitude': ('R Amplitude', 'mV'),
    'rr_cv_pct': ('RR Irregularity', '%'),
}


def run_simulation(ctx: SimulationContext, scenario: Scenario, model_type: str = 'gb') -> dict:
    """Run a full simulation and return trends, alerts, forecasts."""
    df_sessions = run_scenario(ctx, scenario, model_type=model_type)

    if df_sessions.empty:
        st.error("Simulation produced no results.")
        st.stop()

    n_baseline = 3
    ewma_span = 3

    df_trends = compute_trends(df_sessions, n_baseline=n_baseline, ewma_span=ewma_span)
    actual_alerts = detect_alerts(df_trends, n_baseline=n_baseline)
    forecast_alerts, forecasts_by_metric = detect_forecast_alerts(
        df_trends, actual_alerts, n_baseline=n_baseline)

    all_alerts = actual_alerts + forecast_alerts
    severity_order = {'forecast': 0, 'danger': 1, 'warning': 2}
    all_alerts.sort(key=lambda a: (a.session, severity_order.get(a.severity, 3)))

    summary = summarize_alerts(all_alerts, df_sessions)

    return {
        'scenario_name': scenario.name,
        'session_metrics': df_sessions.to_dict(orient='records'),
        'trends': df_trends.to_dict(orient='records'),
        'alerts': [dataclasses.asdict(a) for a in all_alerts],
        'forecasts': {
            metric: [dataclasses.asdict(f) for f in flist]
            for metric, flist in forecasts_by_metric.items()
        },
        'summary': summary,
    }


# ============================================================================
# Sidebar — scenario selection
# ============================================================================

available_records = sorted(DS1_RECORDS | DS2_RECORDS)

with st.sidebar:
    st.header("Scenario Configuration")

    mode = st.radio("Mode", ["Preset", "Custom"], horizontal=True)

    if mode == "Preset":
        scenario_keys = list(PRESET_SCENARIOS.keys())
        scenarios = list(PRESET_SCENARIOS.values())
        scenario_names = [s.name for s in scenarios]
        selected_idx = st.selectbox(
            "Select scenario",
            range(len(scenarios)),
            format_func=lambda i: scenario_names[i],
        )
        selected_scenario = scenarios[selected_idx]
        st.markdown(f"**{selected_scenario.name}**")
        st.markdown(selected_scenario.description)

        # Record override — default to the scenario's base record
        preset_record = selected_scenario.base_record
        default_rec_idx = (available_records.index(preset_record)
                           if preset_record in available_records else 0)
        base_record = st.selectbox(
            "Patient record", available_records, index=default_rec_idx,
            key="preset_record",
        )

        # Build scenario (override base_record if changed)
        if base_record != selected_scenario.base_record:
            scenario = dataclasses.replace(selected_scenario, base_record=base_record)
        else:
            scenario = selected_scenario

    else:
        st.subheader("Custom Scenario Builder")

        base_record = st.selectbox(
            "Patient record", available_records,
            index=available_records.index('100') if '100' in available_records else 0,
            key="custom_record",
        )
        n_sessions = st.slider("Number of sessions", 5, 60, 50)

        st.markdown("**Transform parameters**")
        st.caption("Set the start (session 1) and end (final session) values. "
                   "Values are interpolated with realistic noise between sessions.")
        param_ranges = {}

        def _param_sliders(label, description, key, min_val, max_val,
                           default, step, neutral=None):
            """Render start/end sliders for a transform parameter.

            Returns (start, end) or None if both are at the neutral value.
            """
            st.markdown(f"**{label}** — {description}")
            c1, c2 = st.columns(2)
            with c1:
                start = st.slider(
                    f"{label}: start", min_val, max_val, default,
                    step, key=f"{key}_start")
            with c2:
                end = st.slider(
                    f"{label}: end", min_val, max_val, default,
                    step, key=f"{key}_end")
            if neutral is not None:
                if start != neutral or end != neutral:
                    return [start, end]
            else:
                if start > min_val or end > min_val:
                    return [start, end]
            return None

        result = _param_sliders(
            "Heart rate factor",
            "<1.0 = tachycardia, >1.0 = bradycardia",
            "hr_factor", 0.50, 2.00, 1.00, 0.05, neutral=1.0)
        if result:
            param_ranges['hr_factor'] = result

        result = _param_sliders(
            "HRV compression",
            "reduces beat-to-beat variability",
            "hrv_comp", 0.0, 0.95, 0.0, 0.05)
        if result:
            param_ranges['hrv_compression'] = result

        result = _param_sliders(
            "PVC insertion rate",
            "fraction of beats replaced with PVCs",
            "pvc_rate", 0.0, 0.50, 0.0, 0.01)
        if result:
            param_ranges['pvc_rate'] = result

        result = _param_sliders(
            "SVE insertion rate",
            "fraction of beats replaced with SVEs",
            "sve_rate", 0.0, 0.50, 0.0, 0.01)
        if result:
            param_ranges['sve_rate'] = result

        result = _param_sliders(
            "QRS widening",
            "simulates bundle branch block",
            "qrs_width", 1.0, 2.5, 1.0, 0.1, neutral=1.0)
        if result:
            param_ranges['qrs_width_factor'] = result

        result = _param_sliders(
            "P-wave attenuation",
            "simulates atrial disease / AF progression",
            "pwave_atten", 0.0, 1.0, 0.0, 0.05)
        if result:
            param_ranges['pwave_attenuation'] = result

        result = _param_sliders(
            "AF irregularity",
            "randomizes RR intervals",
            "af_irreg", 0.0, 0.50, 0.0, 0.05)
        if result:
            param_ranges['af_irregularity'] = result

        st.markdown("**Inserted pauses** — simulates AV conduction block")
        pc1, pc2 = st.columns(2)
        with pc1:
            pause_start = st.slider(
                "Pauses: start", 0, 15, 0, 1, key="n_pauses_start")
        with pc2:
            pause_end = st.slider(
                "Pauses: end", 0, 15, 0, 1, key="n_pauses_end")
        if pause_start > 0 or pause_end > 0:
            param_ranges['n_pauses'] = [float(pause_start), float(pause_end)]

        result = _param_sliders(
            "Amplitude scaling",
            "simulates effusion / cardiomyopathy",
            "amp_factor", 0.20, 1.00, 1.00, 0.05, neutral=1.0)
        if result:
            param_ranges['amplitude_factor'] = result

        # Build custom scenario
        ranges = {k: (v[0], v[1]) for k, v in param_ranges.items() if len(v) == 2}
        scenario = build_custom_scenario(
            name="Custom",
            base_record=base_record,
            n_sessions=n_sessions,
            param_ranges=ranges,
            description="User-defined custom scenario",
        )

    st.divider()
    st.subheader("Display options")
    show_ewma = st.checkbox("Show EWMA line", value=False)

    st.divider()
    run_button = st.button("Run Simulation", type="primary", use_container_width=True)


# ============================================================================
# Main content — results
# ============================================================================

if run_button:
    ctx = load_context()
    with st.spinner("Running simulation... (this may take 30-60 seconds)"):
        result = run_simulation(ctx, scenario)
    st.session_state['result'] = result

if 'result' not in st.session_state:
    st.info("Select a scenario and click **Run Simulation** to begin.")
    st.stop()

result = st.session_state['result']
df_sessions = pd.DataFrame(result['session_metrics'])
df_trends = pd.DataFrame(result['trends'])
alerts = result['alerts']
forecasts = result.get('forecasts', {})
summary = result['summary']

# --- Summary card ---
st.header(f"Results: {result['scenario_name']}")

n_danger = sum(1 for a in alerts if a['severity'] == 'danger')
n_warning = sum(1 for a in alerts if a['severity'] == 'warning')
n_forecast = sum(1 for a in alerts if a['severity'] == 'forecast')

# Alert count badges
alert_cols = st.columns(3)
with alert_cols[0]:
    st.metric("Danger", n_danger)
with alert_cols[1]:
    st.metric("Warning", n_warning)
with alert_cols[2]:
    st.metric("Forecast", n_forecast)

# Metric change summary — first vs last session for alerted metrics
alerted_metrics = {a['metric'] for a in alerts}
first_row = df_sessions.iloc[0]
last_row = df_sessions.iloc[-1]
n_sessions = len(df_sessions)

change_rows = []
for metric_key, (display_name, unit) in METRIC_DISPLAY.items():
    if metric_key not in alerted_metrics:
        continue
    if metric_key not in df_sessions.columns:
        continue
    v0 = first_row.get(metric_key)
    v1 = last_row.get(metric_key)
    if v0 is None or v1 is None or pd.isna(v0) or pd.isna(v1):
        continue
    delta = v1 - v0
    pct = 100.0 * delta / abs(v0) if abs(v0) > 1e-9 else 0.0
    arrow = "\u2191" if delta > 0 else "\u2193" if delta < 0 else "\u2192"
    change_rows.append({
        'Metric': display_name,
        'Baseline': f"{v0:.1f} {unit}",
        'Final': f"{v1:.1f} {unit}",
        'Change': f"{arrow} {abs(delta):.1f} {unit} ({pct:+.0f}%)",
    })

if change_rows:
    st.markdown(f"**Metric changes over {n_sessions} sessions** (alerted metrics only):")
    st.dataframe(
        pd.DataFrame(change_rows),
        use_container_width=True,
        hide_index=True,
    )
elif not alerts:
    st.success("No clinical alerts triggered. All metrics within baseline range.")

st.divider()

# --- Trend charts ---
st.subheader("Metric Trends")

with st.expander("Chart legend", expanded=False):
    st.markdown(
        '<span style="color:#636EFA">\u2500\u2500\u25CF\u2500\u2500</span> '
        "Raw metric value &nbsp;&nbsp;|&nbsp;&nbsp; "
        '<span style="color:#636EFA">\u2500 \u2500 \u2500</span> '
        "Forecast projection &nbsp;&nbsp;|&nbsp;&nbsp; "
        '<span style="color:gray">\u2500 \u2500</span> '
        "Baseline mean<br>"
        '<span style="background:rgba(255,165,0,0.3)">&nbsp;&nbsp;&nbsp;&nbsp;</span> '
        "Warning zone (2\u20133\u03c3) &nbsp;&nbsp;|&nbsp;&nbsp; "
        '<span style="background:rgba(255,0,0,0.3)">&nbsp;&nbsp;&nbsp;&nbsp;</span> '
        "Danger zone (>3\u03c3) &nbsp;&nbsp;|&nbsp;&nbsp; "
        '<span style="background:rgba(99,110,250,0.2)">&nbsp;&nbsp;&nbsp;&nbsp;</span> '
        "Forecast confidence interval<br>"
        '<span style="color:orange">\u25B2</span> '
        "Warning alert &nbsp;&nbsp;|&nbsp;&nbsp; "
        '<span style="color:red">\u2716</span> '
        "Danger alert &nbsp;&nbsp;|&nbsp;&nbsp; "
        '<span style="color:orange">\u2605</span> '
        "Projected warning crossing &nbsp;&nbsp;|&nbsp;&nbsp; "
        '<span style="color:red">\u2605</span> '
        "Projected danger crossing",
        unsafe_allow_html=True,
    )

# Group metrics into rows for a cleaner layout
metric_groups = [
    ['mean_hr_bpm', 'sdnn_ms', 'rr_cv_pct'],
    ['pvc_burden_pct', 'sve_burden_pct', 'mean_qrs_width_ms'],
    ['pause_count_2s', 'max_rr_interval_s', 'mean_r_amplitude'],
]

for group in metric_groups:
    cols = st.columns(len(group))
    for i, metric in enumerate(group):
        with cols[i]:
            metric_data = df_trends[df_trends['metric'] == metric]
            if metric_data.empty:
                st.caption(f"{metric}: no data")
                continue

            display_name, unit = METRIC_DISPLAY.get(metric, (metric, ''))
            baseline_mean = metric_data['baseline_mean'].iloc[0]
            baseline_std = metric_data['baseline_std'].iloc[0]

            fig = go.Figure()

            # Baseline bands — only for alerted metrics, on the clinically concerning side
            ALERT_DIRECTIONS = {
                'mean_hr_bpm': 'both',
                'sdnn_ms': 'falling',
                'rr_cv_pct': 'rising',
                'pvc_burden_pct': 'rising',
                'sve_burden_pct': 'rising',
                'mean_qrs_width_ms': 'rising',
                'pause_count_2s': 'rising',
                'max_rr_interval_s': 'rising',
                'mean_r_amplitude': 'falling',
            }
            # Display-only metrics (no alert bands): rmssd_ms, pnn50_pct
            alert_dir = ALERT_DIRECTIONS.get(metric)
            sessions = metric_data['session'].values
            sess_list = list(sessions)
            sess_rev = list(sessions[::-1])

            if alert_dir and alert_dir in ('rising', 'both'):
                # Upper danger band (3σ)
                fig.add_trace(go.Scatter(
                    x=sess_list + sess_rev,
                    y=([baseline_mean + 3*baseline_std]*len(sessions) +
                       [baseline_mean + 2*baseline_std]*len(sessions)),
                    fill='toself', fillcolor='rgba(255,0,0,0.15)',
                    line=dict(width=0), showlegend=False, hoverinfo='skip',
                ))
                # Upper warning band (2σ)
                fig.add_trace(go.Scatter(
                    x=sess_list + sess_rev,
                    y=([baseline_mean + 2*baseline_std]*len(sessions) +
                       [baseline_mean + 1*baseline_std]*len(sessions)),
                    fill='toself', fillcolor='rgba(255,165,0,0.18)',
                    line=dict(width=0), showlegend=False, hoverinfo='skip',
                ))

            if alert_dir and alert_dir in ('falling', 'both'):
                # Lower danger band (-3σ)
                fig.add_trace(go.Scatter(
                    x=sess_list + sess_rev,
                    y=([baseline_mean - 2*baseline_std]*len(sessions) +
                       [baseline_mean - 3*baseline_std]*len(sessions)),
                    fill='toself', fillcolor='rgba(255,0,0,0.15)',
                    line=dict(width=0), showlegend=False, hoverinfo='skip',
                ))
                # Lower warning band (-2σ)
                fig.add_trace(go.Scatter(
                    x=sess_list + sess_rev,
                    y=([baseline_mean - 1*baseline_std]*len(sessions) +
                       [baseline_mean - 2*baseline_std]*len(sessions)),
                    fill='toself', fillcolor='rgba(255,165,0,0.18)',
                    line=dict(width=0), showlegend=False, hoverinfo='skip',
                ))

            # Baseline mean line
            fig.add_hline(y=baseline_mean, line_dash="dash",
                          line_color="gray", opacity=0.5)

            # Raw values
            fig.add_trace(go.Scatter(
                x=metric_data['session'], y=metric_data['raw_value'],
                mode='lines+markers', name='Raw',
                line=dict(color='#636EFA', width=1),
                marker=dict(size=4),
            ))

            # EWMA (optional)
            if show_ewma:
                fig.add_trace(go.Scatter(
                    x=metric_data['session'], y=metric_data['ewma_value'],
                    mode='lines', name='EWMA',
                    line=dict(color='#EF553B', width=2),
                ))

            # Forecast projection — show from the first session where P >= 70%
            metric_forecasts = forecasts.get(metric, [])
            if metric_forecasts:
                # Find the trigger forecast (first where prob >= 0.70)
                trigger_fc = None
                for fc in metric_forecasts:
                    if fc.get('prob_danger', 0) >= 0.70 or fc.get('prob_warn', 0) >= 0.70:
                        trigger_fc = fc
                        break
                # Fall back to latest if no trigger
                display_fc = trigger_fc or metric_forecasts[-1]

                fc_sessions = display_fc['projected_sessions']
                fc_values = display_fc['projected_values']
                fc_lo = display_fc['ci_lower']
                fc_hi = display_fc['ci_upper']
                fit_session = display_fc['fit_session']
                prob_w = display_fc.get('prob_warn', 0)
                prob_d = display_fc.get('prob_danger', 0)

                # Confidence cone
                fig.add_trace(go.Scatter(
                    x=fc_sessions + fc_sessions[::-1],
                    y=fc_hi + fc_lo[::-1],
                    fill='toself', fillcolor='rgba(99,110,250,0.10)',
                    line=dict(width=0), showlegend=False, hoverinfo='skip',
                ))

                # Connect from fit session's observed value to projection
                fit_row = metric_data[metric_data['session'] == fit_session]
                if not fit_row.empty:
                    anchor_val = float(fit_row['raw_value'].iloc[0])
                else:
                    anchor_val = float(metric_data['raw_value'].iloc[-1])

                fig.add_trace(go.Scatter(
                    x=[float(fit_session)] + fc_sessions,
                    y=[anchor_val] + fc_values,
                    mode='lines', name='Forecast',
                    line=dict(color='#636EFA', width=2, dash='dash'),
                ))

                # Probability annotation
                max_prob = max(prob_w, prob_d)
                if max_prob > 0.01:
                    prob_label = f"P={max_prob:.0%}"
                    fig.add_annotation(
                        x=fc_sessions[len(fc_sessions)//2],
                        y=fc_values[len(fc_values)//2],
                        text=prob_label,
                        showarrow=False,
                        font=dict(size=10, color='#636EFA'),
                        bgcolor='rgba(255,255,255,0.8)',
                    )

                # Threshold crossing markers
                if display_fc.get('warn_crossing') is not None:
                    wc = display_fc['warn_crossing']
                    fig.add_trace(go.Scatter(
                        x=[wc], y=[baseline_mean + 2*baseline_std
                                    if fc_values[-1] > baseline_mean
                                    else baseline_mean - 2*baseline_std],
                        mode='markers', name='Warn crossing',
                        marker=dict(color='orange', size=12,
                                    symbol='star', line=dict(width=1, color='black')),
                        hovertext=f"Projected warning at session {wc:.1f}",
                    ))
                if display_fc.get('danger_crossing') is not None:
                    dc = display_fc['danger_crossing']
                    fig.add_trace(go.Scatter(
                        x=[dc], y=[baseline_mean + 3*baseline_std
                                    if fc_values[-1] > baseline_mean
                                    else baseline_mean - 3*baseline_std],
                        mode='markers', name='Danger crossing',
                        marker=dict(color='red', size=12,
                                    symbol='star', line=dict(width=1, color='black')),
                        hovertext=f"Projected danger at session {dc:.1f}",
                    ))

            # Alert markers
            metric_alerts = [a for a in alerts if a['metric'] == metric
                             and a['severity'] != 'forecast']
            if metric_alerts:
                danger_sessions = [a['session'] for a in metric_alerts
                                   if a['severity'] == 'danger']
                danger_values = [a['value'] for a in metric_alerts
                                 if a['severity'] == 'danger']
                warn_sessions = [a['session'] for a in metric_alerts
                                 if a['severity'] == 'warning']
                warn_values = [a['value'] for a in metric_alerts
                               if a['severity'] == 'warning']

                if danger_sessions:
                    fig.add_trace(go.Scatter(
                        x=danger_sessions, y=danger_values,
                        mode='markers', name='Danger',
                        marker=dict(color='red', size=10, symbol='x'),
                    ))
                if warn_sessions:
                    fig.add_trace(go.Scatter(
                        x=warn_sessions, y=warn_values,
                        mode='markers', name='Warning',
                        marker=dict(color='orange', size=8,
                                    symbol='triangle-up'),
                    ))

            fig.update_layout(
                title=f"{display_name} ({unit})",
                xaxis_title="Session",
                yaxis_title=unit,
                height=280,
                margin=dict(l=40, r=20, t=40, b=30),
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

st.divider()

# --- Alert timeline ---
st.subheader("Alert Timeline")

if not alerts:
    st.success("No alerts fired. All metrics remained within baseline range.")
else:
    n_total = len(alerts)
    with st.expander(f"View all {n_total} alerts", expanded=False):
        for alert in alerts:
            severity = alert['severity']
            if severity == 'danger':
                color = 'red'
            elif severity == 'warning':
                color = 'orange'
            else:
                color = 'purple'
            z = alert['z_score']
            val = alert['value']
            base = alert['baseline_mean']
            unit = alert['unit']
            display_name = METRIC_DISPLAY.get(alert['metric'], (alert['metric'], ''))[0]
            severity_label = severity.upper()

            st.markdown(
                f"**Session {alert['session']}** "
                f"[<span style='color:{color};font-weight:bold'>{severity_label}</span>] | "
                f"**{display_name}**: {val:.1f} {unit} "
                f"(baseline: {base:.1f}, z={z:+.1f}) | "
                f"_{alert['interpretation']}_",
                unsafe_allow_html=True,
            )

st.divider()

# --- Session drill-down ---
st.subheader("Session Details")

session_idx = st.selectbox(
    "Select session to inspect",
    sorted(df_sessions['session'].unique()),
    format_func=lambda s: f"Session {s}",
)

session_row = df_sessions[df_sessions['session'] == session_idx].iloc[0]

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Heart Rate", f"{session_row.get('mean_hr_bpm', 0):.1f} bpm")
    st.metric("SDNN", f"{session_row.get('sdnn_ms', 0):.1f} ms")
with col2:
    st.metric("PVC Burden", f"{session_row.get('pvc_burden_pct', 0):.1f}%")
    st.metric("SVE Burden", f"{session_row.get('sve_burden_pct', 0):.1f}%")
with col3:
    st.metric("QRS Width", f"{session_row.get('mean_qrs_width_ms', 0):.1f} ms")
    st.metric("R Amplitude", f"{session_row.get('mean_r_amplitude', 0):.3f} mV")
with col4:
    st.metric("Total Beats", f"{int(session_row.get('total_beats', 0))}")
    st.metric("Pauses (>2s)", f"{int(session_row.get('pause_count_2s', 0))}")

# Beat classification breakdown
n_count = int(session_row.get('n_count', 0))
s_count = int(session_row.get('s_count', 0))
v_count = int(session_row.get('v_count', 0))
total = n_count + s_count + v_count

if total > 0:
    beat_df = pd.DataFrame({
        'Class': ['Normal (N)', 'Supraventricular (S)', 'Ventricular (V)'],
        'Count': [n_count, s_count, v_count],
        'Percentage': [f"{100*n_count/total:.1f}%",
                       f"{100*s_count/total:.1f}%",
                       f"{100*v_count/total:.1f}%"],
    })
    st.dataframe(beat_df, use_container_width=True, hide_index=True)

# Session alerts
session_alerts = [a for a in alerts if a['session'] == session_idx]
if session_alerts:
    st.markdown("**Alerts for this session:**")
    for a in session_alerts:
        if a['severity'] == 'danger':
            color = 'red'
        elif a['severity'] == 'warning':
            color = 'orange'
        else:
            color = 'purple'
        display_name = METRIC_DISPLAY.get(a['metric'], (a['metric'], ''))[0]
        label = a['severity'].upper()
        st.markdown(
            f"<span style='color:{color};font-weight:bold'>[{label}]</span> "
            f"{display_name}: {a['interpretation']}",
            unsafe_allow_html=True,
        )
else:
    st.markdown("*No alerts for this session.*")

# Forecast probability evolution chart
# Collect P(danger) across all sessions for each metric
prob_data = []
for metric, fc_list in forecasts.items():
    display_name = METRIC_DISPLAY.get(metric, (metric, ''))[0]
    for fc in fc_list:
        pd_val = fc.get('prob_danger', 0)
        if pd_val > 0.001:
            prob_data.append({
                'session': fc['fit_session'],
                'metric': display_name,
                'prob_danger': pd_val,
            })

if prob_data:
    st.markdown("**Forecast: P(danger within 20 sessions)**")
    prob_fig = go.Figure()

    # Group by metric
    prob_df = pd.DataFrame(prob_data)
    for metric_name in prob_df['metric'].unique():
        mdata = prob_df[prob_df['metric'] == metric_name]
        prob_fig.add_trace(go.Scatter(
            x=mdata['session'], y=mdata['prob_danger'],
            mode='lines+markers', name=metric_name,
            line=dict(width=2), marker=dict(size=4),
        ))

    # 70% alert threshold
    prob_fig.add_hline(
        y=0.70, line_dash="dash", line_color="red", opacity=0.6,
        annotation_text="Alert threshold (70%)",
        annotation_position="top left",
    )

    prob_fig.update_layout(
        xaxis_title="Session",
        yaxis_title="P(danger)",
        yaxis=dict(range=[0, 1], tickformat='.0%'),
        height=300,
        margin=dict(l=40, r=20, t=20, b=30),
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
    )

    # Highlight current session
    prob_fig.add_vline(
        x=session_idx, line_dash="dot", line_color="gray", opacity=0.5,
    )

    st.plotly_chart(prob_fig, use_container_width=True)
