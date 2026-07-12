# -*- coding: utf-8 -*-
"""
searcher.py

Purpose
-------
Screen candidate physical quantities (Hd magnitude, exchange energy,
torque, vortex count, etc.) for whether they act as *leading indicators*
of upcoming magnetic transitions (vortex nucleation/annihilation, domain
switching) during an M-H sweep.

This is meant to run on FFT-only sweep data (film1 / full_fft), with no
trained UNet required, so you can shortlist promising physics-informed
loss-weighting candidates *before* spending ~2 days training a UNet
variant on one of them.

Design principle: everything routes through ONE transition-event detector
(detect_transition_events, based on vortex-count changes) and ONE lag
correlation helper (_lag_correlate), so "event 1" and "best lag" mean the
same thing in every function and CSV this module produces.
"""

import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from dataclasses import dataclass, asdict
import pandas as pd

@dataclass
class PhysicsSnapshot:
    """
    Stores one complete MH-step worth of physics.

    Every row corresponds to ONE external field value after BOTH
    FFT and UNet have converged.
    """

    # -----------------------------
    # MH Sweep Information
    # -----------------------------
    mh_step: int
    Hext: float

    # -----------------------------
    # Field Magnitudes (FFT)
    # -----------------------------
    hd_mean: float
    hd_std: float
    hd_max: float

    he_mean: float
    he_std: float
    he_max: float

    ha_mean: float
    ha_std: float
    ha_max: float

    heff_mean: float
    heff_std: float
    heff_max: float

    # -----------------------------
    # Torque Magnitudes
    # -----------------------------
    tau_hd: float
    tau_he: float
    tau_ha: float
    tau_heff: float

    # -----------------------------
    # Alignment
    # -----------------------------
    align_hd: float
    align_he: float
    align_ha: float
    align_heff: float

    # -----------------------------
    # Energies
    # -----------------------------
    e_demag: float
    e_exchange: float
    e_anis: float
    e_external: float
    e_total: float

    # -----------------------------
    # Magnetization
    # -----------------------------
    mx: float
    my: float
    mz: float

    m_projection: float

    # -----------------------------
    # Topology
    # -----------------------------
    winding_abs: float
    vortex_count: int

    # -----------------------------
    # UNet Errors
    # -----------------------------
    hd_error: float
    trajectory_error: float


class PhysicsRecorder:

    def __init__(self):
        self.snapshots = []

    def add(self, snapshot):
        self.snapshots.append(snapshot)

    def dataframe(self):
        return pd.DataFrame(
            [asdict(s) for s in self.snapshots]
        )

    def save_csv(self, filename):

        df = self.dataframe()

        df.to_csv(filename, index=False)

        return df 

# ============================================================================
# MODEL INTROSPECTION
# ============================================================================

def inspect_model_quantities(model, name_filter=None):
    """
    Print every numerical quantity stored on the MAG2305 model, to
    see what's available to build a candidate predictor from.
    """
    print("\n" + "=" * 75)
    print(f"{'DISCOVERING AVAILABLE MODEL QUANTITIES':^75}")
    print("=" * 75)

    for name in sorted(dir(model)):
        if name.startswith("_"):
            continue
        if name_filter and name_filter.lower() not in name.lower():
            continue
        try:
            value = getattr(model, name)
        except Exception:
            continue
        if callable(value):
            continue

        if isinstance(value, torch.Tensor):
            type_str = f"Tensor({value.device})"
            info_str = f"shape={tuple(value.shape)}"
        elif isinstance(value, np.ndarray):
            type_str = "ndarray"
            info_str = f"shape={value.shape}"
        elif np.isscalar(value):
            type_str = "scalar"
            info_str = f"value={value:.4g}" if isinstance(value, (float, np.floating)) else f"value={value}"
        else:
            type_str = type(value).__name__
            val_str = str(value)
            info_str = f"value={val_str[:40]}..." if len(val_str) > 40 else f"value={val_str}"

        print(f"{'name: ' + name:32s} | type: {type_str:15s} | {info_str}")
    print("=" * 75)


def extract_candidate_predictors(model):
    """
    Automatically extract every scalar (or reducible tensor) quantity from
    the model as a candidate predictor for this Hext step.

    Returns
    -------
    predictors : dict, name -> scalar value at this step
    """
    predictors = {}

    for name in sorted(dir(model)):
        if name.startswith("_"):
            continue
        try:
            value = getattr(model, name)
        except Exception:
            continue
        if callable(value):
            continue

        if isinstance(value, torch.Tensor):
            x = value.detach().cpu().numpy()
            if x.size == 0:
                continue
            if x.ndim >= 2:
                predictors[f"{name}_mean"] = np.mean(x)
                predictors[f"{name}_std"] = np.std(x)
                predictors[f"{name}_max"] = np.max(x)
                predictors[f"{name}_min"] = np.min(x)
                predictors[f"{name}_rms"] = np.sqrt(np.mean(x**2))
            else:
                predictors[name] = np.mean(x)
        elif np.isscalar(value):
            predictors[name] = value

    return predictors


# ============================================================================
# TRANSITION EVENT DETECTION -- single shared convention
# ============================================================================

def detect_transition_events(vortex_count, event_type='both'):
    """
    Detect distinct transition events from a vortex-count trajectory.
    An event is a maximal run of consecutive Hext steps over which the
    vortex count changes; the FIRST step of that run is reported as the
    event onset. Every function in this module uses this detector, so
    "event 1", "event 2", etc. refer to the same physical occurrences
    everywhere -- CSV outputs and plots line up.

    Parameters
    ----------
    vortex_count : array-like
    event_type   : 'both' | 'nucleation' | 'annihilation'

    Returns
    -------
    events : ndarray[int]   onset index of each event
    kinds  : ndarray[str]   'nucleation' or 'annihilation' per event
    """
    vc = np.asarray(vortex_count, dtype=float)
    dv = np.diff(vc)

    if event_type == 'nucleation':
        change_idx = np.where(dv > 0)[0] + 1
    elif event_type == 'annihilation':
        change_idx = np.where(dv < 0)[0] + 1
    else:
        change_idx = np.where(dv != 0)[0] + 1

    if len(change_idx) == 0:
        return np.array([], dtype=int), np.array([], dtype=str)

    groups = [[change_idx[0]]]
    for idx in change_idx[1:]:
        if idx == groups[-1][-1] + 1:
            groups[-1].append(idx)
        else:
            groups.append([idx])

    events = np.array([g[0] for g in groups], dtype=int)
    kinds = np.array(['nucleation' if dv[e - 1] > 0 else 'annihilation' for e in events])
    return events, kinds


# ============================================================================
# LAG CORRELATION -- single shared convention
# ============================================================================

def _lag_correlate(reference, signal, max_lag):
    """
    Scan lags in [-max_lag, +max_lag] for the strongest |Pearson r|
    between `reference` and `signal`.

    Sign convention (fixed across this whole module):
        best_lag > 0  =>  `signal` LEADS `reference` by best_lag steps
                          (signal[t] lines up with reference[t + best_lag]).
    Call with reference=trajectory_error (or an event indicator) and
    signal=candidate_predictor, so a positive best_lag directly means
    "this predictor rises before the error/event" -- the leading-indicator
    property you actually want for a loss weight.
    """
    reference = np.asarray(reference, dtype=float)
    signal = np.asarray(signal, dtype=float)

    r0, _ = pearsonr(reference, signal)
    best_r, best_lag = r0, 0

    for lag in range(1, max_lag + 1):
        r_lead, _ = pearsonr(reference[lag:], signal[:-lag])   # signal leads
        r_lag, _ = pearsonr(reference[:-lag], signal[lag:])    # signal lags
        if abs(r_lead) > abs(best_r):
            best_r, best_lag = r_lead, lag
        if abs(r_lag) > abs(best_r):
            best_r, best_lag = r_lag, -lag

    return best_r, best_lag


# ============================================================================
# LEAD-TIME ESTIMATION 
# ============================================================================

def estimate_lead_time(predictor, events, max_window=20, z_thresh=1.5):
    """
    For each detected transition event, ask: how many steps BEFORE the
    event does `predictor` first deviate from its "quiet" baseline and
    stay deviated through to the event? That step count is the event's
    lead time -- directly answers "would this quantity give my loss
    function advance warning of a transition."

    Baseline is computed from all sweep points that are NOT within
    max_window of any event, so a predictor that's simply high everywhere
    doesn't get credited with a false lead time.

    Parameters
    ----------
    predictor  : array-like, one value per Hext step
    events     : ndarray[int], event onset indices (from detect_transition_events)
    max_window : int, how many steps back to search for onset of deviation
    z_thresh   : float, deviation threshold in baseline std units

    Returns
    -------
    lead_times : ndarray[int], per event, steps of early warning (0 = none detected)
    detected   : ndarray[bool], per event, whether any lead time was found
    baseline_mean, baseline_std : float
    """
    predictor = np.asarray(predictor, dtype=float)
    n = len(predictor)

    quiet_mask = np.ones(n, dtype=bool)
    for e in events:
        lo, hi = max(0, e - max_window), min(n, e + max_window + 1)
        quiet_mask[lo:hi] = False

    if quiet_mask.sum() < 2:
        baseline_mean, baseline_std = np.mean(predictor), np.std(predictor)
    else:
        baseline_mean = predictor[quiet_mask].mean()
        baseline_std = predictor[quiet_mask].std()
    baseline_std = baseline_std if baseline_std > 1e-12 else 1e-12

    lead_times = np.zeros(len(events), dtype=int)
    detected = np.zeros(len(events), dtype=bool)

    for i, e in enumerate(events):
        lo = max(0, e - max_window)
        window_vals = predictor[lo:e]          # strictly before the event
        z = (window_vals - baseline_mean) / baseline_std

        # walk backward from the event; find the longest unbroken run of
        # |z| > z_thresh ending immediately before the event
        run = 0
        for val in z[::-1]:
            if abs(val) > z_thresh:
                run += 1
            else:
                break
        lead_times[i] = run
        detected[i] = run > 0

    return lead_times, detected, baseline_mean, baseline_std


# ============================================================================
# MAIN RANKING PIPELINE
# ============================================================================

def rank_leading_indicators(predictor_dict, vortex_count, trajectory_error,
                             Hext_range, save_path, event_type='both',
                             max_window=20, z_thresh=1.5, max_lag=15):
    """
    Rank every candidate predictor by how well it anticipates transitions.

    Two complementary metrics per predictor:
      1. Lead time against detected transition events (primary) --
         mean steps of early warning, and detection rate across events.
      2. Best leading-lag correlation against trajectory_error (secondary)
         -- ties the predictor to the error behavior your prior work
         already established correlates with transitions.

    Writes a CSV and prints a ranked summary. Sort order: detection rate,
    then mean lead time, then |leading lag correlation| -- i.e. "does it
    warn you at all" beats "how early" beats "how strongly correlated."

    Returns
    -------
    summary_df : pandas.DataFrame, one row per predictor, sorted best-first
    """
    out_dir = os.path.join(save_path, "leading_indicator_analysis")
    os.makedirs(out_dir, exist_ok=True)

    events, kinds = detect_transition_events(vortex_count, event_type=event_type)
    if len(events) == 0:
        print(f"[rank_leading_indicators] No '{event_type}' transition events detected; skipping.")
        return pd.DataFrame()

    error = np.asarray(trajectory_error, dtype=float)
    rows = []

    for name, values in predictor_dict.items():
        values = np.asarray(values, dtype=float)
        if len(values) != len(error):
            print(f"  [skip] '{name}' length {len(values)} != trajectory_error length {len(error)}")
            continue

        lead_times, detected, base_mean, base_std = estimate_lead_time(
            values, events, max_window=max_window, z_thresh=z_thresh)

        best_r, best_lag = _lag_correlate(error, values, max_lag=max_lag)

        rows.append({
            "Predictor": name,
            "Detection Rate": detected.mean(),
            "Mean Lead Time (steps)": lead_times[detected].mean() if detected.any() else 0.0,
            "Max Lead Time (steps)": int(lead_times.max()) if len(lead_times) else 0,
            "N Events": len(events),
            "N Detected": int(detected.sum()),
            "Leading Lag r (vs error)": best_r,
            "Leading Lag (steps, +=leads)": best_lag,
            "Baseline Mean": base_mean,
            "Baseline Std": base_std,
        })

    summary_df = pd.DataFrame(rows)
    if summary_df.empty:
        print("[rank_leading_indicators] No usable predictors (length mismatch with trajectory_error).")
        return summary_df

    summary_df = summary_df.sort_values(
        by=["Detection Rate", "Mean Lead Time (steps)", "Leading Lag r (vs error)"],
        key=lambda col: col.abs() if col.name == "Leading Lag r (vs error)" else col,
        ascending=False
    ).reset_index(drop=True)

    csv_file = os.path.join(out_dir, f"leading_indicator_ranking_{event_type}.csv")
    summary_df.to_csv(csv_file, index=False)

    events_file = os.path.join(out_dir, f"detected_events_{event_type}.csv")
    pd.DataFrame({
        "event_index": events,
        "Hext": np.asarray(Hext_range)[events],
        "kind": kinds,
    }).to_csv(events_file, index=False)

    print()
    print(f"Detected {len(events)} '{event_type}' transition event(s).")
    print(csv_file)
    print()
    print("Top Leading Indicators")
    print("-" * 90)
    print(summary_df[["Predictor", "Detection Rate", "Mean Lead Time (steps)",
                       "Leading Lag r (vs error)", "Leading Lag (steps, +=leads)"]]
          .head(10).to_string(index=False))

    return summary_df


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_leading_indicators(Hext_range, trajectory_error, predictor_dict,
                             vortex_count, general_title, save_path,
                             event_type='both', top_n=6, max_window=20, z_thresh=1.5):
    """
    Plot normalized trajectory error alongside the top-N candidate
    predictors (ranked by rank_leading_indicators), with vertical markers
    at each detected transition event. Legend reports each predictor's
    mean lead time so you can see at a glance which quantities rise
    before the dashed event lines rather than after.
    """
    folder = os.path.join(save_path, "leading_indicator_analysis")
    os.makedirs(folder, exist_ok=True)

    events, kinds = detect_transition_events(vortex_count, event_type=event_type)
    ranking = rank_leading_indicators(predictor_dict, vortex_count, trajectory_error,
                                       Hext_range, save_path, event_type=event_type,
                                       max_window=max_window, z_thresh=z_thresh)
    if ranking.empty:
        return

    top_names = ranking["Predictor"].head(top_n).tolist()

    def normalize(x):
        x = np.asarray(x, dtype=float)
        rng = np.max(x) - np.min(x)
        return np.zeros_like(x) if rng == 0 else (x - np.min(x)) / rng

    error = np.asarray(trajectory_error, dtype=float)

    plt.figure(figsize=(14, 8))
    plt.plot(Hext_range, normalize(error), linewidth=3, color="black", label="Trajectory Error")

    colors = plt.cm.tab10(np.linspace(0, 1, len(top_names)))
    for color, name in zip(colors, top_names):
        row = ranking[ranking["Predictor"] == name].iloc[0]
        label = f"{name} (lead={row['Mean Lead Time (steps)']:.1f} steps, det={row['Detection Rate']:.0%})"
        plt.plot(Hext_range, normalize(predictor_dict[name]), linewidth=2, alpha=0.85,
                  color=color, label=label)

    for e, kind in zip(events, kinds):
        ls = '--' if kind == 'nucleation' else ':'
        plt.axvline(Hext_range[e], color="red", linestyle=ls, alpha=0.5)

    plt.grid(alpha=0.3)
    plt.xlabel("External Field (Oe)")
    plt.ylabel("Normalized Quantity")
    plt.title(general_title + f"\n\nLeading Indicators of Transitions ({event_type})",
              fontsize=13, fontweight='bold')
    plt.legend(fontsize=8, loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(folder, f"leading_indicators_{event_type}.png"), dpi=250)
    plt.close()