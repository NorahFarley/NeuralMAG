# -*- coding: utf-8 -*-
"""
Physics-aware data collection and leading-indicator analysis for NeuralMAG
M-H sweeps.

One PhysicsSnapshot is recorded after the FFT and UNet solvers have converged
at one external-field step. Predictor columns are derived only from the FFT
reference solution. UNet quantities are stored separately as diagnostics and
error targets.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr


_EPS = 1.0e-12


def _as_float(value, default=np.nan) -> float:
    """Convert Python/NumPy/Torch scalar-like values to a plain float."""
    if value is None:
        return float(default)
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return float(default)
        return float(value.detach().mean().cpu().item())
    try:
        return float(np.asarray(value).mean())
    except (TypeError, ValueError):
        return float(default)


def _masked_values(values: torch.Tensor, active_mask: torch.Tensor) -> torch.Tensor:
    """Return values at magnetic cells only; fall back to all values if needed."""
    selected = values[active_mask]
    return selected if selected.numel() else values.reshape(-1)


def _field_statistics(field: torch.Tensor, active_mask: torch.Tensor) -> Dict[str, float]:
    magnitudes = torch.linalg.vector_norm(field, dim=-1)
    magnitudes = _masked_values(magnitudes, active_mask)
    return {
        "mean": float(magnitudes.mean().item()),
        "std": float(magnitudes.std(unbiased=False).item()),
        "max": float(magnitudes.max().item()),
        "rms": float(torch.sqrt(torch.mean(magnitudes.square())).item()),
    }


def _torque_statistics(spin: torch.Tensor, field: torch.Tensor,
                       active_mask: torch.Tensor) -> Dict[str, float]:
    torque = torch.linalg.vector_norm(torch.cross(spin, field, dim=-1), dim=-1)
    torque = _masked_values(torque, active_mask)
    return {
        "mean": float(torque.mean().item()),
        "max": float(torque.max().item()),
        "rms": float(torch.sqrt(torch.mean(torque.square())).item()),
    }


def _alignment_statistics(spin: torch.Tensor, field: torch.Tensor,
                          active_mask: torch.Tensor) -> Dict[str, float]:
    spin_norm = torch.linalg.vector_norm(spin, dim=-1)
    field_norm = torch.linalg.vector_norm(field, dim=-1)
    cosine = torch.sum(spin * field, dim=-1) / (spin_norm * field_norm + _EPS)
    cosine = _masked_values(cosine, active_mask)
    return {
        "mean": float(cosine.mean().item()),
        "abs_mean": float(cosine.abs().mean().item()),
        "std": float(cosine.std(unbiased=False).item()),
    }


def _energy_value(model, attribute: str) -> float:
    """Read an energy tensor/scalar after model.GetEnergy_detailed(...)."""
    return _as_float(getattr(model, attribute, None))


def _vector_error(reference: torch.Tensor, prediction: torch.Tensor,
                  active_mask: torch.Tensor) -> Dict[str, float]:
    """Spatial errors between two vector fields, restricted to magnetic cells."""
    difference = prediction - reference
    component_abs = difference.abs()
    vector_l2 = torch.linalg.vector_norm(difference, dim=-1)
    ref_norm = torch.linalg.vector_norm(reference, dim=-1)
    pred_norm = torch.linalg.vector_norm(prediction, dim=-1)
    cosine = torch.sum(reference * prediction, dim=-1) / (ref_norm * pred_norm + _EPS)

    component_abs = component_abs[active_mask]
    vector_l2 = _masked_values(vector_l2, active_mask)
    cosine = _masked_values(cosine, active_mask)

    ref_active = reference[active_mask]
    denom = torch.sqrt(torch.mean(ref_active.square())) + _EPS
    return {
        "mae": float(component_abs.mean().item()),
        "rmse": float(torch.sqrt(torch.mean(difference[active_mask].square())).item()),
        "vector_l2_mean": float(vector_l2.mean().item()),
        "vector_l2_max": float(vector_l2.max().item()),
        "relative_rmse": float((torch.sqrt(torch.mean(difference[active_mask].square())) / denom).item()),
        "cosine_mean": float(cosine.mean().item()),
    }


@dataclass(frozen=True)
class PhysicsSnapshot:
    """One converged external-field step from an M-H sweep."""

    mh_step: int
    hext_scalar: float
    hext_x: float
    hext_y: float
    hext_z: float

    # FFT ground-truth field statistics: physical predictors.
    fft_hd_mean: float
    fft_hd_std: float
    fft_hd_max: float
    fft_hd_rms: float
    fft_he_mean: float
    fft_he_std: float
    fft_he_max: float
    fft_he_rms: float
    fft_ha_mean: float
    fft_ha_std: float
    fft_ha_max: float
    fft_ha_rms: float
    fft_heff_mean: float
    fft_heff_std: float
    fft_heff_max: float
    fft_heff_rms: float

    # FFT ground-truth torques.
    fft_tau_hd_mean: float
    fft_tau_hd_max: float
    fft_tau_hd_rms: float
    fft_tau_he_mean: float
    fft_tau_he_max: float
    fft_tau_he_rms: float
    fft_tau_ha_mean: float
    fft_tau_ha_max: float
    fft_tau_ha_rms: float
    fft_tau_heff_mean: float
    fft_tau_heff_max: float
    fft_tau_heff_rms: float

    # FFT ground-truth spin/field alignment.
    fft_align_hd_mean: float
    fft_align_hd_abs_mean: float
    fft_align_hd_std: float
    fft_align_he_mean: float
    fft_align_he_abs_mean: float
    fft_align_he_std: float
    fft_align_ha_mean: float
    fft_align_ha_abs_mean: float
    fft_align_ha_std: float
    fft_align_heff_mean: float
    fft_align_heff_abs_mean: float
    fft_align_heff_std: float

    # FFT ground-truth energies.
    fft_e_demag: float
    fft_e_exchange: float
    fft_e_anis: float
    fft_e_external: float
    fft_e_total: float

    # FFT ground-truth magnetization and topology.
    fft_mx: float
    fft_my: float
    fft_mz: float
    fft_mz_abs_mean: float
    fft_m_projection: float
    fft_winding_abs: float
    fft_winding_sum: float

    # UNet state summaries: diagnostics, never predictor candidates by default.
    unet_mx: float
    unet_my: float
    unet_mz: float
    unet_mz_abs_mean: float
    unet_m_projection: float
    unet_winding_abs: float
    unet_winding_sum: float

    # FFT-vs-UNet accuracy targets.
    hd_mae: float
    hd_rmse: float
    hd_vector_l2_mean: float
    hd_vector_l2_max: float
    hd_relative_rmse: float
    hd_cosine_mean: float
    spin_mae: float
    spin_rmse: float
    spin_vector_l2_mean: float
    spin_vector_l2_max: float
    spin_relative_rmse: float
    spin_cosine_mean: float
    he_mae: float
    ha_mae: float
    heff_mae: float
    m_projection_abs_error: float

    # Solver diagnostics.
    fft_iterations: int
    unet_iterations: int
    fft_final_convergence_error: float
    unet_final_convergence_error: float
    fft_runtime_seconds: float
    unet_runtime_seconds: float


class PhysicsRecorder:
    """
    Collect one PhysicsSnapshot per converged M-H field step.

    Predictor data are selected with predictor_columns(); only columns whose
    names start with ``fft_`` are considered, with topology labels and sweep
    coordinates excluded by default.
    """

    def __init__(self) -> None:
        self.snapshots = []

    def __len__(self) -> int:
        return len(self.snapshots)

    @torch.no_grad()
    def capture(
        self,
        *,
        mh_step: int,
        hext_scalar: float,
        hext_vector: Sequence[float],
        film_fft,
        film_unet,
        fft_winding_abs,
        fft_winding_sum,
        unet_winding_abs,
        unet_winding_sum,
        fft_iterations: int,
        unet_iterations: int,
        fft_final_convergence_error: float,
        unet_final_convergence_error: float,
        fft_runtime_seconds: float,
        unet_runtime_seconds: float,
        cell_count: Optional[int] = None,
    ) -> PhysicsSnapshot:
        """Compute and append one converged M-H-step snapshot."""
        hext = torch.as_tensor(hext_vector, dtype=film_fft.Spin.dtype,
                               device=film_fft.Spin.device).reshape(3)

        # A magnetic cell has nonzero spin. This excludes masked holes from all
        # averages, preventing zero-valued geometry cells from diluting metrics.
        active_fft = torch.linalg.vector_norm(film_fft.Spin, dim=-1) > _EPS
        active_unet = torch.linalg.vector_norm(film_unet.Spin, dim=-1) > _EPS
        active = active_fft & active_unet
        if not torch.any(active):
            raise ValueError("No active magnetic cells were found while recording physics.")

        n_active = int(active.sum().item())
        if cell_count is not None and int(cell_count) != n_active:
            # Use the tensor-derived mask as the source of truth, but make the
            # discrepancy visible because it usually indicates mask handling.
            print(f"[PhysicsRecorder] active-cell count {n_active} differs from cell_count={cell_count}.")

        fft_fields = {
            "hd": _field_statistics(film_fft.Hd, active),
            "he": _field_statistics(film_fft.He, active),
            "ha": _field_statistics(film_fft.Ha, active),
            "heff": _field_statistics(film_fft.Heff, active),
        }
        fft_torques = {
            "hd": _torque_statistics(film_fft.Spin, film_fft.Hd, active),
            "he": _torque_statistics(film_fft.Spin, film_fft.He, active),
            "ha": _torque_statistics(film_fft.Spin, film_fft.Ha, active),
            "heff": _torque_statistics(film_fft.Spin, film_fft.Heff, active),
        }
        fft_alignments = {
            "hd": _alignment_statistics(film_fft.Spin, film_fft.Hd, active),
            "he": _alignment_statistics(film_fft.Spin, film_fft.He, active),
            "ha": _alignment_statistics(film_fft.Spin, film_fft.Ha, active),
            "heff": _alignment_statistics(film_fft.Spin, film_fft.Heff, active),
        }

        fft_spin_active = film_fft.Spin[active]
        unet_spin_active = film_unet.Spin[active]
        fft_m = fft_spin_active.mean(dim=0)
        unet_m = unet_spin_active.mean(dim=0)
        hext_norm = torch.linalg.vector_norm(hext)
        hext_direction = hext / hext_norm if hext_norm > _EPS else hext
        fft_projection = torch.dot(fft_m, hext_direction).item()
        unet_projection = torch.dot(unet_m, hext_direction).item()

        hd_error = _vector_error(film_fft.Hd, film_unet.Hd, active)
        spin_error = _vector_error(film_fft.Spin, film_unet.Spin, active)
        he_error = _vector_error(film_fft.He, film_unet.He, active)
        ha_error = _vector_error(film_fft.Ha, film_unet.Ha, active)
        heff_error = _vector_error(film_fft.Heff, film_unet.Heff, active)

        snapshot = PhysicsSnapshot(
            mh_step=int(mh_step),
            hext_scalar=float(hext_scalar),
            hext_x=float(hext[0].item()),
            hext_y=float(hext[1].item()),
            hext_z=float(hext[2].item()),

            fft_hd_mean=fft_fields["hd"]["mean"], fft_hd_std=fft_fields["hd"]["std"],
            fft_hd_max=fft_fields["hd"]["max"], fft_hd_rms=fft_fields["hd"]["rms"],
            fft_he_mean=fft_fields["he"]["mean"], fft_he_std=fft_fields["he"]["std"],
            fft_he_max=fft_fields["he"]["max"], fft_he_rms=fft_fields["he"]["rms"],
            fft_ha_mean=fft_fields["ha"]["mean"], fft_ha_std=fft_fields["ha"]["std"],
            fft_ha_max=fft_fields["ha"]["max"], fft_ha_rms=fft_fields["ha"]["rms"],
            fft_heff_mean=fft_fields["heff"]["mean"], fft_heff_std=fft_fields["heff"]["std"],
            fft_heff_max=fft_fields["heff"]["max"], fft_heff_rms=fft_fields["heff"]["rms"],

            fft_tau_hd_mean=fft_torques["hd"]["mean"], fft_tau_hd_max=fft_torques["hd"]["max"],
            fft_tau_hd_rms=fft_torques["hd"]["rms"],
            fft_tau_he_mean=fft_torques["he"]["mean"], fft_tau_he_max=fft_torques["he"]["max"],
            fft_tau_he_rms=fft_torques["he"]["rms"],
            fft_tau_ha_mean=fft_torques["ha"]["mean"], fft_tau_ha_max=fft_torques["ha"]["max"],
            fft_tau_ha_rms=fft_torques["ha"]["rms"],
            fft_tau_heff_mean=fft_torques["heff"]["mean"], fft_tau_heff_max=fft_torques["heff"]["max"],
            fft_tau_heff_rms=fft_torques["heff"]["rms"],

            fft_align_hd_mean=fft_alignments["hd"]["mean"], fft_align_hd_abs_mean=fft_alignments["hd"]["abs_mean"],
            fft_align_hd_std=fft_alignments["hd"]["std"],
            fft_align_he_mean=fft_alignments["he"]["mean"], fft_align_he_abs_mean=fft_alignments["he"]["abs_mean"],
            fft_align_he_std=fft_alignments["he"]["std"],
            fft_align_ha_mean=fft_alignments["ha"]["mean"], fft_align_ha_abs_mean=fft_alignments["ha"]["abs_mean"],
            fft_align_ha_std=fft_alignments["ha"]["std"],
            fft_align_heff_mean=fft_alignments["heff"]["mean"], fft_align_heff_abs_mean=fft_alignments["heff"]["abs_mean"],
            fft_align_heff_std=fft_alignments["heff"]["std"],

            fft_e_demag=_energy_value(film_fft, "Energy_demag"),
            fft_e_exchange=_energy_value(film_fft, "Energy_excha"),
            fft_e_anis=_energy_value(film_fft, "Energy_aniso"),
            fft_e_external=_energy_value(film_fft, "Energy_exter"),
            fft_e_total=_energy_value(film_fft, "Energy"),

            fft_mx=float(fft_m[0].item()), fft_my=float(fft_m[1].item()), fft_mz=float(fft_m[2].item()),
            fft_mz_abs_mean=float(fft_spin_active[:, 2].abs().mean().item()),
            fft_m_projection=float(fft_projection),
            fft_winding_abs=_as_float(fft_winding_abs), fft_winding_sum=_as_float(fft_winding_sum),

            unet_mx=float(unet_m[0].item()), unet_my=float(unet_m[1].item()), unet_mz=float(unet_m[2].item()),
            unet_mz_abs_mean=float(unet_spin_active[:, 2].abs().mean().item()),
            unet_m_projection=float(unet_projection),
            unet_winding_abs=_as_float(unet_winding_abs), unet_winding_sum=_as_float(unet_winding_sum),

            hd_mae=hd_error["mae"], hd_rmse=hd_error["rmse"],
            hd_vector_l2_mean=hd_error["vector_l2_mean"], hd_vector_l2_max=hd_error["vector_l2_max"],
            hd_relative_rmse=hd_error["relative_rmse"], hd_cosine_mean=hd_error["cosine_mean"],
            spin_mae=spin_error["mae"], spin_rmse=spin_error["rmse"],
            spin_vector_l2_mean=spin_error["vector_l2_mean"], spin_vector_l2_max=spin_error["vector_l2_max"],
            spin_relative_rmse=spin_error["relative_rmse"], spin_cosine_mean=spin_error["cosine_mean"],
            he_mae=he_error["mae"], ha_mae=ha_error["mae"], heff_mae=heff_error["mae"],
            m_projection_abs_error=abs(float(unet_projection - fft_projection)),

            fft_iterations=int(fft_iterations), unet_iterations=int(unet_iterations),
            fft_final_convergence_error=float(fft_final_convergence_error),
            unet_final_convergence_error=float(unet_final_convergence_error),
            fft_runtime_seconds=float(fft_runtime_seconds), unet_runtime_seconds=float(unet_runtime_seconds),
        )
        self.snapshots.append(snapshot)
        return snapshot

    def dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(snapshot) for snapshot in self.snapshots])

    def save_csv(self, filename) -> pd.DataFrame:
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        df = self.dataframe()
        df.to_csv(path, index=False)
        return df

    def predictor_columns(self) -> Sequence[str]:
        """Return FFT-only scalar predictor columns for Part 2 analysis."""
        excluded = {
            "fft_winding_abs", "fft_winding_sum",  # transition labels/definitions
            "fft_iterations", "fft_final_convergence_error", "fft_runtime_seconds",
        }
        return [
            column for column in self.dataframe().columns
            if column.startswith("fft_") and column not in excluded
        ]

    def predictor_dict(self) -> Dict[str, np.ndarray]:
        df = self.dataframe()
        return {column: df[column].to_numpy(dtype=float) for column in self.predictor_columns()}

    def target_dict(self) -> Dict[str, np.ndarray]:
        df = self.dataframe()
        names = [
            "hd_mae", "hd_rmse", "hd_vector_l2_mean", "hd_vector_l2_max",
            "hd_relative_rmse", "spin_mae", "spin_rmse", "spin_vector_l2_mean",
            "spin_vector_l2_max", "spin_relative_rmse", "m_projection_abs_error",
        ]
        return {name: df[name].to_numpy(dtype=float) for name in names if name in df}


# ============================================================================
# MODEL INTROSPECTION (optional debugging helper)
# ============================================================================

def inspect_model_quantities(model, name_filter=None):
    """Print numerical quantities currently stored on a MAG2305 model."""
    print("\n" + "=" * 75)
    print(f"{'DISCOVERING AVAILABLE MODEL QUANTITIES':^75}")
    print("=" * 75)
    for name in sorted(dir(model)):
        if name.startswith("_") or (name_filter and name_filter.lower() not in name.lower()):
            continue
        try:
            value = getattr(model, name)
        except Exception:
            continue
        if callable(value):
            continue
        if isinstance(value, torch.Tensor):
            type_str, info_str = f"Tensor({value.device})", f"shape={tuple(value.shape)}"
        elif isinstance(value, np.ndarray):
            type_str, info_str = "ndarray", f"shape={value.shape}"
        elif np.isscalar(value):
            type_str, info_str = "scalar", f"value={value}"
        else:
            continue
        print(f"{'name: ' + name:32s} | type: {type_str:15s} | {info_str}")
    print("=" * 75)


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