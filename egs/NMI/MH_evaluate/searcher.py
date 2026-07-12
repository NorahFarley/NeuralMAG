# -*- coding: utf-8 -*-
"""
Physics-aware transition and leading-indicator analysis for NeuralMAG M-H sweeps.

Pipeline
--------
Part 1: PhysicsRecorder
    Records one converged FFT/UNet state per external-field step.
Part 2: TransitionAnalyzer
    Detects FFT-defined topology/magnetization transition events and labels rows.
Part 3: LeadingIndicatorAnalyzer
    Ranks independent FFT quantities by advance warning and future UNet error.
Part 4: PublicationFigureGenerator
    Generates reproducible manuscript figures and panel-data files.
Part 5: ManuscriptOutputGenerator / CrossSweepAggregator
    Exports manuscript tables, run records, and cross-run stability summaries.

The transition definition never uses UNet error. UNet quantities are targets and
accuracy diagnostics only. This avoids defining the physical events by the model
failure that the analysis is intended to explain.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.ndimage import label as connected_components
from scipy.stats import mannwhitneyu, pearsonr, spearmanr

try:
    from sklearn.feature_selection import mutual_info_regression
    from sklearn.metrics import roc_auc_score
except Exception:  # pragma: no cover - optional dependency fallback
    mutual_info_regression = None
    roc_auc_score = None

_EPS = 1.0e-12


# =============================================================================
# General helpers
# =============================================================================


def _as_float(value: Any, default: float = np.nan) -> float:
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


def _as_numpy(value: Any, dtype=float) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().astype(dtype, copy=False)
    return np.asarray(value, dtype=dtype)


def _finite(values: Sequence[float]) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    return x[np.isfinite(x)]


def _safe_std(values: Sequence[float]) -> float:
    x = _finite(values)
    if x.size < 2:
        return 0.0
    return float(np.std(x, ddof=0))


def _safe_mean(values: Sequence[float]) -> float:
    x = _finite(values)
    return float(np.mean(x)) if x.size else np.nan


def _safe_median(values: Sequence[float]) -> float:
    x = _finite(values)
    return float(np.median(x)) if x.size else np.nan


def _robust_scale(values: Sequence[float]) -> Tuple[float, float]:
    x = _finite(values)
    if x.size == 0:
        return 0.0, 1.0
    center = float(np.median(x))
    mad = float(np.median(np.abs(x - center)))
    scale = 1.4826 * mad
    if scale <= _EPS:
        scale = float(np.std(x))
    if scale <= _EPS:
        scale = 1.0
    return center, scale


def _robust_z(values: Sequence[float]) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    center, scale = _robust_scale(x)
    return (x - center) / scale


def _masked_values(values: torch.Tensor, active_mask: torch.Tensor) -> torch.Tensor:
    selected = values[active_mask]
    return selected if selected.numel() else values.reshape(-1)


def _field_statistics(field: torch.Tensor, active_mask: torch.Tensor) -> Dict[str, float]:
    magnitudes = _masked_values(torch.linalg.vector_norm(field, dim=-1), active_mask)
    return {
        "mean": float(magnitudes.mean().item()),
        "std": float(magnitudes.std(unbiased=False).item()),
        "max": float(magnitudes.max().item()),
        "rms": float(torch.sqrt(torch.mean(magnitudes.square())).item()),
    }


def _torque_statistics(spin: torch.Tensor, field: torch.Tensor,
                       active_mask: torch.Tensor) -> Dict[str, float]:
    magnitude = torch.linalg.vector_norm(torch.cross(spin, field, dim=-1), dim=-1)
    magnitude = _masked_values(magnitude, active_mask)
    return {
        "mean": float(magnitude.mean().item()),
        "max": float(magnitude.max().item()),
        "rms": float(torch.sqrt(torch.mean(magnitude.square())).item()),
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


def _energy_value(model: Any, attribute: str) -> float:
    return _as_float(getattr(model, attribute, None))


def _vector_error(reference: torch.Tensor, prediction: torch.Tensor,
                  active_mask: torch.Tensor) -> Dict[str, float]:
    difference = prediction - reference
    diff_active = difference[active_mask]
    ref_active = reference[active_mask]
    pred_active = prediction[active_mask]
    if diff_active.numel() == 0:
        raise ValueError("No active cells are available for the FFT/UNet error calculation.")

    vector_l2 = torch.linalg.vector_norm(diff_active, dim=-1)
    ref_norm = torch.linalg.vector_norm(ref_active, dim=-1)
    pred_norm = torch.linalg.vector_norm(pred_active, dim=-1)
    cosine = torch.sum(ref_active * pred_active, dim=-1) / (ref_norm * pred_norm + _EPS)
    rmse = torch.sqrt(torch.mean(diff_active.square()))
    denom = torch.sqrt(torch.mean(ref_active.square())) + _EPS
    return {
        "mae": float(diff_active.abs().mean().item()),
        "rmse": float(rmse.item()),
        "vector_l2_mean": float(vector_l2.mean().item()),
        "vector_l2_max": float(vector_l2.max().item()),
        "relative_rmse": float((rmse / denom).item()),
        "cosine_mean": float(cosine.mean().item()),
    }


def _safe_pearson(x: Sequence[float], y: Sequence[float]) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3 or np.std(x[mask]) <= _EPS or np.std(y[mask]) <= _EPS:
        return 0.0
    return float(pearsonr(x[mask], y[mask])[0])


def _safe_spearman(x: Sequence[float], y: Sequence[float]) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3 or np.std(x[mask]) <= _EPS or np.std(y[mask]) <= _EPS:
        return 0.0
    return float(spearmanr(x[mask], y[mask]).statistic)


def _auc(labels: Sequence[bool], scores: Sequence[float]) -> float:
    y = np.asarray(labels, dtype=int)
    s = np.asarray(scores, dtype=float)
    mask = np.isfinite(s)
    y, s = y[mask], s[mask]
    if y.size < 2 or np.unique(y).size < 2 or np.std(s) <= _EPS:
        return 0.5
    if roc_auc_score is not None:
        return float(roc_auc_score(y, s))
    # Mann-Whitney equivalence fallback.
    pos = s[y == 1]
    neg = s[y == 0]
    u = mannwhitneyu(pos, neg, alternative="two-sided").statistic
    return float(u / (len(pos) * len(neg)))


def _cohens_d(group1: Sequence[float], group0: Sequence[float]) -> float:
    a = _finite(group1)
    b = _finite(group0)
    if a.size < 2 or b.size < 2:
        return 0.0
    pooled_num = (a.size - 1) * np.var(a, ddof=1) + (b.size - 1) * np.var(b, ddof=1)
    pooled_den = a.size + b.size - 2
    pooled = math.sqrt(max(pooled_num / max(pooled_den, 1), 0.0))
    return float((np.mean(a) - np.mean(b)) / pooled) if pooled > _EPS else 0.0


def _bh_fdr(p_values: Sequence[float]) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    q = np.full_like(p, np.nan)
    valid = np.isfinite(p)
    if not valid.any():
        return q
    pv = p[valid]
    order = np.argsort(pv)
    ranked = pv[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    restored = np.empty_like(adjusted)
    restored[order] = adjusted
    q[valid] = restored
    return q


def _moving_block_indices(n: int, block_size: int, rng: np.random.Generator) -> np.ndarray:
    if n <= 0:
        return np.array([], dtype=int)
    block_size = max(1, min(block_size, n))
    chunks: List[np.ndarray] = []
    while sum(len(chunk) for chunk in chunks) < n:
        start = int(rng.integers(0, n))
        chunks.append((start + np.arange(block_size)) % n)
    return np.concatenate(chunks)[:n]


def _detrend_against_field(values: Sequence[float], hext: Sequence[float],
                           fit_mask: Sequence[bool], degree: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    y = np.asarray(values, dtype=float)
    x = np.asarray(hext, dtype=float)
    mask = np.asarray(fit_mask, dtype=bool) & np.isfinite(x) & np.isfinite(y)
    if mask.sum() < degree + 2 or np.unique(x[mask]).size < degree + 1:
        trend = np.full_like(y, np.nanmedian(y))
        return y - trend, trend
    coeff = np.polyfit(x[mask], y[mask], deg=degree)
    trend = np.polyval(coeff, x)
    return y - trend, trend


def _predictor_group(name: str) -> str:
    if "tau_" in name:
        return "Torque"
    if "align_" in name:
        return "Alignment"
    if "e_" in name:
        return "Energy"
    if any(token in name for token in ("_hd_", "_he_", "_ha_", "_heff_")):
        return "Field"
    if "winding" in name or "core" in name:
        return "Topology"
    if "_m" in name:
        return "Magnetization"
    return "Other"


def _friendly_name(name: str) -> str:
    text = name.replace("fft_", "").replace("unet_", "").replace("_", " ")
    replacements = {
        "hd": "Hdemag", "he": "Hexchange", "ha": "Hanisotropy",
        "heff": "Heffective", "tau": "torque", "rms": "RMS",
        "mae": "MAE", "rmse": "RMSE",
    }
    words = [replacements.get(word, word) for word in text.split()]
    return " ".join(words).title().replace("Rms", "RMS").replace("Mae", "MAE").replace("Rmse", "RMSE")


def _save_figure(fig: plt.Figure, base_path: Path, formats: Sequence[str], dpi: int) -> List[str]:
    paths: List[str] = []
    base_path.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fmt = fmt.lower().strip().lstrip(".")
        if not fmt:
            continue
        path = base_path.with_suffix(f".{fmt}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        paths.append(str(path))
    plt.close(fig)
    return paths


def _export_dataframe(df: pd.DataFrame, base_path: Path, formats: Sequence[str]) -> List[str]:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    outputs: List[str] = []
    for fmt in formats:
        fmt = fmt.lower().strip().lstrip(".")
        path = base_path.with_suffix(f".{fmt}")
        if fmt == "csv":
            df.to_csv(path, index=False)
        elif fmt in {"tex", "latex"}:
            path = base_path.with_suffix(".tex")
            path.write_text(df.to_latex(index=False, escape=True), encoding="utf-8")
        elif fmt in {"md", "markdown"}:
            path = base_path.with_suffix(".md")
            try:
                text = df.to_markdown(index=False)
            except Exception:
                text = df.to_csv(index=False)
            path.write_text(text + "\n", encoding="utf-8")
        else:
            continue
        outputs.append(str(path))
    return outputs


def _sha256_file(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    file_path = Path(path)
    if not file_path.is_file():
        return None
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# =============================================================================
# Part 1: topology and converged-state recorder
# =============================================================================


def analyze_winding_components(
    winding_map: Any,
    *,
    relative_threshold: float = 0.25,
    absolute_threshold: float = 0.02,
    min_cells: int = 1,
    min_abs_charge: float = 0.05,
) -> Dict[str, float]:
    """Count spatially distinct positive/negative winding-density components.

    This is a connected-component diagnostic, not an assertion that every
    component is a mathematically exact vortex core. The thresholds are saved
    so their sensitivity can be audited.
    """
    array = np.squeeze(_as_numpy(winding_map, dtype=float))
    if array.ndim != 2:
        raise ValueError(f"Expected a 2-D winding map after squeeze; got shape {array.shape}.")
    max_abs = float(np.nanmax(np.abs(array))) if array.size else 0.0
    threshold = max(float(absolute_threshold), float(relative_threshold) * max_abs)
    structure = np.ones((3, 3), dtype=int)

    result: Dict[str, float] = {
        "positive_core_count": 0,
        "negative_core_count": 0,
        "total_core_count": 0,
        "positive_core_charge": 0.0,
        "negative_core_charge": 0.0,
        "core_abs_charge": 0.0,
        "core_area_cells": 0,
        "winding_max_abs": max_abs,
        "core_threshold_used": threshold,
    }

    for sign_name, mask in (
        ("positive", array >= threshold),
        ("negative", array <= -threshold),
    ):
        labels, count = connected_components(mask, structure=structure)
        accepted = 0
        signed_charge = 0.0
        abs_charge = 0.0
        area = 0
        for component_id in range(1, count + 1):
            component = labels == component_id
            component_area = int(component.sum())
            component_charge = float(array[component].sum())
            component_abs_charge = float(np.abs(array[component]).sum())
            if component_area < int(min_cells) or component_abs_charge < float(min_abs_charge):
                continue
            accepted += 1
            signed_charge += component_charge
            abs_charge += component_abs_charge
            area += component_area
        result[f"{sign_name}_core_count"] = accepted
        result[f"{sign_name}_core_charge"] = signed_charge
        result["core_abs_charge"] += abs_charge
        result["core_area_cells"] += area

    result["total_core_count"] = int(result["positive_core_count"] + result["negative_core_count"])
    return result


@dataclass(frozen=True)
class PhysicsSnapshot:
    """Flexible one-row container for a converged external-field state."""

    values: Mapping[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.values)

    def __getitem__(self, key: str) -> Any:
        return self.values[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.values.get(key, default)

    def __getattr__(self, name: str) -> Any:
        try:
            return self.values[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


class PhysicsRecorder:
    """Collect one converged FFT/UNet snapshot per M-H field step."""

    def __init__(self) -> None:
        self.snapshots: List[PhysicsSnapshot] = []

    def __len__(self) -> int:
        return len(self.snapshots)

    @torch.no_grad()
    def capture(
        self,
        *,
        mh_step: int,
        hext_scalar: float,
        hext_vector: Sequence[float],
        projection_direction: Sequence[float],
        film_fft: Any,
        film_unet: Any,
        fft_winding_abs: Any,
        fft_winding_sum: Any,
        unet_winding_abs: Any,
        unet_winding_sum: Any,
        fft_topology: Optional[Mapping[str, Any]] = None,
        unet_topology: Optional[Mapping[str, Any]] = None,
        fft_iterations: int = 0,
        unet_iterations: int = 0,
        fft_final_convergence_error: float = np.nan,
        unet_final_convergence_error: float = np.nan,
        fft_runtime_seconds: float = np.nan,
        unet_runtime_seconds: float = np.nan,
        cell_count: Optional[int] = None,
    ) -> PhysicsSnapshot:
        device = film_fft.Spin.device
        dtype = film_fft.Spin.dtype
        hext = torch.as_tensor(hext_vector, dtype=dtype, device=device).reshape(3)
        projection = torch.as_tensor(projection_direction, dtype=dtype, device=device).reshape(3)
        projection = projection / (torch.linalg.vector_norm(projection) + _EPS)

        active_fft = torch.linalg.vector_norm(film_fft.Spin, dim=-1) > _EPS
        active_unet = torch.linalg.vector_norm(film_unet.Spin, dim=-1) > _EPS
        active = active_fft & active_unet
        if not torch.any(active):
            raise ValueError("No active magnetic cells were found while recording physics.")
        n_active = int(active.sum().item())
        if cell_count is not None and int(cell_count) != n_active:
            print(f"[PhysicsRecorder] tensor mask has {n_active} active cells; supplied cell_count={cell_count}.")

        data: Dict[str, Any] = {
            "mh_step": int(mh_step),
            "hext_scalar": float(hext_scalar),
            "hext_x": float(hext[0].item()),
            "hext_y": float(hext[1].item()),
            "hext_z": float(hext[2].item()),
        }

        fields = {"hd": "Hd", "he": "He", "ha": "Ha", "heff": "Heff"}
        for short, attr in fields.items():
            stats = _field_statistics(getattr(film_fft, attr), active)
            for stat, value in stats.items():
                data[f"fft_{short}_{stat}"] = value
            torque = _torque_statistics(film_fft.Spin, getattr(film_fft, attr), active)
            for stat, value in torque.items():
                data[f"fft_tau_{short}_{stat}"] = value
            alignment = _alignment_statistics(film_fft.Spin, getattr(film_fft, attr), active)
            for stat, value in alignment.items():
                data[f"fft_align_{short}_{stat}"] = value

        for prefix, model in (("fft", film_fft), ("unet", film_unet)):
            data[f"{prefix}_e_demag"] = _energy_value(model, "Energy_demag")
            data[f"{prefix}_e_exchange"] = _energy_value(model, "Energy_excha")
            data[f"{prefix}_e_anis"] = _energy_value(model, "Energy_aniso")
            data[f"{prefix}_e_external"] = _energy_value(model, "Energy_exter")
            data[f"{prefix}_e_total"] = _energy_value(model, "Energy")

        fft_spin = film_fft.Spin[active]
        unet_spin = film_unet.Spin[active]
        for prefix, spin in (("fft", fft_spin), ("unet", unet_spin)):
            magnetization = spin.mean(dim=0)
            data[f"{prefix}_mx"] = float(magnetization[0].item())
            data[f"{prefix}_my"] = float(magnetization[1].item())
            data[f"{prefix}_mz"] = float(magnetization[2].item())
            data[f"{prefix}_mz_abs_mean"] = float(spin[:, 2].abs().mean().item())
            data[f"{prefix}_m_projection"] = float(torch.dot(magnetization, projection).item())

        data.update({
            "fft_winding_abs": _as_float(fft_winding_abs),
            "fft_winding_sum": _as_float(fft_winding_sum),
            "unet_winding_abs": _as_float(unet_winding_abs),
            "unet_winding_sum": _as_float(unet_winding_sum),
        })
        for prefix, topology in (("fft", fft_topology or {}), ("unet", unet_topology or {})):
            for key, value in topology.items():
                data[f"{prefix}_{key}"] = _as_float(value)

        errors = {
            "hd": _vector_error(film_fft.Hd, film_unet.Hd, active),
            "spin": _vector_error(film_fft.Spin, film_unet.Spin, active),
            "he": _vector_error(film_fft.He, film_unet.He, active),
            "ha": _vector_error(film_fft.Ha, film_unet.Ha, active),
            "heff": _vector_error(film_fft.Heff, film_unet.Heff, active),
        }
        for target in ("hd", "spin"):
            for metric, value in errors[target].items():
                data[f"{target}_{metric}"] = value
        data["he_mae"] = errors["he"]["mae"]
        data["ha_mae"] = errors["ha"]["mae"]
        data["heff_mae"] = errors["heff"]["mae"]
        data["m_projection_abs_error"] = abs(data["unet_m_projection"] - data["fft_m_projection"])

        data.update({
            "fft_iterations": int(fft_iterations),
            "unet_iterations": int(unet_iterations),
            "fft_final_convergence_error": float(fft_final_convergence_error),
            "unet_final_convergence_error": float(unet_final_convergence_error),
            "fft_runtime_seconds": float(fft_runtime_seconds),
            "unet_runtime_seconds": float(unet_runtime_seconds),
        })
        snapshot = PhysicsSnapshot(data)
        self.snapshots.append(snapshot)
        return snapshot

    def dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([snapshot.to_dict() for snapshot in self.snapshots])

    def save_csv(self, filename: os.PathLike[str] | str) -> pd.DataFrame:
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        df = self.dataframe()
        df.to_csv(path, index=False)
        return df

    def predictor_columns(self, independent_only: bool = False) -> List[str]:
        df = self.dataframe()
        columns = [column for column in df.columns if column.startswith("fft_")]
        solver = {"fft_iterations", "fft_final_convergence_error", "fft_runtime_seconds"}
        columns = [column for column in columns if column not in solver]
        if independent_only:
            columns = [column for column in columns if column not in LeadingIndicatorAnalyzer.CIRCULAR_COLUMNS]
        return columns

    def predictor_dict(self, independent_only: bool = False) -> Dict[str, np.ndarray]:
        df = self.dataframe()
        return {name: df[name].to_numpy(dtype=float) for name in self.predictor_columns(independent_only)}


# =============================================================================
# Part 2: transition detection and row labels
# =============================================================================


class TransitionAnalyzer:
    """Detect transition events from FFT topology and FFT magnetization only."""

    EVENT_COLUMNS = [
        "event_id", "event_kind", "event_source", "confidence", "baseline_step",
        "onset_step", "peak_step", "end_step", "span_steps", "baseline_hext",
        "onset_hext", "peak_hext", "end_hext", "field_span_oe",
        "delta_positive_cores", "delta_negative_cores", "delta_total_cores",
        "delta_winding_abs", "delta_winding_sum", "delta_m_projection",
        "peak_transition_score", "integrated_transition_score", "peak_spin_mae",
        "peak_hd_mae",
    ]

    def __init__(
        self,
        physics_df: pd.DataFrame,
        *,
        merge_gap: int = 1,
        winding_tolerance: float = 0.0,
        magnetization_z_threshold: float = 3.0,
        min_magnetization_change: float = 0.02,
        pretransition_windows: Sequence[int] = (3, 5, 10, 20),
    ) -> None:
        self.df = physics_df.sort_values("mh_step").reset_index(drop=True).copy()
        self.merge_gap = int(max(0, merge_gap))
        self.winding_tolerance = float(max(0.0, winding_tolerance))
        self.magnetization_z_threshold = float(magnetization_z_threshold)
        self.min_magnetization_change = float(max(0.0, min_magnetization_change))
        self.pretransition_windows = tuple(sorted(set(int(w) for w in pretransition_windows if int(w) > 0)))

    @staticmethod
    def _classify(dp: float, dn: float, dt: float, dw_abs: float,
                  dw_sum: float, dm: float, topology: bool, magnetization: bool) -> str:
        if dp > 0 and dn > 0:
            return "pair_nucleation"
        if dp < 0 and dn < 0:
            return "pair_annihilation"
        if dp > 0 and dn == 0:
            return "positive_core_nucleation"
        if dn > 0 and dp == 0:
            return "negative_core_nucleation"
        if dp < 0 and dn == 0:
            return "positive_core_annihilation"
        if dn < 0 and dp == 0:
            return "negative_core_annihilation"
        if dt > 0:
            return "core_creation_or_splitting"
        if dt < 0:
            return "core_annihilation_or_merging"
        if topology and abs(dw_sum) > _EPS:
            return "core_polarity_reconfiguration"
        if topology and abs(dw_abs) > _EPS:
            return "topological_reconfiguration"
        if magnetization and abs(dm) > _EPS:
            return "magnetization_switching"
        return "mixed_transition"

    def _step_changes(self) -> pd.DataFrame:
        df = self.df
        changes = pd.DataFrame({"mh_step": df["mh_step"], "hext_scalar": df["hext_scalar"]})
        source_columns = {
            "positive_cores": "fft_positive_core_count",
            "negative_cores": "fft_negative_core_count",
            "total_cores": "fft_total_core_count",
            "winding_abs": "fft_winding_abs",
            "winding_sum": "fft_winding_sum",
            "winding_max_abs": "fft_winding_max_abs",
            "m_projection": "fft_m_projection",
        }
        for short, column in source_columns.items():
            values = df[column].to_numpy(dtype=float) if column in df else np.zeros(len(df))
            changes[f"delta_{short}"] = np.diff(values, prepend=values[0])

        abs_dm = np.abs(changes["delta_m_projection"].to_numpy(dtype=float))
        center, scale = _robust_scale(abs_dm[1:])
        changes["magnetization_change_z"] = (abs_dm - center) / scale
        max_winding_delta = np.abs(changes["delta_winding_max_abs"].to_numpy(dtype=float))
        changes["winding_max_change_z"] = _robust_z(max_winding_delta)

        topology_candidate = (
            np.abs(changes["delta_positive_cores"]) > 0
        ) | (
            np.abs(changes["delta_negative_cores"]) > 0
        ) | (
            np.abs(changes["delta_total_cores"]) > 0
        ) | (
            np.abs(changes["delta_winding_abs"]) > self.winding_tolerance
        ) | (
            np.abs(changes["delta_winding_sum"]) > self.winding_tolerance
        )
        magnetization_candidate = (
            changes["magnetization_change_z"] >= self.magnetization_z_threshold
        ) & (
            abs_dm >= self.min_magnetization_change
        )
        changes["topology_candidate"] = topology_candidate
        changes["magnetization_candidate"] = magnetization_candidate
        changes["is_candidate"] = topology_candidate | magnetization_candidate
        changes.loc[0, ["topology_candidate", "magnetization_candidate", "is_candidate"]] = False

        changes["transition_score"] = (
            np.abs(changes["delta_positive_cores"])
            + np.abs(changes["delta_negative_cores"])
            + np.abs(changes["delta_total_cores"])
            + np.abs(changes["delta_winding_abs"])
            + 0.5 * np.abs(changes["delta_winding_sum"])
            + np.maximum(changes["magnetization_change_z"] - self.magnetization_z_threshold, 0.0)
        )
        return changes

    def _group_candidates(self, candidate_indices: np.ndarray) -> List[List[int]]:
        if candidate_indices.size == 0:
            return []
        groups: List[List[int]] = [[int(candidate_indices[0])]]
        max_separation = self.merge_gap + 1
        for index in candidate_indices[1:]:
            index = int(index)
            if index - groups[-1][-1] <= max_separation:
                groups[-1].append(index)
            else:
                groups.append([index])
        return groups

    def analyze(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        changes = self._step_changes()
        groups = self._group_candidates(np.flatnonzero(changes["is_candidate"].to_numpy(dtype=bool)))
        events: List[Dict[str, Any]] = []
        for event_id, group in enumerate(groups, start=1):
            onset, end = group[0], group[-1]
            baseline = max(0, onset - 1)
            group_slice = changes.loc[onset:end]
            peak = int(group_slice["transition_score"].idxmax())

            def delta(column: str) -> float:
                if column not in self.df:
                    return 0.0
                return float(self.df.loc[end, column] - self.df.loc[baseline, column])

            dp = delta("fft_positive_core_count")
            dn = delta("fft_negative_core_count")
            dt = delta("fft_total_core_count")
            dwa = delta("fft_winding_abs")
            dws = delta("fft_winding_sum")
            dm = delta("fft_m_projection")
            topology = bool(changes.loc[onset:end, "topology_candidate"].any())
            magnetization = bool(changes.loc[onset:end, "magnetization_candidate"].any())
            kind = self._classify(dp, dn, dt, dwa, dws, dm, topology, magnetization)
            source = "+".join(part for part, present in (("topology", topology), ("magnetization", magnetization)) if present)
            confidence = "high" if abs(dp) + abs(dn) + abs(dt) > 0 else ("medium" if topology else "low")
            events.append({
                "event_id": event_id,
                "event_kind": kind,
                "event_source": source,
                "confidence": confidence,
                "baseline_step": baseline,
                "onset_step": onset,
                "peak_step": peak,
                "end_step": end,
                "span_steps": end - onset + 1,
                "baseline_hext": float(self.df.loc[baseline, "hext_scalar"]),
                "onset_hext": float(self.df.loc[onset, "hext_scalar"]),
                "peak_hext": float(self.df.loc[peak, "hext_scalar"]),
                "end_hext": float(self.df.loc[end, "hext_scalar"]),
                "field_span_oe": abs(float(self.df.loc[end, "hext_scalar"] - self.df.loc[onset, "hext_scalar"])),
                "delta_positive_cores": dp,
                "delta_negative_cores": dn,
                "delta_total_cores": dt,
                "delta_winding_abs": dwa,
                "delta_winding_sum": dws,
                "delta_m_projection": dm,
                "peak_transition_score": float(changes.loc[peak, "transition_score"]),
                "integrated_transition_score": float(group_slice["transition_score"].sum()),
                "peak_spin_mae": float(self.df.loc[onset:end, "spin_mae"].max()) if "spin_mae" in self.df else np.nan,
                "peak_hd_mae": float(self.df.loc[onset:end, "hd_mae"].max()) if "hd_mae" in self.df else np.nan,
            })
        events_df = pd.DataFrame(events, columns=self.EVENT_COLUMNS)

        labeled = self.df.copy()
        labeled["transition_score"] = changes["transition_score"].to_numpy(dtype=float)
        labeled["transition_event_id"] = 0
        labeled["transition_kind"] = "none"
        labeled["transition_source"] = "none"
        labeled["is_transition_step"] = False
        labeled["is_event_onset"] = False

        onsets: List[int] = []
        for event in events:
            onset, end = int(event["onset_step"]), int(event["end_step"])
            onsets.append(onset)
            labeled.loc[onset:end, "transition_event_id"] = int(event["event_id"])
            labeled.loc[onset:end, "transition_kind"] = str(event["event_kind"])
            labeled.loc[onset:end, "transition_source"] = str(event["event_source"])
            labeled.loc[onset:end, "is_transition_step"] = True
            labeled.loc[onset, "is_event_onset"] = True

        n = len(labeled)
        if onsets:
            onset_array = np.asarray(onsets, dtype=int)
            labeled["steps_to_nearest_event"] = [int(np.min(np.abs(onset_array - i))) for i in range(n)]
            next_steps: List[float] = []
            next_kinds: List[str] = []
            kind_by_onset = {int(row["onset_step"]): str(row["event_kind"]) for _, row in events_df.iterrows()}
            for i in range(n):
                future = onset_array[onset_array >= i]
                if future.size:
                    next_onset = int(future[0])
                    next_steps.append(float(next_onset - i))
                    next_kinds.append(kind_by_onset[next_onset])
                else:
                    next_steps.append(np.nan)
                    next_kinds.append("none")
            labeled["steps_to_next_event"] = next_steps
            labeled["next_event_kind"] = next_kinds
        else:
            labeled["steps_to_nearest_event"] = np.nan
            labeled["steps_to_next_event"] = np.nan
            labeled["next_event_kind"] = "none"

        for window in self.pretransition_windows:
            flag = np.zeros(n, dtype=bool)
            for onset in onsets:
                flag[max(0, onset - window):onset] = True
            flag[labeled["is_transition_step"].to_numpy(dtype=bool)] = False
            labeled[f"is_pretransition_{window}"] = flag
        return events_df, labeled, changes

    def run(self, save_path: os.PathLike[str] | str,
            general_title: str = "") -> Tuple[pd.DataFrame, pd.DataFrame]:
        folder = Path(save_path)
        folder.mkdir(parents=True, exist_ok=True)
        events_df, labeled_df, changes_df = self.analyze()
        events_df.to_csv(folder / "transition_events.csv", index=False)
        labeled_df.to_csv(folder / "physics_snapshots_labeled.csv", index=False)
        changes_df.to_csv(folder / "transition_step_changes.csv", index=False)

        fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=True)
        h = labeled_df["hext_scalar"].to_numpy(dtype=float)
        axes[0].plot(h, labeled_df["fft_m_projection"], label="FFT")
        axes[0].plot(h, labeled_df["unet_m_projection"], label="UNet", alpha=0.8)
        axes[0].set_ylabel("Projected magnetization")
        axes[0].legend()

        for column, label in (("fft_positive_core_count", "Positive"),
                              ("fft_negative_core_count", "Negative"),
                              ("fft_total_core_count", "Total")):
            if column in labeled_df:
                axes[1].step(h, labeled_df[column], where="mid", label=label)
        axes[1].set_ylabel("Core components")
        axes[1].legend()

        axes[2].plot(h, labeled_df["fft_winding_abs"], label="Absolute winding")
        axes[2].plot(h, labeled_df["fft_winding_sum"], label="Net winding")
        axes[2].set_ylabel("Winding")
        axes[2].legend()

        axes[3].plot(h, changes_df["transition_score"], label="Transition score")
        if "spin_mae" in labeled_df:
            error = labeled_df["spin_mae"].to_numpy(dtype=float)
            scaled = error / (np.nanmax(error) + _EPS) * (np.nanmax(changes_df["transition_score"]) + _EPS)
            axes[3].plot(h, scaled, label="Spin MAE (rescaled)", alpha=0.7)
        axes[3].set_ylabel("Score")
        axes[3].set_xlabel("External field (Oe)")
        axes[3].legend()

        for _, event in events_df.iterrows():
            x = event["onset_hext"]
            for axis in axes:
                axis.axvline(x, linestyle="--", alpha=0.45)
        for axis in axes:
            axis.grid(alpha=0.25)
        fig.suptitle((general_title + "\n" if general_title else "") + "FFT-defined transition-event summary")
        fig.tight_layout()
        fig.savefig(folder / "transition_event_summary.png", dpi=250, bbox_inches="tight")
        plt.close(fig)
        return events_df, labeled_df


# =============================================================================
# Part 3: statistical leading-indicator ranking
# =============================================================================


class LeadingIndicatorAnalyzer:
    """Rank FFT physical quantities as advance warnings of transitions/errors."""

    CIRCULAR_COLUMNS = {
        "fft_m_projection", "fft_mx", "fft_my", "fft_e_external",
        "fft_winding_abs", "fft_winding_sum", "fft_winding_max_abs",
        "fft_positive_core_count", "fft_negative_core_count", "fft_total_core_count",
        "fft_positive_core_charge", "fft_negative_core_charge", "fft_core_abs_charge",
        "fft_core_area_cells", "fft_core_threshold_used",
    }
    SOLVER_COLUMNS = {"fft_iterations", "fft_final_convergence_error", "fft_runtime_seconds"}

    def __init__(
        self,
        labeled_df: pd.DataFrame,
        *,
        events_df: Optional[pd.DataFrame] = None,
        error_targets: Sequence[str] = ("spin_mae", "hd_mae"),
        pretransition_windows: Sequence[int] = (3, 5, 10, 20),
        primary_window: int = 10,
        max_lag: int = 20,
        lead_z_threshold: float = 1.5,
        post_event_exclusion: int = 3,
        n_permutations: int = 500,
        n_bootstrap: int = 500,
        random_seed: int = 1234,
    ) -> None:
        self.df = labeled_df.sort_values("mh_step").reset_index(drop=True).copy()
        self.events_df = events_df.copy() if events_df is not None else pd.DataFrame()
        self.error_targets = tuple(target for target in error_targets if target in self.df)
        self.windows = tuple(sorted(set(int(w) for w in pretransition_windows if int(w) > 0)))
        self.primary_window = int(primary_window)
        if self.primary_window not in self.windows:
            self.windows = tuple(sorted(set(self.windows + (self.primary_window,))))
        self.max_lag = int(max(0, max_lag))
        self.lead_z_threshold = float(lead_z_threshold)
        self.post_event_exclusion = int(max(0, post_event_exclusion))
        self.n_permutations = int(max(0, n_permutations))
        self.n_bootstrap = int(max(0, n_bootstrap))
        self.rng = np.random.default_rng(int(random_seed))
        self.random_seed = int(random_seed)

    def _candidate_columns(self) -> List[str]:
        columns: List[str] = []
        for column in self.df.columns:
            if not column.startswith("fft_") or column in self.SOLVER_COLUMNS:
                continue
            values = pd.to_numeric(self.df[column], errors="coerce")
            if values.notna().sum() < 3 or values.std(ddof=0) <= _EPS:
                continue
            columns.append(column)
        return columns

    def _quiet_mask(self, max_window: Optional[int] = None) -> np.ndarray:
        n = len(self.df)
        excluded = self.df.get("is_transition_step", pd.Series(False, index=self.df.index)).to_numpy(dtype=bool)
        max_window = max_window if max_window is not None else max(self.windows, default=0)
        onsets = self.events_df.get("onset_step", pd.Series(dtype=int)).to_numpy(dtype=int)
        for onset in onsets:
            excluded[max(0, onset - max_window):min(n, onset + self.post_event_exclusion + 1)] = True
        return ~excluded

    def _labels(self, window: int) -> np.ndarray:
        column = f"is_pretransition_{window}"
        if column in self.df:
            return self.df[column].to_numpy(dtype=bool)
        labels = np.zeros(len(self.df), dtype=bool)
        for onset in self.events_df.get("onset_step", pd.Series(dtype=int)).to_numpy(dtype=int):
            labels[max(0, onset - window):onset] = True
        return labels

    def _permutation_p(self, labels: np.ndarray, scores: np.ndarray, observed_auc: float) -> float:
        if self.n_permutations <= 0 or np.unique(labels).size < 2:
            return np.nan
        exceed = 0
        n = len(scores)
        for _ in range(self.n_permutations):
            shift = int(self.rng.integers(1, max(n, 2)))
            perm_auc = _auc(labels, np.roll(scores, shift))
            if perm_auc >= observed_auc - 1e-15:
                exceed += 1
        return (exceed + 1.0) / (self.n_permutations + 1.0)

    def _bootstrap_auc(self, labels: np.ndarray, scores: np.ndarray) -> Tuple[float, float]:
        if self.n_bootstrap <= 0 or np.unique(labels).size < 2:
            return np.nan, np.nan
        estimates: List[float] = []
        block = max(2, int(round(math.sqrt(len(scores)))))
        for _ in range(self.n_bootstrap):
            idx = _moving_block_indices(len(scores), block, self.rng)
            if np.unique(labels[idx]).size < 2:
                continue
            estimates.append(_auc(labels[idx], scores[idx]))
        if not estimates:
            return np.nan, np.nan
        return float(np.percentile(estimates, 2.5)), float(np.percentile(estimates, 97.5))

    def _event_lead_times(self, oriented_residual: np.ndarray, predictor: str) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        window = max(self.windows, default=self.primary_window)
        for _, event in self.events_df.iterrows():
            onset = int(event["onset_step"])
            baseline_lo = max(0, onset - 2 * window)
            baseline_hi = max(baseline_lo + 1, onset - window)
            search_lo = max(0, onset - window)
            baseline = oriented_residual[baseline_lo:baseline_hi]
            center, scale = _robust_scale(baseline)
            z = (oriented_residual[search_lo:onset] - center) / scale
            run = 0
            for value in z[::-1]:
                if np.isfinite(value) and value >= self.lead_z_threshold:
                    run += 1
                else:
                    break
            rows.append({
                "predictor": predictor,
                "event_id": int(event["event_id"]),
                "event_kind": str(event["event_kind"]),
                "onset_step": onset,
                "onset_hext": float(event["onset_hext"]),
                "lead_time_steps": int(run),
                "detected": bool(run > 0),
                "baseline_center": center,
                "baseline_scale": scale,
            })
        return pd.DataFrame(rows)

    def _error_metrics(self, predictor: str, oriented_residual: np.ndarray) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for target in self.error_targets:
            error = self.df[target].to_numpy(dtype=float)
            best_lag, best_r = 0, _safe_pearson(oriented_residual, error)
            for lag in range(1, min(self.max_lag, len(error) - 3) + 1):
                r = _safe_pearson(oriented_residual[:-lag], error[lag:])
                if abs(r) > abs(best_r):
                    best_lag, best_r = lag, r
            if best_lag:
                x, y = oriented_residual[:-best_lag], error[best_lag:]
            else:
                x, y = oriented_residual, error
            spearman = _safe_spearman(x, y)
            mi = np.nan
            if mutual_info_regression is not None:
                mask = np.isfinite(x) & np.isfinite(y)
                if mask.sum() >= 5 and np.std(x[mask]) > _EPS:
                    mi = float(mutual_info_regression(x[mask, None], y[mask], random_state=self.random_seed)[0])
            p = np.nan
            if self.n_permutations > 0:
                exceed = 0
                for _ in range(self.n_permutations):
                    shift = int(self.rng.integers(1, max(len(oriented_residual), 2)))
                    shifted = np.roll(oriented_residual, shift)
                    if best_lag:
                        r_perm = _safe_pearson(shifted[:-best_lag], error[best_lag:])
                    else:
                        r_perm = _safe_pearson(shifted, error)
                    if abs(r_perm) >= abs(best_r) - 1e-15:
                        exceed += 1
                p = (exceed + 1.0) / (self.n_permutations + 1.0)
            rows.append({
                "predictor": predictor,
                "error_target": target,
                "pearson_zero_lag": _safe_pearson(oriented_residual, error),
                "spearman_zero_lag": _safe_spearman(oriented_residual, error),
                "mutual_information_zero_lag": mi,
                "best_leading_lag_steps": int(best_lag),
                "best_leading_lag_r": float(best_r),
                "best_leading_lag_p": p,
            })
        return rows

    def analyze(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        hext = self.df["hext_scalar"].to_numpy(dtype=float)
        quiet_mask = self._quiet_mask()
        candidates = self._candidate_columns()
        window_rows: List[Dict[str, Any]] = []
        lead_frames: List[pd.DataFrame] = []
        error_rows: List[Dict[str, Any]] = []
        event_kind_rows: List[Dict[str, Any]] = []
        predictor_cache: Dict[str, Dict[str, Any]] = {}

        for predictor in candidates:
            raw = self.df[predictor].to_numpy(dtype=float)
            residual, _ = _detrend_against_field(raw, hext, quiet_mask, degree=2)
            primary_labels = self._labels(self.primary_window)
            quiet_primary = quiet_mask & ~primary_labels
            pre_median = _safe_median(residual[primary_labels])
            quiet_median = _safe_median(residual[quiet_primary])
            direction = 1.0 if not np.isfinite(pre_median - quiet_median) or pre_median >= quiet_median else -1.0
            oriented = direction * residual
            predictor_cache[predictor] = {"raw": raw, "residual": residual, "oriented": oriented, "direction": direction}

            for window in self.windows:
                labels = self._labels(window)
                valid = labels | quiet_mask
                y = labels[valid]
                score = oriented[valid]
                raw_score = direction * raw[valid]
                auc = _auc(y, score)
                raw_auc = _auc(y, raw_score)
                pre = score[y]
                quiet = score[~y]
                effect = _cohens_d(pre, quiet)
                try:
                    mw_p = float(mannwhitneyu(pre, quiet, alternative="two-sided").pvalue) if len(pre) and len(quiet) else np.nan
                except ValueError:
                    mw_p = np.nan
                circular_p = self._permutation_p(y, score, auc)
                ci_low, ci_high = self._bootstrap_auc(y, score)
                window_rows.append({
                    "predictor": predictor,
                    "predictor_group": _predictor_group(predictor),
                    "window_steps": int(window),
                    "direction": int(direction),
                    "auc_oriented": auc,
                    "auc_raw_oriented": raw_auc,
                    "auc_ci_low": ci_low,
                    "auc_ci_high": ci_high,
                    "cohens_d_oriented": effect,
                    "mannwhitney_p": mw_p,
                    "circular_p": circular_p,
                    "n_pretransition": int(y.sum()),
                    "n_quiet": int((~y).sum()),
                })

            lead_frames.append(self._event_lead_times(oriented, predictor))
            error_rows.extend(self._error_metrics(predictor, oriented))

            for event_kind, group in self.events_df.groupby("event_kind") if not self.events_df.empty else []:
                event_ids = set(group["event_id"].astype(int))
                labels = np.zeros(len(self.df), dtype=bool)
                for _, event in group.iterrows():
                    onset = int(event["onset_step"])
                    labels[max(0, onset - self.primary_window):onset] = True
                valid = labels | quiet_mask
                if np.unique(labels[valid]).size < 2:
                    continue
                event_kind_rows.append({
                    "predictor": predictor,
                    "event_kind": event_kind,
                    "n_events": len(event_ids),
                    "auc_oriented": _auc(labels[valid], oriented[valid]),
                    "direction": int(direction),
                })

        window_df = pd.DataFrame(window_rows)
        lead_df = pd.concat([frame for frame in lead_frames if not frame.empty], ignore_index=True) if lead_frames else pd.DataFrame()
        error_df = pd.DataFrame(error_rows)
        kind_df = pd.DataFrame(event_kind_rows)

        if not window_df.empty:
            primary_mask = window_df["window_steps"] == self.primary_window
            window_df.loc[primary_mask, "circular_q"] = _bh_fdr(window_df.loc[primary_mask, "circular_p"])
        if not error_df.empty:
            error_df["best_leading_lag_q"] = _bh_fdr(error_df["best_leading_lag_p"])

        ranking_rows: List[Dict[str, Any]] = []
        for predictor in candidates:
            pwin = window_df[(window_df["predictor"] == predictor) & (window_df["window_steps"] == self.primary_window)]
            if pwin.empty:
                continue
            pwin_row = pwin.iloc[0]
            all_windows = window_df[window_df["predictor"] == predictor]
            leads = lead_df[lead_df["predictor"] == predictor] if not lead_df.empty else pd.DataFrame()
            errors = error_df[error_df["predictor"] == predictor] if not error_df.empty else pd.DataFrame()
            if not errors.empty:
                best_error_idx = errors["best_leading_lag_r"].abs().idxmax()
                best_error = errors.loc[best_error_idx]
            else:
                best_error = pd.Series(dtype=float)

            detection_rate = float(leads["detected"].mean()) if not leads.empty else 0.0
            detected_leads = leads.loc[leads["detected"], "lead_time_steps"] if not leads.empty else pd.Series(dtype=float)
            mean_lead = float(detected_leads.mean()) if len(detected_leads) else 0.0
            max_lead = float(detected_leads.max()) if len(detected_leads) else 0.0
            auc_values = all_windows["auc_oriented"].to_numpy(dtype=float)
            direction_consistency = float(np.mean(all_windows["direction"] == all_windows["direction"].iloc[0]))
            mean_auc = _safe_mean(auc_values)
            min_auc = float(np.nanmin(auc_values)) if len(auc_values) else 0.5
            transition_score = np.clip((float(pwin_row["auc_oriented"]) - 0.5) / 0.5, 0.0, 1.0)
            coverage_score = np.clip(detection_rate, 0.0, 1.0)
            lead_score = np.clip(mean_lead / max(max(self.windows), 1), 0.0, 1.0)
            error_score = np.clip(abs(float(best_error.get("best_leading_lag_r", 0.0))), 0.0, 1.0)
            robustness_score = np.clip(0.5 * direction_consistency + 0.5 * max((mean_auc - 0.5) / 0.5, 0.0), 0.0, 1.0)
            importance = 0.30 * transition_score + 0.20 * coverage_score + 0.20 * lead_score + 0.20 * error_score + 0.10 * robustness_score
            independent = predictor not in self.CIRCULAR_COLUMNS
            ranking_rows.append({
                "predictor": predictor,
                "predictor_label": _friendly_name(predictor),
                "predictor_group": _predictor_group(predictor),
                "independent_predictor": independent,
                "exclusion_reason": "" if independent else "Used directly or nearly directly in transition definition",
                "physics_importance_score": float(importance),
                "primary_window_steps": self.primary_window,
                "primary_direction": int(pwin_row["direction"]),
                "primary_auc_oriented": float(pwin_row["auc_oriented"]),
                "primary_auc_raw_oriented": float(pwin_row["auc_raw_oriented"]),
                "primary_auc_ci_low": float(pwin_row["auc_ci_low"]),
                "primary_auc_ci_high": float(pwin_row["auc_ci_high"]),
                "primary_circular_p": float(pwin_row["circular_p"]),
                "primary_circular_q": float(pwin_row.get("circular_q", np.nan)),
                "primary_fdr_significant_0_05": bool(pwin_row.get("circular_q", 1.0) <= 0.05),
                "event_detection_rate": detection_rate,
                "mean_lead_time_steps": mean_lead,
                "max_lead_time_steps": max_lead,
                "best_error_target": best_error.get("error_target", ""),
                "best_error_leading_lag_r": float(best_error.get("best_leading_lag_r", 0.0)),
                "best_error_leading_lag_steps": int(best_error.get("best_leading_lag_steps", 0)),
                "best_error_lag_p": float(best_error.get("best_leading_lag_p", np.nan)),
                "best_error_lag_q": float(best_error.get("best_leading_lag_q", np.nan)),
                "best_error_fdr_significant_0_05": bool(best_error.get("best_leading_lag_q", 1.0) <= 0.05),
                "mean_auc_across_windows": mean_auc,
                "minimum_auc_across_windows": min_auc,
                "direction_consistency": direction_consistency,
                "robustness_score": robustness_score,
            })

        ranking_all = pd.DataFrame(ranking_rows)
        if not ranking_all.empty:
            ranking_all = ranking_all.sort_values("physics_importance_score", ascending=False).reset_index(drop=True)
            ranking_all["overall_rank"] = np.arange(1, len(ranking_all) + 1)
        primary = ranking_all[ranking_all.get("independent_predictor", False)].copy() if not ranking_all.empty else pd.DataFrame()
        if not primary.empty:
            primary = primary.sort_values("physics_importance_score", ascending=False).reset_index(drop=True)
            primary.insert(0, "primary_rank", np.arange(1, len(primary) + 1))
        return primary, ranking_all, window_df, error_df, lead_df, kind_df

    def run(self, save_path: os.PathLike[str] | str, general_title: str = "",
            top_n: int = 12) -> pd.DataFrame:
        folder = Path(save_path)
        folder.mkdir(parents=True, exist_ok=True)
        primary, all_ranking, window_df, error_df, lead_df, kind_df = self.analyze()
        primary.to_csv(folder / "leading_indicator_ranking_primary.csv", index=False)
        all_ranking.to_csv(folder / "leading_indicator_ranking_all.csv", index=False)
        window_df.to_csv(folder / "leading_indicator_window_metrics.csv", index=False)
        error_df.to_csv(folder / "leading_indicator_error_metrics.csv", index=False)
        lead_df.to_csv(folder / "leading_indicator_event_lead_times.csv", index=False)
        kind_df.to_csv(folder / "leading_indicator_event_kind_metrics.csv", index=False)
        config = {
            "error_targets": self.error_targets,
            "pretransition_windows": self.windows,
            "primary_window": self.primary_window,
            "max_lag": self.max_lag,
            "lead_z_threshold": self.lead_z_threshold,
            "post_event_exclusion": self.post_event_exclusion,
            "n_permutations": self.n_permutations,
            "n_bootstrap": self.n_bootstrap,
            "random_seed": self.random_seed,
            "circular_columns_excluded_from_primary": sorted(self.CIRCULAR_COLUMNS),
        }
        (folder / "leading_indicator_analysis_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

        if not primary.empty:
            top = primary.head(top_n).iloc[::-1]
            fig, ax = plt.subplots(figsize=(11, max(5, 0.45 * len(top) + 1.5)))
            ax.barh(top["predictor_label"], top["physics_importance_score"])
            ax.set_xlabel("Physics Importance Score")
            ax.set_title((general_title + "\n" if general_title else "") + "Independent FFT leading indicators")
            ax.grid(axis="x", alpha=0.25)
            fig.tight_layout()
            fig.savefig(folder / "physics_importance_ranking.png", dpi=250, bbox_inches="tight")
            plt.close(fig)

            metrics = primary.head(top_n).set_index("predictor_label")[[
                "primary_auc_oriented", "event_detection_rate", "robustness_score",
                "best_error_leading_lag_r", "physics_importance_score",
            ]].copy()
            metrics["best_error_leading_lag_r"] = metrics["best_error_leading_lag_r"].abs()
            fig, ax = plt.subplots(figsize=(10, max(5, 0.42 * len(metrics) + 2)))
            image = ax.imshow(metrics.to_numpy(dtype=float), aspect="auto", vmin=0, vmax=1)
            ax.set_yticks(np.arange(len(metrics)), labels=metrics.index)
            ax.set_xticks(np.arange(len(metrics.columns)), labels=[c.replace("_", " ") for c in metrics.columns], rotation=35, ha="right")
            fig.colorbar(image, ax=ax, label="Metric value")
            ax.set_title("Leading-indicator metric summary")
            fig.tight_layout()
            fig.savefig(folder / "leading_indicator_metric_heatmap.png", dpi=250, bbox_inches="tight")
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(14, 7))
            hext = self.df["hext_scalar"].to_numpy(dtype=float)
            quiet = self._quiet_mask()
            for _, row in primary.head(min(top_n, 6)).iterrows():
                name = row["predictor"]
                residual, _ = _detrend_against_field(self.df[name], hext, quiet)
                oriented_z = _robust_z(float(row["primary_direction"]) * residual)
                ax.plot(hext, oriented_z, label=row["predictor_label"])
            error = self.df[self.error_targets[0]].to_numpy(dtype=float) if self.error_targets else np.zeros(len(self.df))
            ax.plot(hext, _robust_z(np.log10(error + _EPS)), color="black", linewidth=2.5, label=f"log10 {self.error_targets[0] if self.error_targets else 'error'}")
            for _, event in self.events_df.iterrows():
                ax.axvline(event["onset_hext"], linestyle="--", alpha=0.35)
            ax.set_xlabel("External field (Oe)")
            ax.set_ylabel("Oriented robust z-score")
            ax.set_title("Top FFT predictors and UNet error")
            ax.legend(fontsize=8, ncol=2)
            ax.grid(alpha=0.25)
            fig.tight_layout()
            fig.savefig(folder / "top_leading_indicator_trajectories.png", dpi=250, bbox_inches="tight")
            plt.close(fig)
        return primary


# =============================================================================
# Part 4: publication figures
# =============================================================================


class PublicationFigureGenerator:
    def __init__(
        self,
        labeled_df: pd.DataFrame,
        events_df: pd.DataFrame,
        indicator_ranking: pd.DataFrame,
        *,
        primary_window: int = 10,
        top_n: int = 4,
        pre_steps: int = 20,
        post_steps: int = 10,
        dpi: int = 300,
        formats: Sequence[str] = ("png", "pdf"),
        lead_z_threshold: float = 1.5,
    ) -> None:
        self.df = labeled_df.sort_values("mh_step").reset_index(drop=True).copy()
        self.events = events_df.copy()
        self.ranking = indicator_ranking.copy()
        self.primary_window = int(primary_window)
        self.top_n = int(max(1, top_n))
        self.pre_steps = int(max(1, pre_steps))
        self.post_steps = int(max(0, post_steps))
        self.dpi = int(max(72, dpi))
        self.formats = tuple(formats)
        self.lead_z_threshold = float(lead_z_threshold)

    def _top_rows(self, n: Optional[int] = None) -> pd.DataFrame:
        return self.ranking.head(n or self.top_n).copy()

    def _oriented_z(self, row: pd.Series) -> np.ndarray:
        name = str(row["predictor"])
        quiet_column = f"is_pretransition_{max(self.primary_window, 1)}"
        quiet = ~self.df.get("is_transition_step", pd.Series(False, index=self.df.index)).to_numpy(dtype=bool)
        if quiet_column in self.df:
            quiet &= ~self.df[quiet_column].to_numpy(dtype=bool)
        residual, _ = _detrend_against_field(self.df[name], self.df["hext_scalar"], quiet)
        return _robust_z(float(row.get("primary_direction", 1.0)) * residual)

    def _event_centered_long(self, columns: Mapping[str, np.ndarray]) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        for _, event in self.events.iterrows():
            onset = int(event["onset_step"])
            for relative in range(-self.pre_steps, self.post_steps + 1):
                index = onset + relative
                if index < 0 or index >= len(self.df):
                    continue
                for name, values in columns.items():
                    rows.append({
                        "event_id": int(event["event_id"]),
                        "event_kind": str(event["event_kind"]),
                        "relative_step": relative,
                        "quantity": name,
                        "value": float(values[index]),
                    })
        return pd.DataFrame(rows)

    def run(self, save_path: os.PathLike[str] | str, general_title: str = "") -> pd.DataFrame:
        folder = Path(save_path)
        folder.mkdir(parents=True, exist_ok=True)
        manifest: List[Dict[str, Any]] = []
        hext = self.df["hext_scalar"].to_numpy(dtype=float)
        top = self._top_rows()

        # Figure 1: complete sweep overview.
        fig, axes = plt.subplots(4, 1, figsize=(15, 15), sharex=True)
        axes[0].plot(hext, self.df["fft_m_projection"], label="FFT")
        axes[0].plot(hext, self.df["unet_m_projection"], label="UNet", alpha=0.8)
        axes[0].set_ylabel("Projected M")
        axes[0].legend()
        for column, label in (("fft_positive_core_count", "Positive cores"),
                              ("fft_negative_core_count", "Negative cores"),
                              ("fft_total_core_count", "Total cores"),
                              ("fft_winding_abs", "Absolute winding")):
            if column in self.df:
                axes[1].plot(hext, self.df[column], label=label)
        axes[1].set_ylabel("FFT topology")
        axes[1].legend(ncol=2, fontsize=8)
        axes[2].plot(hext, self.df["spin_mae"] + _EPS, label="Spin MAE")
        axes[2].plot(hext, self.df["hd_mae"] + _EPS, label="Hdemag MAE")
        axes[2].set_yscale("log")
        axes[2].set_ylabel("UNet error")
        axes[2].legend()
        fig1_data = self.df[[column for column in ["mh_step", "hext_scalar", "fft_m_projection", "unet_m_projection", "fft_positive_core_count", "fft_negative_core_count", "fft_total_core_count", "fft_winding_abs", "spin_mae", "hd_mae"] if column in self.df]].copy()
        for _, row in top.iterrows():
            values = self._oriented_z(row)
            axes[3].plot(hext, values, label=row.get("predictor_label", _friendly_name(row["predictor"])))
            fig1_data[f"oriented_z__{row['predictor']}"] = values
        axes[3].set_ylabel("Oriented predictor z")
        axes[3].set_xlabel("External field (Oe)")
        axes[3].legend(fontsize=8, ncol=2)
        for _, event in self.events.iterrows():
            for axis in axes:
                axis.axvline(event["onset_hext"], linestyle="--", alpha=0.35)
        for axis in axes:
            axis.grid(alpha=0.22)
        fig.suptitle((general_title + "\n" if general_title else "") + "M-H sweep, topology, error, and leading indicators")
        fig.tight_layout()
        paths = _save_figure(fig, folder / "figure1_sweep_overview", self.formats, self.dpi)
        fig1_data.to_csv(folder / "figure1_sweep_overview_data.csv", index=False)
        manifest.append({"figure": "figure1_sweep_overview", "paths": ";".join(paths), "data": str(folder / "figure1_sweep_overview_data.csv")})

        # Figure 2: event-centered profiles.
        centered_columns: Dict[str, np.ndarray] = {}
        for _, row in top.iterrows():
            centered_columns[str(row["predictor"])] = self._oriented_z(row)
        quiet = ~self.df.get("is_transition_step", pd.Series(False, index=self.df.index)).to_numpy(dtype=bool)
        for target in ("spin_mae", "hd_mae"):
            baseline = _safe_median(self.df.loc[quiet, target])
            centered_columns[f"log10_fold_{target}"] = np.log10((self.df[target].to_numpy(dtype=float) + _EPS) / (baseline + _EPS))
        centered_columns["delta_total_cores"] = self.df.get("fft_total_core_count", pd.Series(0, index=self.df.index)).to_numpy(dtype=float)
        centered_columns["delta_winding_abs"] = self.df["fft_winding_abs"].to_numpy(dtype=float)
        centered_columns["delta_m_projection"] = self.df["fft_m_projection"].to_numpy(dtype=float)
        centered = self._event_centered_long(centered_columns)
        centered.to_csv(folder / "figure2_event_centered_profiles_data.csv", index=False)
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
        if not centered.empty and not top.empty:
            first_name = str(top.iloc[0]["predictor"])
            first = centered[centered["quantity"] == first_name]
            for event_id, group in first.groupby("event_id"):
                axes[0, 0].plot(group["relative_step"], group["value"], alpha=0.3)
            mean = first.groupby("relative_step")["value"].mean()
            axes[0, 0].plot(mean.index, mean.values, linewidth=3, label=_friendly_name(first_name))
            axes[0, 0].legend()
            for _, row in top.iterrows():
                name = str(row["predictor"])
                group = centered[centered["quantity"] == name].groupby("relative_step")["value"].mean()
                axes[0, 1].plot(group.index, group.values, label=row.get("predictor_label", _friendly_name(name)))
            axes[0, 1].legend(fontsize=8)
            for target in ("log10_fold_spin_mae", "log10_fold_hd_mae"):
                group = centered[centered["quantity"] == target].groupby("relative_step")["value"].mean()
                axes[1, 0].plot(group.index, group.values, label=target.replace("log10_fold_", ""))
            axes[1, 0].legend()
            for target in ("delta_total_cores", "delta_winding_abs", "delta_m_projection"):
                subset = centered[centered["quantity"] == target].copy()
                if subset.empty:
                    continue
                subset["value"] = subset.groupby("event_id")["value"].transform(lambda x: x - x.iloc[0])
                group = subset.groupby("relative_step")["value"].mean()
                axes[1, 1].plot(group.index, group.values, label=target.replace("delta_", ""))
            axes[1, 1].legend(fontsize=8)
        titles = ["Highest-ranked predictor", "Top predictor means", "UNet error relative to quiet", "FFT transition observables"]
        for axis, title in zip(axes.flat, titles):
            axis.axvline(0, linestyle="--", color="black", alpha=0.5)
            axis.set_title(title)
            axis.grid(alpha=0.22)
            axis.set_xlabel("M-H steps relative to FFT event onset")
        fig.tight_layout()
        paths = _save_figure(fig, folder / "figure2_event_centered_response", self.formats, self.dpi)
        manifest.append({"figure": "figure2_event_centered_response", "paths": ";".join(paths), "data": str(folder / "figure2_event_centered_profiles_data.csv")})

        # Figure 3: nucleation/creation versus annihilation/merging.
        event_type_rows: List[pd.DataFrame] = []
        categories = {
            "Nucleation / creation": self.events[self.events["event_kind"].str.contains("nucleation|creation|splitting", case=False, regex=True)] if not self.events.empty else pd.DataFrame(),
            "Annihilation / merging": self.events[self.events["event_kind"].str.contains("annihilation|merging", case=False, regex=True)] if not self.events.empty else pd.DataFrame(),
        }
        fig, axes = plt.subplots(max(1, len(top)), 2, figsize=(13, max(5, 3 * len(top))), squeeze=False, sharex=True)
        for row_index, (_, predictor_row) in enumerate(top.iterrows()):
            name = str(predictor_row["predictor"])
            values = self._oriented_z(predictor_row)
            for col_index, (category, category_events) in enumerate(categories.items()):
                axis = axes[row_index, col_index]
                profiles: List[np.ndarray] = []
                relative = np.arange(-self.pre_steps, self.post_steps + 1)
                for _, event in category_events.iterrows():
                    onset = int(event["onset_step"])
                    profile = np.full(len(relative), np.nan)
                    for j, rel in enumerate(relative):
                        index = onset + rel
                        if 0 <= index < len(values):
                            profile[j] = values[index]
                    profiles.append(profile)
                    for j, rel in enumerate(relative):
                        if np.isfinite(profile[j]):
                            event_type_rows.append(pd.DataFrame([{"predictor": name, "category": category, "event_id": int(event["event_id"]), "relative_step": rel, "value": profile[j]}]))
                if profiles:
                    matrix = np.vstack(profiles)
                    mean = np.nanmean(matrix, axis=0)
                    axis.plot(relative, mean)
                    if len(profiles) > 1:
                        sem = np.nanstd(matrix, axis=0, ddof=1) / math.sqrt(len(profiles))
                        axis.fill_between(relative, mean - sem, mean + sem, alpha=0.2)
                axis.axvline(0, linestyle="--", color="black", alpha=0.45)
                axis.grid(alpha=0.2)
                if row_index == 0:
                    axis.set_title(category)
                if col_index == 0:
                    axis.set_ylabel(predictor_row.get("predictor_label", _friendly_name(name)))
                if row_index == len(top) - 1:
                    axis.set_xlabel("Relative M-H step")
        fig.tight_layout()
        paths = _save_figure(fig, folder / "figure3_event_type_comparison", self.formats, self.dpi)
        event_type_data = pd.concat(event_type_rows, ignore_index=True) if event_type_rows else pd.DataFrame()
        event_type_data.to_csv(folder / "figure3_event_type_comparison_data.csv", index=False)
        manifest.append({"figure": "figure3_event_type_comparison", "paths": ";".join(paths), "data": str(folder / "figure3_event_type_comparison_data.csv")})

        # Figure 4: current predictor against future error.
        scatter_rows: List[Dict[str, Any]] = []
        top4 = top.head(4)
        fig, axes = plt.subplots(2, 2, figsize=(12, 10), squeeze=False)
        for axis, (_, row) in zip(axes.flat, top4.iterrows()):
            name = str(row["predictor"])
            target = str(row.get("best_error_target", "spin_mae"))
            if target not in self.df:
                target = "spin_mae"
            lag = int(row.get("best_error_leading_lag_steps", 0))
            predictor = self._oriented_z(row)
            error = self.df[target].to_numpy(dtype=float)
            quiet_median = _safe_median(error[~self.df.get("is_transition_step", pd.Series(False, index=self.df.index)).to_numpy(dtype=bool)])
            future_fold = np.log10((error + _EPS) / (quiet_median + _EPS))
            upper = len(self.df) - lag if lag > 0 else len(self.df)
            for i in range(upper):
                target_i = i + lag
                if bool(self.df.loc[i, "is_transition_step"]):
                    category = "Transition"
                elif bool(self.df.loc[i, f"is_pretransition_{self.primary_window}"]) if f"is_pretransition_{self.primary_window}" in self.df else False:
                    category = "Pre-transition"
                else:
                    category = "Quiet"
                scatter_rows.append({"predictor": name, "error_target": target, "leading_lag_steps": lag, "source_mh_step": i, "target_mh_step": target_i, "source_hext": hext[i], "predictor_oriented_z": predictor[i], "future_error_log10_fold": future_fold[target_i], "category": category})
            subset = pd.DataFrame([r for r in scatter_rows if r["predictor"] == name])
            for category, group in subset.groupby("category"):
                axis.scatter(group["predictor_oriented_z"], group["future_error_log10_fold"], s=20, alpha=0.65, label=category)
            if len(subset) >= 3 and np.std(subset["predictor_oriented_z"]) > _EPS:
                coeff = np.polyfit(subset["predictor_oriented_z"], subset["future_error_log10_fold"], 1)
                grid = np.linspace(subset["predictor_oriented_z"].min(), subset["predictor_oriented_z"].max(), 100)
                axis.plot(grid, np.polyval(coeff, grid), color="black", linewidth=1.5)
            axis.set_title(f"{row.get('predictor_label', _friendly_name(name))}\n{target} at +{lag} steps")
            axis.set_xlabel("FFT predictor oriented z-score")
            axis.set_ylabel("Future error log10 fold")
            axis.grid(alpha=0.2)
            axis.legend(fontsize=8)
        for axis in axes.flat[len(top4):]:
            axis.axis("off")
        fig.tight_layout()
        paths = _save_figure(fig, folder / "figure4_predictor_future_error", self.formats, self.dpi)
        pd.DataFrame(scatter_rows).to_csv(folder / "figure4_future_error_relationships_data.csv", index=False)
        manifest.append({"figure": "figure4_predictor_future_error", "paths": ";".join(paths), "data": str(folder / "figure4_future_error_relationships_data.csv")})

        # Figure 5: event-specific contiguous lead-time heatmap.
        lead_rows: List[Dict[str, Any]] = []
        max_window = max(self.pre_steps, self.primary_window)
        for _, row in top.iterrows():
            name = str(row["predictor"])
            values = self._oriented_z(row)
            for _, event in self.events.iterrows():
                onset = int(event["onset_step"])
                baseline = values[max(0, onset - 2 * max_window):max(1, onset - max_window)]
                center, scale = _robust_scale(baseline)
                z = (values[max(0, onset - max_window):onset] - center) / scale
                run = 0
                for value in z[::-1]:
                    if value >= self.lead_z_threshold:
                        run += 1
                    else:
                        break
                lead_rows.append({"predictor": name, "predictor_label": row.get("predictor_label", _friendly_name(name)), "event_id": int(event["event_id"]), "event_kind": str(event["event_kind"]), "lead_time_steps": run})
        lead_data = pd.DataFrame(lead_rows)
        lead_data.to_csv(folder / "figure5_event_lead_time_matrix_data.csv", index=False)
        fig, ax = plt.subplots(figsize=(max(7, 1.3 * max(len(self.events), 1)), max(5, 0.55 * max(len(top), 1) + 2)))
        if not lead_data.empty:
            matrix = lead_data.pivot(index="predictor_label", columns="event_id", values="lead_time_steps")
            image = ax.imshow(matrix.to_numpy(dtype=float), aspect="auto", vmin=0, vmax=max(max_window, 1))
            ax.set_yticks(np.arange(len(matrix)), labels=matrix.index)
            event_labels = []
            for event_id in matrix.columns:
                kind = self.events.loc[self.events["event_id"] == event_id, "event_kind"].iloc[0]
                event_labels.append(f"E{event_id}\n{kind}")
            ax.set_xticks(np.arange(len(matrix.columns)), labels=event_labels, rotation=35, ha="right")
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    value = matrix.iloc[i, j]
                    ax.text(j, i, f"{int(value)}", ha="center", va="center")
            fig.colorbar(image, ax=ax, label="Lead time (M-H steps)")
        ax.set_title("Per-event contiguous advance warning")
        fig.tight_layout()
        paths = _save_figure(fig, folder / "figure5_event_lead_time_heatmap", self.formats, self.dpi)
        manifest.append({"figure": "figure5_event_lead_time_heatmap", "paths": ";".join(paths), "data": str(folder / "figure5_event_lead_time_matrix_data.csv")})

        config = {
            "primary_window": self.primary_window,
            "top_n": self.top_n,
            "pre_steps": self.pre_steps,
            "post_steps": self.post_steps,
            "dpi": self.dpi,
            "formats": self.formats,
            "lead_z_threshold": self.lead_z_threshold,
            "predictors": top.get("predictor", pd.Series(dtype=str)).tolist(),
            "n_events": len(self.events),
        }
        (folder / "publication_figure_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        manifest_df = pd.DataFrame(manifest)
        manifest_df.to_csv(folder / "publication_figure_manifest.csv", index=False)
        return manifest_df


# =============================================================================
# Part 5: manuscript tables, reproducibility, and cross-sweep aggregation
# =============================================================================


class ManuscriptOutputGenerator:
    def __init__(
        self,
        physics_df: pd.DataFrame,
        labeled_df: pd.DataFrame,
        events_df: pd.DataFrame,
        indicator_ranking: pd.DataFrame,
        *,
        part3_directory: Optional[os.PathLike[str] | str] = None,
        run_metadata: Optional[Mapping[str, Any]] = None,
        primary_window: int = 10,
        top_n: int = 10,
        formats: Sequence[str] = ("csv", "tex", "md"),
    ) -> None:
        self.physics = physics_df.copy()
        self.labeled = labeled_df.copy()
        self.events = events_df.copy()
        self.ranking = indicator_ranking.copy()
        self.part3_directory = Path(part3_directory) if part3_directory else None
        self.metadata = dict(run_metadata or {})
        self.primary_window = int(primary_window)
        self.top_n = int(max(1, top_n))
        self.formats = tuple(formats)

    def _read_part3(self, filename: str) -> pd.DataFrame:
        if self.part3_directory is None:
            return pd.DataFrame()
        path = self.part3_directory / filename
        return pd.read_csv(path) if path.is_file() else pd.DataFrame()

    def _table1(self) -> pd.DataFrame:
        rows = [{"Parameter": key, "Value": value} for key, value in sorted(self.metadata.items())]
        rows.extend([
            {"Parameter": "n_mh_steps", "Value": len(self.physics)},
            {"Parameter": "hext_start_oe", "Value": self.physics["hext_scalar"].iloc[0] if len(self.physics) else np.nan},
            {"Parameter": "hext_end_oe", "Value": self.physics["hext_scalar"].iloc[-1] if len(self.physics) else np.nan},
            {"Parameter": "hext_spacing_oe", "Value": abs(float(np.median(np.diff(self.physics["hext_scalar"])))) if len(self.physics) > 1 else np.nan},
            {"Parameter": "n_detected_events", "Value": len(self.events)},
        ])
        return pd.DataFrame(rows)

    def _table2(self) -> pd.DataFrame:
        columns = [column for column in TransitionAnalyzer.EVENT_COLUMNS if column in self.events]
        return self.events[columns].copy() if columns else pd.DataFrame(columns=TransitionAnalyzer.EVENT_COLUMNS)

    def _table3(self) -> pd.DataFrame:
        transition = self.labeled.get("is_transition_step", pd.Series(False, index=self.labeled.index)).to_numpy(dtype=bool)
        pre_column = f"is_pretransition_{self.primary_window}"
        pre = self.labeled.get(pre_column, pd.Series(False, index=self.labeled.index)).to_numpy(dtype=bool) & ~transition
        regimes = {
            "Quiet": ~(transition | pre),
            "Pre-transition": pre,
            "Transition": transition,
            "All steps": np.ones(len(self.labeled), dtype=bool),
        }
        targets = [target for target in ("spin_mae", "hd_mae", "m_projection_abs_error") if target in self.labeled]
        rows: List[Dict[str, Any]] = []
        for regime, mask in regimes.items():
            row: Dict[str, Any] = {"Regime": regime, "N states": int(mask.sum())}
            for target in targets:
                values = _finite(self.labeled.loc[mask, target])
                row[f"{target} median"] = float(np.median(values)) if values.size else np.nan
                row[f"{target} IQR"] = float(np.percentile(values, 75) - np.percentile(values, 25)) if values.size else np.nan
                row[f"{target} 95th percentile"] = float(np.percentile(values, 95)) if values.size else np.nan
                row[f"{target} maximum"] = float(np.max(values)) if values.size else np.nan
            rows.append(row)
        return pd.DataFrame(rows)

    def _table4(self) -> pd.DataFrame:
        columns = [
            "primary_rank", "predictor_label", "predictor_group", "physics_importance_score",
            "primary_auc_oriented", "primary_auc_ci_low", "primary_auc_ci_high",
            "primary_circular_q", "event_detection_rate", "mean_lead_time_steps",
            "best_error_target", "best_error_leading_lag_r", "best_error_leading_lag_steps",
            "best_error_lag_q", "robustness_score",
        ]
        existing = [column for column in columns if column in self.ranking]
        return self.ranking.head(self.top_n)[existing].copy()

    def _table5(self) -> pd.DataFrame:
        leads = self._read_part3("leading_indicator_event_lead_times.csv")
        if leads.empty:
            return pd.DataFrame()
        top_names = set(self.ranking.head(self.top_n)["predictor"])
        leads = leads[leads["predictor"].isin(top_names)]
        if leads.empty:
            return pd.DataFrame()
        pivot = leads.pivot(index="predictor", columns="event_id", values="lead_time_steps").reset_index()
        pivot["predictor"] = pivot["predictor"].map(_friendly_name)
        pivot = pivot.rename(columns={column: f"Event {column}" for column in pivot.columns if column != "predictor"})
        return pivot

    def _table6(self) -> pd.DataFrame:
        kinds = self._read_part3("leading_indicator_event_kind_metrics.csv")
        if kinds.empty:
            return pd.DataFrame()
        top_names = set(self.ranking.head(self.top_n)["predictor"])
        kinds = kinds[kinds["predictor"].isin(top_names)].copy()
        kinds["predictor_label"] = kinds["predictor"].map(_friendly_name)
        return kinds[[column for column in ("predictor_label", "event_kind", "n_events", "auc_oriented", "direction") if column in kinds]]

    def _table7(self) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        for solver in ("fft", "unet"):
            runtime = self.physics.get(f"{solver}_runtime_seconds", pd.Series(dtype=float)).to_numpy(dtype=float)
            iterations = self.physics.get(f"{solver}_iterations", pd.Series(dtype=float)).to_numpy(dtype=float)
            convergence = self.physics.get(f"{solver}_final_convergence_error", pd.Series(dtype=float)).to_numpy(dtype=float)
            rows.append({
                "Solver": solver.upper(),
                "Total runtime (s)": float(np.nansum(runtime)),
                "Median runtime/field (s)": _safe_median(runtime),
                "95th percentile runtime (s)": float(np.nanpercentile(runtime, 95)) if len(runtime) else np.nan,
                "Median iterations": _safe_median(iterations),
                "95th percentile iterations": float(np.nanpercentile(iterations, 95)) if len(iterations) else np.nan,
                "Maximum final convergence error": float(np.nanmax(convergence)) if len(convergence) else np.nan,
            })
        table = pd.DataFrame(rows)
        if len(table) == 2 and table.loc[0, "Total runtime (s)"] > 0:
            table["Runtime ratio to FFT"] = table["Total runtime (s)"] / table.loc[0, "Total runtime (s)"]
            if table.loc[0, "Median iterations"] > 0:
                table["Iteration ratio to FFT"] = table["Median iterations"] / table.loc[0, "Median iterations"]
        return table

    def _run_record(self) -> Dict[str, Any]:
        metadata_json = json.dumps(self.metadata, sort_keys=True, default=str)
        run_id = hashlib.sha256(metadata_json.encode("utf-8")).hexdigest()[:16]
        quiet = ~self.labeled.get("is_transition_step", pd.Series(False, index=self.labeled.index)).to_numpy(dtype=bool)
        transition = ~quiet
        top = self.ranking.iloc[0].to_dict() if not self.ranking.empty else {}
        versions = {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
        }
        try:
            import scipy
            versions["scipy"] = scipy.__version__
        except Exception:
            pass
        try:
            import sklearn
            versions["scikit_learn"] = sklearn.__version__
        except Exception:
            pass
        return {
            "run_id": run_id,
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "metadata": self.metadata,
            "command": " ".join(sys.argv),
            "versions": versions,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "checkpoint_sha256": _sha256_file(self.metadata.get("checkpoint_path")),
            "evaluation_script_sha256": _sha256_file(self.metadata.get("evaluation_script_path")),
            "searcher_sha256": _sha256_file(self.metadata.get("searcher_path")),
            "n_steps": len(self.physics),
            "n_events": len(self.events),
            "event_kind_counts": self.events["event_kind"].value_counts().to_dict() if "event_kind" in self.events else {},
            "top_predictor": top,
            "quiet_spin_mae_median": _safe_median(self.labeled.loc[quiet, "spin_mae"]) if "spin_mae" in self.labeled else np.nan,
            "transition_spin_mae_median": _safe_median(self.labeled.loc[transition, "spin_mae"]) if "spin_mae" in self.labeled and transition.any() else np.nan,
            "quiet_hd_mae_median": _safe_median(self.labeled.loc[quiet, "hd_mae"]) if "hd_mae" in self.labeled else np.nan,
            "transition_hd_mae_median": _safe_median(self.labeled.loc[transition, "hd_mae"]) if "hd_mae" in self.labeled and transition.any() else np.nan,
            "fft_total_runtime_seconds": float(self.physics.get("fft_runtime_seconds", pd.Series(dtype=float)).sum()),
            "unet_total_runtime_seconds": float(self.physics.get("unet_runtime_seconds", pd.Series(dtype=float)).sum()),
        }

    def run(self, save_path: os.PathLike[str] | str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        folder = Path(save_path)
        folder.mkdir(parents=True, exist_ok=True)
        tables = {
            "table1_run_configuration": self._table1(),
            "table2_transition_event_catalog": self._table2(),
            "table3_unet_accuracy_by_regime": self._table3(),
            "table4_leading_indicator_ranking": self._table4(),
            "table5_event_specific_lead_times": self._table5(),
            "table6_event_kind_performance": self._table6(),
            "table7_computational_performance": self._table7(),
        }
        manifest_rows: List[Dict[str, Any]] = []
        for name, table in tables.items():
            paths = _export_dataframe(table, folder / name, self.formats)
            manifest_rows.append({"artifact": name, "paths": ";".join(paths), "rows": len(table)})

        top = self.ranking.iloc[0] if not self.ranking.empty else None
        transition_mask = self.labeled.get("is_transition_step", pd.Series(False, index=self.labeled.index)).to_numpy(dtype=bool)
        quiet_mask = ~transition_mask
        lines = ["# Automated results summary", ""]
        lines.append(f"The sweep contained **{len(self.physics)}** converged M-H states and **{len(self.events)}** FFT-defined transition events.")
        if not self.events.empty:
            counts = ", ".join(f"{kind}: {count}" for kind, count in self.events["event_kind"].value_counts().items())
            lines.append(f"Detected event classes: {counts}.")
        if top is not None:
            lines.append(
                f"The highest-ranked independent FFT predictor was **{top.get('predictor_label', _friendly_name(top['predictor']))}** "
                f"with Physics Importance Score {top.get('physics_importance_score', np.nan):.3f}, "
                f"pre-transition AUC {top.get('primary_auc_oriented', np.nan):.3f}, and mean detected lead time "
                f"{top.get('mean_lead_time_steps', np.nan):.2f} M-H steps."
            )
        if transition_mask.any() and "spin_mae" in self.labeled:
            ratio = (_safe_median(self.labeled.loc[transition_mask, "spin_mae"]) + _EPS) / (_safe_median(self.labeled.loc[quiet_mask, "spin_mae"]) + _EPS)
            lines.append(f"Median spin MAE during transition rows was {ratio:.2f} times the quiet-state median.")
        lines.extend(["", "These statements describe this completed sweep only. General claims require repeated seeds, geometries, and material settings."])
        (folder / "automated_results_summary.md").write_text("\n\n".join(lines) + "\n", encoding="utf-8")

        quality = {
            "exploratory_event_count": len(self.events) < 5,
            "single_event_type_only": self.events["event_kind"].nunique() < 2 if "event_kind" in self.events else True,
            "no_transition_fdr_significant_predictor": not bool(self.ranking.get("primary_fdr_significant_0_05", pd.Series(False)).any()),
            "no_future_error_fdr_significant_predictor": not bool(self.ranking.get("best_error_fdr_significant_0_05", pd.Series(False)).any()),
        }
        (folder / "analysis_quality_flags.json").write_text(json.dumps(quality, indent=2), encoding="utf-8")

        run_record = self._run_record()
        (folder / "run_record.json").write_text(json.dumps(run_record, indent=2, default=str), encoding="utf-8")
        run_id = run_record["run_id"]

        run_row = {"run_id": run_id, **{f"meta_{key}": value for key, value in self.metadata.items()},
                   "n_steps": len(self.physics), "n_events": len(self.events),
                   "fft_total_runtime_seconds": run_record["fft_total_runtime_seconds"],
                   "unet_total_runtime_seconds": run_record["unet_total_runtime_seconds"]}
        pd.DataFrame([run_row]).to_csv(folder / "cross_sweep_run_record.csv", index=False)
        predictor_records = self.ranking.copy()
        if not predictor_records.empty:
            predictor_records.insert(0, "run_id", run_id)
        predictor_records.to_csv(folder / "cross_sweep_predictor_records.csv", index=False)
        event_records = self.events.copy()
        if not event_records.empty:
            event_records.insert(0, "run_id", run_id)
        event_records.to_csv(folder / "cross_sweep_event_records.csv", index=False)
        regime_records = tables["table3_unet_accuracy_by_regime"].copy()
        if not regime_records.empty:
            regime_records.insert(0, "run_id", run_id)
        regime_records.to_csv(folder / "cross_sweep_accuracy_regime_records.csv", index=False)

        manifest = pd.DataFrame(manifest_rows)
        manifest.to_csv(folder / "manuscript_output_manifest.csv", index=False)
        package = {
            "run_id": run_id,
            "tables": manifest_rows,
            "summary": str(folder / "automated_results_summary.md"),
            "quality_flags": str(folder / "analysis_quality_flags.json"),
            "run_record": str(folder / "run_record.json"),
        }
        (folder / "manuscript_output_package.json").write_text(json.dumps(package, indent=2), encoding="utf-8")
        return manifest, run_record


class CrossSweepAggregator:
    """Combine Part 5 records from multiple completed run directories."""

    def __init__(self, root: os.PathLike[str] | str, *, min_runs: int = 2) -> None:
        self.root = Path(root)
        self.min_runs = int(max(1, min_runs))

    def _collect(self, filename: str) -> pd.DataFrame:
        frames: List[pd.DataFrame] = []
        for path in self.root.rglob(filename):
            try:
                frame = pd.read_csv(path)
                frame["source_file"] = str(path)
                frames.append(frame)
            except Exception as exc:
                print(f"[CrossSweepAggregator] skipping {path}: {exc}")
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def run(self, output_path: os.PathLike[str] | str) -> Dict[str, Any]:
        output = Path(output_path)
        output.mkdir(parents=True, exist_ok=True)
        runs = self._collect("cross_sweep_run_record.csv")
        predictors = self._collect("cross_sweep_predictor_records.csv")
        events = self._collect("cross_sweep_event_records.csv")
        regimes = self._collect("cross_sweep_accuracy_regime_records.csv")
        if "run_id" in runs:
            runs = runs.drop_duplicates("run_id")
        if "run_id" in predictors and "predictor" in predictors:
            predictors = predictors.drop_duplicates(["run_id", "predictor"])
        runs.to_csv(output / "aggregate_run_catalog.csv", index=False)
        predictors.to_csv(output / "aggregate_predictor_records.csv", index=False)
        events.to_csv(output / "aggregate_event_catalog.csv", index=False)
        regimes.to_csv(output / "aggregate_accuracy_regime_catalog.csv", index=False)

        stability_rows: List[Dict[str, Any]] = []
        n_runs = max(runs["run_id"].nunique() if "run_id" in runs else 0, 1)
        if not predictors.empty and "predictor" in predictors:
            for predictor, group in predictors.groupby("predictor"):
                run_count = group["run_id"].nunique() if "run_id" in group else len(group)
                if run_count < self.min_runs:
                    continue
                ranks = group.get("primary_rank", group.get("overall_rank", pd.Series(np.nan, index=group.index)))
                stability_rows.append({
                    "predictor": predictor,
                    "predictor_label": _friendly_name(predictor),
                    "n_runs": int(run_count),
                    "run_coverage": float(run_count / n_runs),
                    "median_rank": _safe_median(ranks),
                    "mean_rank": _safe_mean(ranks),
                    "top3_frequency": float(np.mean(ranks <= 3)),
                    "top5_frequency": float(np.mean(ranks <= 5)),
                    "mean_importance_score": _safe_mean(group.get("physics_importance_score", pd.Series(dtype=float))),
                    "importance_score_std": _safe_std(group.get("physics_importance_score", pd.Series(dtype=float))),
                    "median_transition_auc": _safe_median(group.get("primary_auc_oriented", pd.Series(dtype=float))),
                    "mean_event_detection_rate": _safe_mean(group.get("event_detection_rate", pd.Series(dtype=float))),
                    "median_lead_time_steps": _safe_median(group.get("mean_lead_time_steps", pd.Series(dtype=float))),
                    "median_future_error_r": _safe_median(group.get("best_error_leading_lag_r", pd.Series(dtype=float)).abs()),
                    "transition_significant_fraction": float(np.mean(group.get("primary_fdr_significant_0_05", pd.Series(False, index=group.index)).astype(bool))),
                    "future_error_significant_fraction": float(np.mean(group.get("best_error_fdr_significant_0_05", pd.Series(False, index=group.index)).astype(bool))),
                })
        stability = pd.DataFrame(stability_rows)
        if not stability.empty:
            stability = stability.sort_values(["top5_frequency", "median_rank", "mean_importance_score"], ascending=[False, True, False]).reset_index(drop=True)
            stability.insert(0, "stability_rank", np.arange(1, len(stability) + 1))
        stability.to_csv(output / "aggregate_predictor_stability.csv", index=False)
        try:
            (output / "aggregate_predictor_stability.md").write_text(stability.to_markdown(index=False) + "\n", encoding="utf-8")
        except Exception:
            pass
        summary = {
            "root": str(self.root),
            "n_runs": int(runs["run_id"].nunique()) if "run_id" in runs else 0,
            "n_predictor_records": len(predictors),
            "n_event_records": len(events),
            "min_runs": self.min_runs,
            "n_stable_predictors": len(stability),
        }
        (output / "cross_sweep_aggregation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary


# =============================================================================
# Legacy compatibility helpers
# =============================================================================


def inspect_model_quantities(model: Any, name_filter: Optional[str] = None) -> None:
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
            info = f"Tensor({value.device}), shape={tuple(value.shape)}"
        elif isinstance(value, np.ndarray):
            info = f"ndarray, shape={value.shape}"
        elif np.isscalar(value):
            info = f"scalar, value={value}"
        else:
            continue
        print(f"{name:32s} | {info}")
    print("=" * 75)


def detect_transition_events(vortex_count: Sequence[float], event_type: str = "both") -> Tuple[np.ndarray, np.ndarray]:
    values = np.asarray(vortex_count, dtype=float)
    delta = np.diff(values)
    if event_type == "nucleation":
        indices = np.where(delta > 0)[0] + 1
    elif event_type == "annihilation":
        indices = np.where(delta < 0)[0] + 1
    else:
        indices = np.where(delta != 0)[0] + 1
    if indices.size == 0:
        return np.array([], dtype=int), np.array([], dtype=str)
    groups = [[int(indices[0])]]
    for index in indices[1:]:
        if int(index) == groups[-1][-1] + 1:
            groups[-1].append(int(index))
        else:
            groups.append([int(index)])
    events = np.asarray([group[0] for group in groups], dtype=int)
    kinds = np.asarray(["nucleation" if delta[event - 1] > 0 else "annihilation" for event in events], dtype=str)
    return events, kinds


def _lag_correlate(reference: Sequence[float], signal: Sequence[float], max_lag: int) -> Tuple[float, int]:
    reference = np.asarray(reference, dtype=float)
    signal = np.asarray(signal, dtype=float)
    best_r, best_lag = _safe_pearson(reference, signal), 0
    for lag in range(1, min(int(max_lag), len(reference) - 3) + 1):
        lead_r = _safe_pearson(reference[lag:], signal[:-lag])
        lag_r = _safe_pearson(reference[:-lag], signal[lag:])
        if abs(lead_r) > abs(best_r):
            best_r, best_lag = lead_r, lag
        if abs(lag_r) > abs(best_r):
            best_r, best_lag = lag_r, -lag
    return best_r, best_lag


def estimate_lead_time(predictor: Sequence[float], events: Sequence[int],
                       max_window: int = 20, z_thresh: float = 1.5) -> Tuple[np.ndarray, np.ndarray, float, float]:
    values = np.asarray(predictor, dtype=float)
    quiet = np.ones(len(values), dtype=bool)
    for event in events:
        quiet[max(0, int(event) - max_window):min(len(values), int(event) + max_window + 1)] = False
    center, scale = _robust_scale(values[quiet] if quiet.any() else values)
    leads, detected = [], []
    for event in events:
        z = (values[max(0, int(event) - max_window):int(event)] - center) / scale
        run = 0
        for value in z[::-1]:
            if abs(value) >= z_thresh:
                run += 1
            else:
                break
        leads.append(run)
        detected.append(run > 0)
    return np.asarray(leads, dtype=int), np.asarray(detected, dtype=bool), center, scale


def rank_leading_indicators(predictor_dict: Mapping[str, Sequence[float]], vortex_count: Sequence[float],
                            trajectory_error: Sequence[float], Hext_range: Sequence[float], save_path: str,
                            event_type: str = "both", max_window: int = 20, z_thresh: float = 1.5,
                            max_lag: int = 15) -> pd.DataFrame:
    events, _ = detect_transition_events(vortex_count, event_type)
    rows = []
    for name, values in predictor_dict.items():
        leads, detected, center, scale = estimate_lead_time(values, events, max_window, z_thresh)
        r, lag = _lag_correlate(trajectory_error, values, max_lag)
        rows.append({"Predictor": name, "Detection Rate": detected.mean() if len(detected) else 0.0,
                     "Mean Lead Time (steps)": leads[detected].mean() if detected.any() else 0.0,
                     "Leading Lag r (vs error)": r, "Leading Lag (steps, +=leads)": lag,
                     "Baseline Mean": center, "Baseline Std": scale})
    result = pd.DataFrame(rows).sort_values(["Detection Rate", "Mean Lead Time (steps)"], ascending=False)
    folder = Path(save_path) / "leading_indicator_analysis"
    folder.mkdir(parents=True, exist_ok=True)
    result.to_csv(folder / f"leading_indicator_ranking_{event_type}.csv", index=False)
    return result


def plot_leading_indicators(Hext_range: Sequence[float], trajectory_error: Sequence[float],
                            predictor_dict: Mapping[str, Sequence[float]], vortex_count: Sequence[float],
                            general_title: str, save_path: str, event_type: str = "both", top_n: int = 6,
                            max_window: int = 20, z_thresh: float = 1.5) -> None:
    ranking = rank_leading_indicators(predictor_dict, vortex_count, trajectory_error, Hext_range,
                                      save_path, event_type, max_window, z_thresh)
    if ranking.empty:
        return
    folder = Path(save_path) / "leading_indicator_analysis"
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.plot(Hext_range, _robust_z(trajectory_error), color="black", linewidth=2.5, label="Trajectory error")
    for name in ranking["Predictor"].head(top_n):
        ax.plot(Hext_range, _robust_z(predictor_dict[name]), label=name)
    events, _ = detect_transition_events(vortex_count, event_type)
    for event in events:
        ax.axvline(np.asarray(Hext_range)[event], linestyle="--", alpha=0.4)
    ax.set_title(general_title + "\nLeading indicators")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(folder / f"leading_indicators_{event_type}.png", dpi=250)
    plt.close(fig)

