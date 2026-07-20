# -*- coding: utf-8 -*-
"""Minimal converged-state recorder used by the NeuralMAG M-H evaluator.

This file intentionally contains only the two pieces still used by the
streamlined evaluation program:

1. ``analyze_winding_components`` for vortex/core diagnostics.
2. ``PhysicsRecorder`` for one CSV row per converged external-field state.

The previous transition-ranking, publication-figure, manuscript, and
cross-sweep analysis pipeline has been removed from this version.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import label as connected_components

_EPS = 1.0e-12


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


def _masked_values(values: torch.Tensor, active_mask: torch.Tensor) -> torch.Tensor:
    selected = values[active_mask]
    return selected if selected.numel() else values.reshape(-1)


def _field_statistics(field: torch.Tensor, active_mask: torch.Tensor) -> Dict[str, float]:
    magnitude = _masked_values(torch.linalg.vector_norm(field, dim=-1), active_mask)
    return {
        "mean": float(magnitude.mean().item()),
        "std": float(magnitude.std(unbiased=False).item()),
        "max": float(magnitude.max().item()),
        "rms": float(torch.sqrt(torch.mean(magnitude.square())).item()),
    }


def _torque_statistics(
    spin: torch.Tensor,
    field: torch.Tensor,
    active_mask: torch.Tensor,
) -> Dict[str, float]:
    magnitude = torch.linalg.vector_norm(torch.cross(spin, field, dim=-1), dim=-1)
    magnitude = _masked_values(magnitude, active_mask)
    return {
        "mean": float(magnitude.mean().item()),
        "max": float(magnitude.max().item()),
        "rms": float(torch.sqrt(torch.mean(magnitude.square())).item()),
    }


def _energy_value(model: Any, attribute: str) -> float:
    return _as_float(getattr(model, attribute, None))


def _vector_error(
    reference: torch.Tensor,
    prediction: torch.Tensor,
    active_mask: torch.Tensor,
) -> Dict[str, float]:
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


def analyze_winding_components(
    winding_map: Any,
    *,
    relative_threshold: float = 0.25,
    absolute_threshold: float = 0.02,
    min_cells: int = 1,
    min_abs_charge: float = 0.05,
) -> Dict[str, float]:
    """Count connected positive and negative winding-density components."""
    array = np.squeeze(_as_numpy(winding_map, dtype=float))
    if array.ndim != 2:
        raise ValueError(f"Expected a 2-D winding map after squeeze; got {array.shape}.")

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

    result["total_core_count"] = int(
        result["positive_core_count"] + result["negative_core_count"]
    )
    return result


@dataclass(frozen=True)
class PhysicsSnapshot:
    values: Mapping[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.values)

    def __getitem__(self, key: str) -> Any:
        return self.values[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.values.get(key, default)


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
        projection = torch.as_tensor(
            projection_direction, dtype=dtype, device=device
        ).reshape(3)
        projection = projection / (torch.linalg.vector_norm(projection) + _EPS)

        active_fft = torch.linalg.vector_norm(film_fft.Spin, dim=-1) > _EPS
        active_unet = torch.linalg.vector_norm(film_unet.Spin, dim=-1) > _EPS
        active = active_fft & active_unet
        if not torch.any(active):
            raise ValueError("No active magnetic cells were found while recording physics.")

        n_active = int(active.sum().item())
        if cell_count is not None and int(cell_count) != n_active:
            print(
                f"[PhysicsRecorder] tensor mask has {n_active} active cells; "
                f"supplied cell_count={cell_count}."
            )

        data: Dict[str, Any] = {
            "mh_step": int(mh_step),
            "hext_scalar": float(hext_scalar),
            "hext_x": float(hext[0].item()),
            "hext_y": float(hext[1].item()),
            "hext_z": float(hext[2].item()),
        }

        fields = {"hd": "Hd", "he": "He", "ha": "Ha", "heff": "Heff"}
        for prefix, model in (("fft", film_fft), ("unet", film_unet)):
            for short, attribute in fields.items():
                field = getattr(model, attribute)
                for stat, value in _field_statistics(field, active).items():
                    data[f"{prefix}_{short}_{stat}"] = value
                for stat, value in _torque_statistics(model.Spin, field, active).items():
                    data[f"{prefix}_tau_{short}_{stat}"] = value

            data[f"{prefix}_e_demag"] = _energy_value(model, "Energy_demag")
            data[f"{prefix}_e_exchange"] = _energy_value(model, "Energy_excha")
            data[f"{prefix}_e_anis"] = _energy_value(model, "Energy_aniso")
            data[f"{prefix}_e_external"] = _energy_value(model, "Energy_exter")
            data[f"{prefix}_e_total"] = _energy_value(model, "Energy")

            spin_active = model.Spin[active]
            magnetization = spin_active.mean(dim=0)
            data[f"{prefix}_mx"] = float(magnetization[0].item())
            data[f"{prefix}_my"] = float(magnetization[1].item())
            data[f"{prefix}_mz"] = float(magnetization[2].item())
            data[f"{prefix}_mz_abs_mean"] = float(spin_active[:, 2].abs().mean().item())
            data[f"{prefix}_m_projection"] = float(
                torch.dot(magnetization, projection).item()
            )

        data.update(
            {
                "fft_winding_abs": _as_float(fft_winding_abs),
                "fft_winding_sum": _as_float(fft_winding_sum),
                "unet_winding_abs": _as_float(unet_winding_abs),
                "unet_winding_sum": _as_float(unet_winding_sum),
            }
        )

        for prefix, topology in (
            ("fft", fft_topology or {}),
            ("unet", unet_topology or {}),
        ):
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
        data["m_projection_abs_error"] = abs(
            data["unet_m_projection"] - data["fft_m_projection"]
        )

        data.update(
            {
                "fft_iterations": int(fft_iterations),
                "unet_iterations": int(unet_iterations),
                "fft_final_convergence_error": float(fft_final_convergence_error),
                "unet_final_convergence_error": float(unet_final_convergence_error),
                "fft_runtime_seconds": float(fft_runtime_seconds),
                "unet_runtime_seconds": float(unet_runtime_seconds),
            }
        )

        snapshot = PhysicsSnapshot(data)
        self.snapshots.append(snapshot)
        return snapshot

    def dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([snapshot.to_dict() for snapshot in self.snapshots])

    def save_csv(self, filename: os.PathLike[str] | str) -> pd.DataFrame:
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame = self.dataframe()
        frame.to_csv(path, index=False)
        return frame

