# -*- coding: utf-8 -*-
"""
 M-H diagnostics for the NeuralMAG accuracy project.

- error summary
- torque summary
- torque-error summary
- energy summary
- field summary
- performance summary
- magnetization-gradient summary
- exchange-energy-density summary
- winding-density summary
- gradient-magnitude, full gradient-tensor, and exchange rate-of-change summary
- exact rate/error overlay figures for gradient magnitude, gradient tensor,
  exchange-energy density, exchange-field vector, exchange-torque vector,
  demagnetizing-torque vector, total LLG drive, and winding density
"""

from __future__ import annotations

import os
from typing import Dict, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch


# =============================================================================
# Texture and loop-change calculations
# =============================================================================


def _layer0_training_tensor(spin):
    """Return layer 0 as ``(batch, 3, Nx, Ny)`` without changing values."""
    tensor = spin if isinstance(spin, torch.Tensor) else torch.as_tensor(spin)

    if tensor.ndim != 4:
        raise ValueError(
            "spin must have shape (Nx, Ny, Nz, 3) or "
            f"(batch, channels, Nx, Ny); received {tuple(tensor.shape)}."
        )

    # Already channel-first.
    if tensor.shape[1] >= 3 and tensor.shape[-1] != 3:
        return tensor[:, :3]

    # MAG2305 layout: (Nx, Ny, Nz, 3).
    if tensor.shape[-1] == 3:
        if tensor.shape[2] < 1:
            raise ValueError("The spin tensor contains no magnetic layers.")
        return tensor[:, :, 0, :].permute(2, 0, 1).unsqueeze(0)

    if tensor.shape[1] >= 3:
        return tensor[:, :3]

    raise ValueError(f"Could not identify magnetization channels in {tuple(tensor.shape)}.")


def _training_gradient_map(spin_channel_first):
    """Reproduce the finite-difference ``|grad(m)|`` map used in training."""
    if spin_channel_first.ndim != 4 or spin_channel_first.shape[1] < 3:
        raise ValueError("Expected spin shape (batch, >=3, Nx, Ny).")

    grad_sq = torch.zeros_like(spin_channel_first[:, 0])
    for component in range(3):
        m = spin_channel_first[:, component]

        m_xp = torch.roll(m, shifts=-1, dims=1)
        m_xm = torch.roll(m, shifts=1, dims=1)
        m_yp = torch.roll(m, shifts=-1, dims=2)
        m_ym = torch.roll(m, shifts=1, dims=2)

        m_xp[:, -1, :] = m[:, -1, :]
        m_xm[:, 0, :] = m[:, 0, :]
        m_yp[:, :, -1] = m[:, :, -1]
        m_ym[:, :, 0] = m[:, :, 0]

        dm_dx = (m_xp - m_xm) / 2.0
        dm_dy = (m_yp - m_ym) / 2.0
        grad_sq = grad_sq + dm_dx.square() + dm_dy.square()

    return torch.sqrt(grad_sq)


def _training_winding_map(spin_channel_first):
    """Reproduce the signed local winding-density map used in training."""
    if spin_channel_first.ndim != 4 or spin_channel_first.shape[1] < 2:
        raise ValueError("Expected spin shape (batch, >=2, Nx, Ny).")

    mx = spin_channel_first[:, 0]
    my = spin_channel_first[:, 1]

    mx_xp = torch.roll(mx, shifts=-1, dims=1)
    mx_xm = torch.roll(mx, shifts=1, dims=1)
    mx_yp = torch.roll(mx, shifts=-1, dims=2)
    mx_ym = torch.roll(mx, shifts=1, dims=2)

    my_xp = torch.roll(my, shifts=-1, dims=1)
    my_xm = torch.roll(my, shifts=1, dims=1)
    my_yp = torch.roll(my, shifts=-1, dims=2)
    my_ym = torch.roll(my, shifts=1, dims=2)

    mx_xp[:, -1, :] = mx[:, -1, :]
    mx_xm[:, 0, :] = mx[:, 0, :]
    my_xp[:, -1, :] = my[:, -1, :]
    my_xm[:, 0, :] = my[:, 0, :]

    mx_yp[:, :, -1] = mx[:, :, -1]
    mx_ym[:, :, 0] = mx[:, :, 0]
    my_yp[:, :, -1] = my[:, :, -1]
    my_ym[:, :, 0] = my[:, :, 0]

    dmx_dx = (mx_xp - mx_xm) / 2.0
    dmx_dy = (mx_yp - mx_ym) / 2.0
    dmy_dx = (my_xp - my_xm) / 2.0
    dmy_dy = (my_yp - my_ym) / 2.0

    return (dmx_dx * dmy_dy - dmy_dx * dmx_dy) / np.pi


def _strict_interior_mask(active_mask):
    if active_mask.ndim != 3:
        raise ValueError("active_mask must have shape (batch, Nx, Ny).")

    interior = torch.zeros_like(active_mask, dtype=torch.bool)
    if active_mask.shape[1] < 3 or active_mask.shape[2] < 3:
        return interior

    interior[:, 1:-1, 1:-1] = (
        active_mask[:, 1:-1, 1:-1]
        & active_mask[:, :-2, 1:-1]
        & active_mask[:, 2:, 1:-1]
        & active_mask[:, 1:-1, :-2]
        & active_mask[:, 1:-1, 2:]
    )
    return interior


def _selected_mean(values, mask):
    selected = values[mask]
    return float(selected.mean().item()) if selected.numel() else float("nan")


def _selected_max(values, mask):
    selected = values[mask]
    return float(selected.max().item()) if selected.numel() else float("nan")


def _scalar_spatial_variation_map(values_batch):
    """Magnitude of the spatial gradient of a scalar map."""
    if values_batch.ndim != 3:
        raise ValueError("values_batch must have shape (batch, Nx, Ny).")

    xp = torch.roll(values_batch, shifts=-1, dims=1)
    xm = torch.roll(values_batch, shifts=1, dims=1)
    yp = torch.roll(values_batch, shifts=-1, dims=2)
    ym = torch.roll(values_batch, shifts=1, dims=2)

    xp[:, -1, :] = values_batch[:, -1, :]
    xm[:, 0, :] = values_batch[:, 0, :]
    yp[:, :, -1] = values_batch[:, :, -1]
    ym[:, :, 0] = values_batch[:, :, 0]

    dv_dx = (xp - xm) / 2.0
    dv_dy = (yp - ym) / 2.0
    return torch.sqrt(dv_dx.square() + dv_dy.square())


@torch.no_grad()
def compute_gradient_spatial_variation_metrics(spin, active_threshold=1.0e-12):
    """Spatial variation of ``|grad(m)|`` and ``|grad(m)|^2`` in one state."""
    layer0 = _layer0_training_tensor(spin)
    active = torch.linalg.vector_norm(layer0[:, :3], dim=1) > active_threshold
    gradient = _training_gradient_map(layer0)
    exchange_proxy = gradient.square()

    return {
        "gradient_spatial_variation_mean": _selected_mean(
            _scalar_spatial_variation_map(gradient), active
        ),
        "exchange_proxy_spatial_variation_mean": _selected_mean(
            _scalar_spatial_variation_map(exchange_proxy), active
        ),
    }


@torch.no_grad()
def compute_training_texture_metrics(
    spin,
    exchange_energy=None,
    active_threshold=1.0e-12,
):
    """Compute scalar texture histories used by the retained M-H plots."""
    tensor = spin if isinstance(spin, torch.Tensor) else torch.as_tensor(spin)
    layer0 = _layer0_training_tensor(tensor)
    active = torch.linalg.vector_norm(layer0[:, :3], dim=1) > active_threshold
    interior = _strict_interior_mask(active)

    gradient = _training_gradient_map(layer0)
    winding_abs = _training_winding_map(layer0).abs()

    if tensor.ndim == 4 and tensor.shape[-1] == 3:
        full_active_count = int(
            (torch.linalg.vector_norm(tensor, dim=-1) > active_threshold).sum().item()
        )
    else:
        full_active_count = int(active.sum().item())

    exchange_density = float("nan")
    if exchange_energy is not None and full_active_count > 0:
        exchange_value = (
            float(exchange_energy.detach().cpu().item())
            if isinstance(exchange_energy, torch.Tensor)
            else float(exchange_energy)
        )
        exchange_density = exchange_value / full_active_count

    return {
        "gradient_training_grid_mean": float(gradient.mean().item()),
        "gradient_active_mean": _selected_mean(gradient, active),
        "gradient_interior_mean": _selected_mean(gradient, interior),
        "exchange_proxy_training_mean": float(gradient.square().mean().item()),
        "winding_training_abs_mean": _selected_mean(winding_abs, active),
        "winding_training_abs_max": _selected_max(winding_abs, active),
        "exchange_energy_density": exchange_density,
        "active_cell_count": full_active_count,
        "interior_layer0_cell_count": int(interior.sum().item()),
    }


def _training_first_derivative_all_cells(values, dim, spacing):
    """Finite difference used by the training-code gradient tensor."""
    if spacing <= 0:
        raise ValueError("Spatial spacing must be positive.")

    if values.shape[dim] < 2:
        return torch.zeros_like(values)

    derivative = torch.empty_like(values)

    center = [slice(None)] * values.ndim
    plus = [slice(None)] * values.ndim
    minus = [slice(None)] * values.ndim
    center[dim] = slice(1, -1)
    plus[dim] = slice(2, None)
    minus[dim] = slice(None, -2)

    if values.shape[dim] > 2:
        derivative[tuple(center)] = (
            values[tuple(plus)] - values[tuple(minus)]
        ) / (2.0 * spacing)

    first = [slice(None)] * values.ndim
    second = [slice(None)] * values.ndim
    first[dim] = 0
    second[dim] = 1
    derivative[tuple(first)] = (
        values[tuple(second)] - values[tuple(first)]
    ) / spacing

    last = [slice(None)] * values.ndim
    before_last = [slice(None)] * values.ndim
    last[dim] = -1
    before_last[dim] = -2
    derivative[tuple(last)] = (
        values[tuple(last)] - values[tuple(before_last)]
    ) / spacing

    return derivative


def _training_gradient_tensor_3d_from_mag2305(
    spin,
    dx=1.0,
    dy=1.0,
    dz=1.0,
):
    """Exact equivalent of training utils.magnetization_gradient_tensor_3d().

    MAG2305 stores spin as (Nx, Ny, layers, 3), whereas the training utility
    internally uses (B, Nx, Ny, layers, 3). This function only adds the batch
    dimension; the finite-difference math is kept the same.
    """
    m = spin if isinstance(spin, torch.Tensor) else torch.as_tensor(spin)

    if m.ndim != 4 or m.shape[-1] != 3:
        raise ValueError(
            "Expected MAG2305 spin shape (Nx, Ny, layers, 3); "
            f"received {tuple(m.shape)}."
        )

    m = m.unsqueeze(0)

    dm_dx = _training_first_derivative_all_cells(m, dim=1, spacing=dx)
    dm_dy = _training_first_derivative_all_cells(m, dim=2, spacing=dy)

    if m.shape[3] == 1:
        dm_dz = torch.zeros_like(m)
    elif m.shape[3] == 2:
        dz_pair = (m[:, :, :, 1, :] - m[:, :, :, 0, :]) / dz
        dm_dz = torch.stack((dz_pair, dz_pair), dim=3)
    else:
        dm_dz = _training_first_derivative_all_cells(m, dim=3, spacing=dz)

    return torch.stack((dm_dx, dm_dy, dm_dz), dim=-1)


def _training_gradient_tensor_rate_map(
    current_spin,
    previous_spin,
    dt=1.0,
    dx=1.0,
    dy=1.0,
    dz=1.0,
):
    """Exact local map produced by training utils.gradient_tensor_rate().

    The training call does not pass dt, dx, dy, or dz, so all four defaults
    are 1.0. The returned map has shape (1, Nx, Ny).
    """
    if dt <= 0:
        raise ValueError("dt must be positive.")

    current_tensor = (
        current_spin
        if isinstance(current_spin, torch.Tensor)
        else torch.as_tensor(current_spin)
    )
    previous_tensor = (
        previous_spin
        if isinstance(previous_spin, torch.Tensor)
        else torch.as_tensor(previous_spin)
    ).to(device=current_tensor.device, dtype=current_tensor.dtype)

    if current_tensor.shape != previous_tensor.shape:
        raise ValueError(
            "Current and previous spins must have identical shapes; "
            f"got {tuple(current_tensor.shape)} and "
            f"{tuple(previous_tensor.shape)}."
        )

    grad_now = _training_gradient_tensor_3d_from_mag2305(
        current_tensor, dx=dx, dy=dy, dz=dz
    )
    grad_prev = _training_gradient_tensor_3d_from_mag2305(
        previous_tensor, dx=dx, dy=dy, dz=dz
    )

    delta_grad = (grad_now - grad_prev) / dt
    return torch.sqrt(torch.sum(delta_grad.square(), dim=(3, 4, 5)))



def _training_shift_nonperiodic(values, shift, dim):
    """Exact boundary handling used by training utils._shift_nonperiodic()."""
    shifted = torch.roll(values, shifts=shift, dims=dim)
    index = [slice(None)] * values.ndim

    if shift == 1:
        index[dim] = 0
        shifted[tuple(index)] = values[tuple(index)]
    elif shift == -1:
        index[dim] = -1
        shifted[tuple(index)] = values[tuple(index)]
    else:
        raise ValueError("Only nearest-neighbor shifts +/-1 are supported.")

    return shifted


def _training_exchange_field_from_mag2305(
    spin,
    Ms=1000.0,
    Ax=0.5e-6,
    cell_nm=(3.0, 3.0, 3.0),
):
    """Exact equivalent of training utils.exchange_field() for MAG2305 spin.

    MAG2305 layout is (Nx, Ny, layers, 3); training internally uses
    (B, Nx, Ny, layers, 3). The coefficient and finite-boundary treatment
    match utils.exchange_field() exactly.
    """
    if Ms <= 0:
        raise ValueError("Ms must be positive.")
    if Ax < 0:
        raise ValueError("Ax must be nonnegative.")
    if len(cell_nm) != 3 or any(float(d) <= 0 for d in cell_nm):
        raise ValueError("cell_nm must contain three positive cell sizes.")

    m = spin if isinstance(spin, torch.Tensor) else torch.as_tensor(spin)
    if m.ndim != 4 or m.shape[-1] != 3:
        raise ValueError(
            "Expected MAG2305 spin shape (Nx, Ny, layers, 3); "
            f"received {tuple(m.shape)}."
        )
    m = m.unsqueeze(0)

    coeff_x = 2.0e14 * float(Ax) / (float(Ms) * float(cell_nm[0]) ** 2)
    coeff_y = 2.0e14 * float(Ax) / (float(Ms) * float(cell_nm[1]) ** 2)
    coeff_z = 2.0e14 * float(Ax) / (float(Ms) * float(cell_nm[2]) ** 2)

    hx = coeff_x * (
        (_training_shift_nonperiodic(m, 1, 1) - m)
        + (_training_shift_nonperiodic(m, -1, 1) - m)
    )
    hy = coeff_y * (
        (_training_shift_nonperiodic(m, 1, 2) - m)
        + (_training_shift_nonperiodic(m, -1, 2) - m)
    )
    hz = coeff_z * (
        (_training_shift_nonperiodic(m, 1, 3) - m)
        + (_training_shift_nonperiodic(m, -1, 3) - m)
    )
    return hx + hy + hz


def _training_exchange_torque_rate_map(
    current_spin,
    previous_spin,
    dt=1.0,
    Ms=1000.0,
    Ax=0.5e-6,
    cell_nm=(3.0, 3.0, 3.0),
):
    """Exact local map produced by training utils.exchange_torque_rate().

    R_torque(x,y) = ||tau_ex(t)-tau_ex(t-1)||_(layers,xyz) / dt,
    tau_ex = m x H_ex. The current training call leaves dt at its default 1.0.
    """
    if dt <= 0:
        raise ValueError("dt must be positive.")

    current = current_spin if isinstance(current_spin, torch.Tensor) else torch.as_tensor(current_spin)
    previous = previous_spin if isinstance(previous_spin, torch.Tensor) else torch.as_tensor(previous_spin)
    previous = previous.to(device=current.device, dtype=current.dtype)
    if current.shape != previous.shape:
        raise ValueError(
            "Current and previous spins must have identical shapes; "
            f"got {tuple(current.shape)} and {tuple(previous.shape)}."
        )

    current_grid = current.unsqueeze(0)
    previous_grid = previous.unsqueeze(0)
    h_now = _training_exchange_field_from_mag2305(
        current, Ms=Ms, Ax=Ax, cell_nm=cell_nm
    )
    h_prev = _training_exchange_field_from_mag2305(
        previous, Ms=Ms, Ax=Ax, cell_nm=cell_nm
    )
    tau_now = torch.linalg.cross(current_grid, h_now, dim=-1)
    tau_prev = torch.linalg.cross(previous_grid, h_prev, dim=-1)
    delta_tau = (tau_now - tau_prev) / dt

    # Collapse both the magnetic-layer axis and vector-component axis,
    # exactly matching training utils._collapse_vector_layers().
    return torch.sqrt(torch.sum(delta_tau.square(), dim=(-1, -2)))


@torch.no_grad()
def compute_exact_loop_change_metrics(
    current_spin,
    previous_spin=None,
    *,
    delta_hext_oe=None,
    active_threshold=1.0e-12,
    Ms=1000.0,
    Ax=0.5e-6,
    cell_nm=(3.0, 3.0, 3.0),
):
    """Compute exact cellwise map changes before spatial averaging.

    For a map ``q(r)`` this evaluates ``mean(|q_i(r)-q_(i-1)(r)|)``.
    It does not use a difference of spatial means.
    """
    current = _layer0_training_tensor(current_spin)
    current_active = torch.linalg.vector_norm(current[:, :3], dim=1) > active_threshold
    current_gradient = _training_gradient_map(current)
    current_exchange_proxy = current_gradient.square()
    current_winding = _training_winding_map(current)

    result = {
        # Change in the scalar gradient-magnitude map:
        #     ||grad(m_i)| - |grad(m_(i-1))||
        "gradient_loop_abs_change_mean": float("nan"),
        "gradient_loop_abs_change_per_oe": float("nan"),

        # Full in-plane magnetization-gradient-tensor change:
        #     ||grad(m_i - m_(i-1))||_F
        # The Frobenius norm includes dm_x/dx, dm_x/dy, dm_y/dx,
        # dm_y/dy, dm_z/dx, and dm_z/dy on layer 0.
        "gradient_tensor_loop_abs_change_mean": float("nan"),
        "gradient_tensor_loop_abs_change_per_oe": float("nan"),

        # Spatial mean of the EXACT local map used by the current
        # gradient_tensor_rate training loss. This uses both magnetic layers,
        # x/y/z derivatives, and the training defaults dt=dx=dy=dz=1.
        "gradient_tensor_training_rate_mean": float("nan"),

        # Spatial mean of the EXACT local map used by the current
        # exchange_torque_rate training loss. This is R_torque before alpha
        # is applied in w = 1 + alpha * R_torque.
        "exchange_torque_training_rate_mean": float("nan"),

        "exchange_proxy_loop_abs_change_mean": float("nan"),
        "exchange_proxy_loop_abs_change_per_oe": float("nan"),
        "winding_map_loop_abs_change_mean": float("nan"),
        "winding_map_loop_abs_change_per_oe": float("nan"),
    }

    if previous_spin is None:
        return result

    # Exact training-code gradient-tensor-rate map. The training loss uses
    # this local map directly (after unsqueeze/broadcast into its weight).
    training_rate_map = _training_gradient_tensor_rate_map(
        current_spin,
        previous_spin,
        dt=1.0,
        dx=1.0,
        dy=1.0,
        dz=1.0,
    )
    result["gradient_tensor_training_rate_mean"] = float(
        training_rate_map.mean().item()
    )

    exchange_torque_rate_map = _training_exchange_torque_rate_map(
        current_spin,
        previous_spin,
        dt=1.0,
        Ms=Ms,
        Ax=Ax,
        cell_nm=cell_nm,
    )
    result["exchange_torque_training_rate_mean"] = float(
        exchange_torque_rate_map.mean().item()
    )

    previous = _layer0_training_tensor(previous_spin).to(
        device=current.device,
        dtype=current.dtype,
    )
    previous_active = torch.linalg.vector_norm(previous[:, :3], dim=1) > active_threshold
    compare_mask = current_active & previous_active
    if not torch.any(compare_mask):
        return result

    previous_gradient = _training_gradient_map(previous)
    previous_exchange_proxy = previous_gradient.square()
    previous_winding = _training_winding_map(previous)

    gradient_change = _selected_mean(
        torch.abs(current_gradient - previous_gradient), compare_mask
    )

    # Because the finite-difference operator is linear,
    # _training_gradient_map(current - previous) equals the Frobenius norm
    # of grad(m_i) - grad(m_(i-1)) at each cell. Unlike the scalar quantity
    # above, this also detects changes in gradient direction/component makeup
    # when |grad(m)| itself stays nearly unchanged.
    gradient_tensor_change_map = _training_gradient_map(current - previous)
    gradient_tensor_change = _selected_mean(
        gradient_tensor_change_map,
        compare_mask,
    )

    exchange_proxy_change = _selected_mean(
        torch.abs(current_exchange_proxy - previous_exchange_proxy), compare_mask
    )
    winding_change = _selected_mean(
        torch.abs(current_winding - previous_winding), compare_mask
    )

    result["gradient_loop_abs_change_mean"] = gradient_change
    result["gradient_tensor_loop_abs_change_mean"] = gradient_tensor_change
    result["exchange_proxy_loop_abs_change_mean"] = exchange_proxy_change
    result["winding_map_loop_abs_change_mean"] = winding_change

    if delta_hext_oe is not None:
        delta_h = abs(float(delta_hext_oe))
        if delta_h > 0.0:
            result["gradient_loop_abs_change_per_oe"] = gradient_change / delta_h
            result["gradient_tensor_loop_abs_change_per_oe"] = (
                gradient_tensor_change / delta_h
            )
            result["exchange_proxy_loop_abs_change_per_oe"] = (
                exchange_proxy_change / delta_h
            )
            result["winding_map_loop_abs_change_per_oe"] = winding_change / delta_h

    return result


# =============================================================================
# Plot helpers
# =============================================================================


def _output_folder(save_path_summary):
    folder = os.fspath(save_path_summary)
    os.makedirs(folder, exist_ok=True)
    return folder


def _set_reversed_hext_axis(ax, hext_range):
    values = np.asarray(hext_range, dtype=float)
    maximum = float(np.nanmax(values))
    minimum = float(np.nanmin(values))
    pad = 0.05 * (maximum - minimum if maximum != minimum else 1.0)
    ax.set_xlim(maximum + pad, minimum - pad)


def _set_ascending_hext_axis(ax, hext_range, pad_fraction=0.0):
    """Display Hext conventionally from negative on the left to positive right."""
    values = np.asarray(hext_range, dtype=float)
    minimum = float(np.nanmin(values))
    maximum = float(np.nanmax(values))
    pad = pad_fraction * (maximum - minimum if maximum != minimum else 1.0)
    ax.set_xlim(minimum - pad, maximum + pad)


def _combined_legend(ax, twin):
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = twin.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper right")


# =============================================================================
# Retained plots
# =============================================================================


def plot_magnetization_gradient_vs_hext(
    general_title_summary,
    save_path_summary,
    hext_range,
    gradient_fft,
    gradient_unet,
):
    folder = _output_folder(save_path_summary)
    fig, axes = plt.subplots(1, 3, figsize=(19, 6.2), sharex=True)
    fig.suptitle(
        "Magnetization-Gradient Definitions Across the M-H Sweep\n\n"
        + general_title_summary,
        fontsize=13,
        fontweight="bold",
    )

    panels = (
        (
            "gradient_training_grid_mean",
            "Exact Training Map: Full-Grid Mean",
            r"Mean $|\nabla m|$ [cell$^{-1}$]",
            "Includes zero cells and sample-boundary gradients.",
        ),
        (
            "gradient_active_mean",
            "Exact Training Map: Magnetic Centers",
            r"Mean $|\nabla m|$ [cell$^{-1}$]",
            "Excludes nonmagnetic center cells.",
        ),
        (
            "gradient_interior_mean",
            "Strict Magnetic Interior",
            r"Mean $|\nabla m|$ [cell$^{-1}$]",
            "Uses cells with four magnetic in-plane neighbors.",
        ),
    )

    for ax, (key, title, ylabel, subtitle) in zip(axes, panels):
        ax.plot(hext_range, gradient_fft[key], lw=2.3, label="FFT/LLG")
        ax.plot(hext_range, gradient_unet[key], lw=2.3, label="UNet/LLG")
        ax.set_title(title + "\n" + subtitle, fontsize=10.5, fontweight="bold")
        ax.set_xlabel(r"External Field $H_{ext}$ [Oe]")
        ax.set_ylabel(ylabel)
        _set_reversed_hext_axis(ax, hext_range)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(
        os.path.join(folder, "magnetization_gradient_definitions_vs_hext.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_training_winding_density_vs_hext(
    general_title_summary,
    save_path_summary,
    hext_range,
    winding_fft,
    winding_unet,
):
    folder = _output_folder(save_path_summary)
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.2), sharex=True)
    fig.suptitle(
        "Training Winding-Density Signal Across the M-H Sweep\n\n"
        + general_title_summary,
        fontsize=13,
        fontweight="bold",
    )

    panels = (
        (
            "winding_training_abs_mean",
            r"Mean Local Winding Signal $\langle |w| \rangle$",
            r"Mean $|w|$ on magnetic cells",
        ),
        (
            "winding_training_abs_max",
            r"Peak Local Winding Signal $\max |w|$",
            r"Maximum $|w|$ on magnetic cells",
        ),
    )

    for ax, (key, title, ylabel) in zip(axes, panels):
        ax.plot(hext_range, winding_fft[key], lw=2.3, label="FFT/LLG")
        ax.plot(hext_range, winding_unet[key], lw=2.3, label="UNet/LLG")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel(r"External Field $H_{ext}$ [Oe]")
        ax.set_ylabel(ylabel)
        _set_reversed_hext_axis(ax, hext_range)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(
        os.path.join(folder, "training_winding_density_vs_hext.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_exchange_energy_density_vs_hext(
    general_title_summary,
    save_path_summary,
    hext_range,
    exchange_fft,
    exchange_unet,
):
    folder = _output_folder(save_path_summary)
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.2), sharex=True)
    fig.suptitle(
        "Exchange-Energy Signals Across the M-H Sweep\n\n" + general_title_summary,
        fontsize=13,
        fontweight="bold",
    )

    panels = (
        (
            "exchange_proxy_training_mean",
            r"Training Proxy: $\langle |\nabla m|^2 \rangle$",
            r"Mean $|\nabla m|^2$ [cell$^{-2}$]",
        ),
        (
            "exchange_energy_density",
            "MAG2305 Mean Exchange-Energy Density",
            r"Exchange-energy density [erg/cm$^3$]",
        ),
    )

    for ax, (key, title, ylabel) in zip(axes, panels):
        ax.plot(hext_range, exchange_fft[key], lw=2.3, label="FFT/LLG")
        ax.plot(hext_range, exchange_unet[key], lw=2.3, label="UNet/LLG")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel(r"External Field $H_{ext}$ [Oe]")
        ax.set_ylabel(ylabel)
        _set_reversed_hext_axis(ax, hext_range)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(
        os.path.join(folder, "exchange_energy_density_vs_hext.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_gradient_change_rate_summary(
    general_title_summary,
    save_path_summary,
    hext_range,
    spatial_fft,
    spatial_unet,
    loop_fft,
    loop_unet,
):
    """Six-panel summary containing both gradient-rate definitions.

    The scalar gradient-magnitude rate is

        mean(||grad(m_i)| - |grad(m_(i-1))||) / |delta Hext|.

    The fourth panel is the exact training-code gradient_tensor_rate
    quantity, spatially averaged only for visualization. It uses both layers,
    x/y/z derivatives, and the same default dt=dx=dy=dz=1 used by train.py.
    """
    folder = _output_folder(save_path_summary)
    fig, axes = plt.subplots(2, 3, figsize=(21, 11), sharex=True)
    fig.suptitle(
        "Magnetization-Gradient and Exchange Change Across the M-H Sweep\n\n"
        + general_title_summary,
        fontsize=13,
        fontweight="bold",
    )

    panels = (
        (
            spatial_fft,
            spatial_unet,
            "gradient_spatial_variation_mean",
            "Spatial Variation of Gradient Magnitude",
            r"Mean $|\nabla(|\nabla m|)|$ [cell$^{-2}$]",
        ),
        (
            spatial_fft,
            spatial_unet,
            "exchange_proxy_spatial_variation_mean",
            "Spatial Variation of Exchange Proxy",
            r"Mean $|\nabla(|\nabla m|^2)|$ [cell$^{-3}$]",
        ),
        (
            loop_fft,
            loop_unet,
            "gradient_loop_abs_change_per_oe",
            "Gradient-Magnitude Rate",
            r"Mean $|\Delta|\nabla m||/|\Delta H_{ext}|$ "
            r"[cell$^{-1}$/Oe]",
        ),
        (
            loop_fft,
            loop_unet,
            "gradient_tensor_training_rate_mean",
            "Training Gradient-Tensor Rate (Exact Loss Quantity)",
            r"Full-grid mean $||\Delta(\nabla m)||_F$ "
            r"(training defaults: $dt=dx=dy=dz=1$)",
        ),
        (
            loop_fft,
            loop_unet,
            "exchange_proxy_loop_abs_change_per_oe",
            "Exchange-Proxy Rate",
            r"Mean $|\Delta(|\nabla m|^2)|/|\Delta H_{ext}|$ "
            r"[cell$^{-2}$/Oe]",
        ),
        (
            loop_fft,
            loop_unet,
            "exchange_energy_density_loop_abs_change_per_oe",
            "Physical Exchange-Energy-Density Rate",
            r"Mean $|\Delta\epsilon_{ex}|/|\Delta H_{ext}|$ "
            r"[erg cm$^{-3}$/Oe]",
        ),
    )

    for ax, (fft_data, unet_data, key, title, ylabel) in zip(axes.flat, panels):
        ax.plot(hext_range, fft_data[key], lw=2.3, label="FFT/LLG")
        ax.plot(hext_range, unet_data[key], lw=2.3, label="UNet/LLG")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel(r"External Field $H_{ext}$ [Oe]")
        ax.set_ylabel(ylabel)
        _set_ascending_hext_axis(ax, hext_range)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(
        os.path.join(folder, "gradient_change_rate_summary_vs_hext.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_training_gradient_tensor_rate_vs_hext(
    general_title_summary,
    save_path_summary,
    hext_range,
    loop_fft,
    loop_unet,
):
    """Standalone plot of the exact gradient_tensor_rate training quantity."""
    folder = _output_folder(save_path_summary)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(
        hext_range,
        loop_fft["gradient_tensor_training_rate_mean"],
        lw=2.4,
        label="FFT/LLG",
    )
    ax.plot(
        hext_range,
        loop_unet["gradient_tensor_training_rate_mean"],
        lw=2.4,
        label="UNet/LLG",
    )
    ax.set_title(
        "Training Gradient-Tensor Rate Across the M-H Sweep\n"
        + general_title_summary,
        fontsize=11,
        fontweight="bold",
    )
    ax.set_xlabel(r"External Field $H_{ext}$ [Oe]")
    ax.set_ylabel(
        r"Full-grid mean $||\Delta(\nabla m)||_F$ "
        r"($dt=dx=dy=dz=1$)"
    )
    _set_ascending_hext_axis(ax, hext_range)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(
        os.path.join(folder, "training_gradient_tensor_rate_vs_hext.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)




def plot_training_weight_error_overlays(
    general_title_summary,
    save_path_summary,
    hext_range,
    loop_fft,
    trajectory_mae,
    alpha=0.5,
):
    """Create the two mentor-requested weight/error plots.

    The rate signals are evaluated on the FFT/LLG reference trajectory because
    the training weight is a function of the magnetization input, not of the
    UNet prediction. Each curve is the spatial mean of the exact local training
    multiplier w(x,y) = 1 + alpha * R(x,y). The trajectory error on the second
    y-axis is the FFT-vs-UNet spin MAE already recorded by this evaluator.
    """
    folder = _output_folder(save_path_summary)
    hext = np.asarray(hext_range, dtype=float)
    trajectory = np.asarray(trajectory_mae, dtype=float)

    specs = (
        (
            "gradient_tensor_training_rate_mean",
            r"$R_{grad}$",
            r"$w_{grad}=1+\alpha R_{grad}$",
            "R_grad_weight_vs_hext_with_trajectory_error.png",
        ),
        (
            "exchange_torque_training_rate_mean",
            r"$R_{torque}$",
            r"$w_{torque}=1+\alpha R_{torque}$",
            "R_torque_weight_vs_hext_with_trajectory_error.png",
        ),
    )

    for rate_key, rate_label, weight_label, filename in specs:
        rate = np.asarray(loop_fft[rate_key], dtype=float)
        weight = 1.0 + float(alpha) * rate

        fig, ax_weight = plt.subplots(figsize=(8.5, 6.2))
        ax_error = ax_weight.twinx()

        ax_weight.plot(
            hext,
            weight,
            linewidth=2.5,
            label=weight_label + rf" ($\alpha={float(alpha):g}$)",
        )
        ax_error.plot(
            hext,
            trajectory,
            linewidth=2.1,
            linestyle="--",
            color="black",
            label="Magnetization trajectory error",
        )

        ax_weight.set_title(
            rate_label + " Training Weight vs. External Field\n" + general_title_summary,
            fontsize=11,
            fontweight="bold",
        )
        ax_weight.set_xlabel(r"External Field $H_{ext}$ [Oe]")
        ax_weight.set_ylabel("Spatial mean training weight")
        ax_error.set_ylabel("Magnetization trajectory MAE")
        _set_ascending_hext_axis(ax_weight, hext)
        ax_weight.grid(True, linestyle="--", alpha=0.35)

        finite_error = trajectory[np.isfinite(trajectory)]
        if finite_error.size:
            error_max = float(np.max(finite_error))
            ax_error.set_ylim(0.0, 1.05 * error_max if error_max > 0.0 else 1.0)

        _combined_legend(ax_weight, ax_error)
        fig.tight_layout()
        fig.savefig(
            os.path.join(folder, filename),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)


def plot_physics_vector_rate_summary(
    general_title_summary,
    save_path_summary,
    hext_range,
    loop_fft,
    loop_unet,
):
    """Four-panel summary of exact vector-map rates between M-H states.

    Every quantity is calculated cell by cell as the Euclidean magnitude of
    the vector difference, spatially averaged, and divided by |delta Hext|.
    The total LLG drive is MAG2305's field-scale drive before multiplication
    by the gyromagnetic factor and time step.
    """
    folder = _output_folder(save_path_summary)
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), sharex=True)
    fig.suptitle(
        "Vector Physics Rates Across the M-H Sweep\n\n"
        + general_title_summary,
        fontsize=13,
        fontweight="bold",
    )

    panels = (
        (
            "exchange_field_vector_loop_abs_change_per_oe",
            "Exchange-Field Vector Rate",
            r"Mean $||\Delta H_{ex}||_2/|\Delta H_{ext}|$ [Oe/Oe]",
        ),
        (
            "exchange_torque_vector_loop_abs_change_per_oe",
            "Exchange-Torque Vector Rate",
            r"Mean $||\Delta(m\times H_{ex})||_2/|\Delta H_{ext}|$ "
            r"[Oe/Oe]",
        ),
        (
            "demag_torque_vector_loop_abs_change_per_oe",
            "Demagnetizing-Torque Vector Rate",
            r"Mean $||\Delta(m\times H_d)||_2/|\Delta H_{ext}|$ "
            r"[Oe/Oe]",
        ),
        (
            "total_llg_drive_vector_loop_abs_change_per_oe",
            "Total Reduced-LLG-Drive Vector Rate",
            r"Mean $||\Delta F_{LLG}||_2/|\Delta H_{ext}|$ [Oe/Oe]",
        ),
    )

    for ax, (key, title, ylabel) in zip(axes.flat, panels):
        ax.plot(hext_range, loop_fft[key], lw=2.3, label="FFT/LLG")
        ax.plot(hext_range, loop_unet[key], lw=2.3, label="UNet/LLG")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel(r"External Field $H_{ext}$ [Oe]")
        ax.set_ylabel(ylabel)
        _set_reversed_hext_axis(ax, hext_range)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(
        os.path.join(folder, "physics_vector_rate_summary_vs_hext.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_full_energy_summary(
    general_title_summary,
    save_path_summary,
    full_data_fft,
    full_data_unet,
    hext_range,
):
    folder = _output_folder(save_path_summary)
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), sharex=True)
    fig.suptitle(
        "Full M-H Curve Energy Summary\n\n" + general_title_summary,
        fontsize=13,
        fontweight="bold",
    )

    panels = (
        ("demag", r"Demagnetizing Energy $E_{demag}$"),
        ("anis", r"Anisotropy Energy $E_{anis}$"),
        ("excha", r"Exchange Energy $E_{ex}$"),
        ("exter", r"External-Field Energy $E_{ext}$"),
    )

    for ax, (key, title) in zip(axes.flat, panels):
        ax.plot(hext_range, full_data_fft[key], lw=2.2, label="FFT/LLG")
        ax.plot(hext_range, full_data_unet[key], lw=2.2, label="UNet/LLG")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel(r"External Field $H_{ext}$ [Oe]")
        ax.set_ylabel("MAG2305 energy value [cgs code units]")
        _set_reversed_hext_axis(ax, hext_range)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(
        os.path.join(folder, "full_equilibrium_energy_summary.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(hext_range, full_data_fft["total"], lw=2.5, label="FFT/LLG")
    ax.plot(hext_range, full_data_unet["total"], lw=2.5, label="UNet/LLG")
    ax.set_title("Total MAG2305 Energy Across the M-H Sweep\n" + general_title_summary)
    ax.set_xlabel(r"External Field $H_{ext}$ [Oe]")
    ax.set_ylabel("MAG2305 energy value [cgs code units]")
    _set_reversed_hext_axis(ax, hext_range)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(
        os.path.join(folder, "total_energy_summary.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_performance_summary(
    general_title_summary,
    save_path_summary,
    performance_fft,
    performance_unet,
    hext_range,
):
    folder = _output_folder(save_path_summary)
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), sharex=True)
    fig.suptitle(
        "Performance Summary\n\n" + general_title_summary,
        fontsize=13,
        fontweight="bold",
    )

    panels = (
        ("iters", "Solver Iterations per Field Step", "Iteration count"),
        ("vortices", "Detected Vortex/Core Components", "Component count"),
        ("mz", r"Mean Out-of-Plane Magnetization $|m_z|$", r"Mean $|m_z|$"),
        ("time", "Execution Time per Field Step", "Runtime [s]"),
    )

    for ax, (key, title, ylabel) in zip(axes.flat, panels):
        ax.plot(hext_range, performance_fft[key], lw=2.2, label="FFT/LLG")
        ax.plot(hext_range, performance_unet[key], lw=2.2, label="UNet/LLG")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel(r"External Field $H_{ext}$ [Oe]")
        ax.set_ylabel(ylabel)
        _set_reversed_hext_axis(ax, hext_range)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(
        os.path.join(folder, "performance_summary.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_error_summary(
    general_title_summary,
    save_path_summary,
    hext_range,
    inst_hd_mae,
    trajectory_mae,
    exchange_field_mae,
    anisotropy_field_mae,
    y_limits=None,
):
    folder = _output_folder(save_path_summary)
    print("Generating comprehensive 4-panel error tracking analysis...")

    default_limits = {
        "hd": (0.0, 400.0),
        "trajectory": (0.0, 0.75),
        "exchange": (0.0, 400.0),
        "anisotropy": (0.0, 400.0),
    }
    fixed_limits = default_limits.copy()
    if y_limits is not None:
        fixed_limits.update(y_limits)

    panels = (
        (
            inst_hd_mae,
            "darkorange",
            "UNet Demagnetizing-Field Error",
            r"$H_{demag}$ component MAE [Oe]",
            "hd",
        ),
        (
            trajectory_mae,
            "crimson",
            "Magnetization Trajectory Error",
            r"Magnetization component MAE",
            "trajectory",
        ),
        (
            exchange_field_mae,
            "purple",
            "Exchange-Field Error",
            r"$H_{ex}$ component MAE [Oe]",
            "exchange",
        ),
        (
            anisotropy_field_mae,
            "teal",
            "Anisotropy-Field Error",
            r"$H_{anis}$ component MAE [Oe]",
            "anisotropy",
        ),
    )

    # Comprehensive four-panel plot.
    fig, axes = plt.subplots(2, 2, figsize=(15, 12), sharex=True)
    fig.suptitle(
        "Field and Magnetization Error Summary\n\n" + general_title_summary,
        fontsize=13,
        fontweight="bold",
    )

    for ax, (data, color, title, ylabel, key) in zip(axes.flat, panels):
        values = np.asarray(data, dtype=float)
        ax.plot(hext_range, values, color=color, lw=2.2, label=title)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel(r"External Field $H_{ext}$ [Oe]")
        ax.set_ylabel(ylabel)
        ax.set_ylim(*fixed_limits[key])
        _set_ascending_hext_axis(ax, hext_range)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(fontsize=9)

        finite = values[np.isfinite(values)]
        if finite.size and (
            float(np.min(finite)) < fixed_limits[key][0]
            or float(np.max(finite)) > fixed_limits[key][1]
        ):
            print(
                f"[plot_error_summary] WARNING: {key} data exceed fixed "
                f"limits {fixed_limits[key]}."
            )

    fig.tight_layout()
    fig.savefig(
        os.path.join(folder, "comprehensive_error_analysis.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    # Save the two main error histories as standalone 8 x 6 plots as well.
    standalone = (
        (
            inst_hd_mae,
            "darkorange",
            "UNet Demagnetizing-Field Error",
            r"$H_{demag}$ component MAE [Oe]",
            "hd",
            "hdemag_error_vs_hext.png",
        ),
        (
            trajectory_mae,
            "crimson",
            "Magnetization Trajectory Error",
            "Magnetization component MAE",
            "trajectory",
            "magnetization_trajectory_error_vs_hext.png",
        ),
    )

    for data, color, title, ylabel, key, filename in standalone:
        fig, ax = plt.subplots(figsize=(8, 6))
        values = np.asarray(data, dtype=float)
        ax.plot(hext_range, values, color=color, lw=2.4)
        ax.set_title(title + "\n" + general_title_summary, fontsize=11, fontweight="bold")
        ax.set_xlabel(r"External Field $H_{ext}$ [Oe]")
        ax.set_ylabel(ylabel)
        ax.set_ylim(*fixed_limits[key])
        _set_ascending_hext_axis(ax, hext_range)
        ax.grid(True, linestyle="--", alpha=0.4)
        fig.tight_layout()
        fig.savefig(
            os.path.join(folder, filename),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)


def plot_fields_summary(
    general_title_summary,
    save_path_summary,
    hext_range,
    he_fft,
    he_unet,
    ha_fft,
    ha_unet,
    hd_fft,
    hd_unet,
    heff_fft,
    heff_unet,
):
    folder = _output_folder(save_path_summary)
    print("Generating final 4-panel physical field summary plot...")

    fig, axes = plt.subplots(2, 2, figsize=(15, 11), sharex=True)
    fig.suptitle(
        "Field Components Across the M-H Sweep\n\n" + general_title_summary,
        fontsize=13,
        fontweight="bold",
    )

    panels = (
        (he_fft, he_unet, r"Exchange Field $H_{ex}$"),
        (ha_fft, ha_unet, r"Anisotropy Field $H_{anis}$"),
        (hd_fft, hd_unet, r"Demagnetizing Field $H_{demag}$"),
        (heff_fft, heff_unet, r"Effective Field $H_{eff}$"),
    )

    for ax, (fft_data, unet_data, title) in zip(axes.flat, panels):
        ax.plot(hext_range, fft_data, lw=2.3, label="FFT/LLG")
        ax.plot(hext_range, unet_data, lw=2.3, label="UNet/LLG")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel(r"External Field $H_{ext}$ [Oe]")
        ax.set_ylabel("Mean field magnitude [Oe]")
        _set_reversed_hext_axis(ax, hext_range)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(
        os.path.join(folder, "all_internal_fields_mh_sweep.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_torque_summary(
    general_title_summary,
    save_path_summary,
    hext_range,
    torque_fft,
    torque_unet,
):
    folder = _output_folder(save_path_summary)
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), sharex=True)
    fig.suptitle(
        "Torque Summary Across the M-H Sweep\n\n" + general_title_summary,
        fontsize=13,
        fontweight="bold",
    )

    panels = (
        ("he", "Exchange Torque", r"Mean $|m\times H_{ex}|$ [Oe]"),
        ("ha", "Anisotropy Torque", r"Mean $|m\times H_{anis}|$ [Oe]"),
        ("hd", "Demagnetizing Torque", r"Mean $|m\times H_{demag}|$ [Oe]"),
        ("heff", "Effective-Field Torque", r"Mean $|m\times H_{eff}|$ [Oe]"),
    )

    for ax, (key, title, ylabel) in zip(axes.flat, panels):
        ax.plot(hext_range, torque_fft[key], lw=2.3, label="FFT/LLG")
        ax.plot(hext_range, torque_unet[key], lw=2.3, label="UNet/LLG")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel(r"External Field $H_{ext}$ [Oe]")
        ax.set_ylabel(ylabel)
        _set_reversed_hext_axis(ax, hext_range)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(
        os.path.join(folder, "summary_torques_vs_hext.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_torque_error_summary(
    general_title_summary,
    save_path_summary,
    hext_range,
    torque_error_history,
):
    """Plot causal same-state and accumulated demag torque errors."""
    folder = _output_folder(save_path_summary)
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), sharex=True)
    fig.suptitle(
        "Demagnetizing-Field Torque Error Across the M-H Sweep\n\n"
        + general_title_summary,
        fontsize=13,
        fontweight="bold",
    )

    panels = (
        (
            "same_state_hd_vector_error_mean",
            "Same-State UNet Demag-Field Error",
            r"Mean $|H_d^{UNet}(m_{FFT})-H_d^{FFT}(m_{FFT})|$ [Oe]",
        ),
        (
            "same_state_torque_error_mean",
            "Same-State Torque-Producing Error",
            r"Mean $|m_{FFT}\times\Delta H_d|$ [Oe]",
        ),
        (
            "trajectory_demag_torque_mismatch_mean",
            "Full-Trajectory Demag Torque Mismatch",
            r"Mean $|m_U\times H_{d,U}-m_F\times H_{d,F}|$ [Oe]",
        ),
        (
            "trajectory_demag_llg_drive_mismatch_mean",
            "Full-Trajectory Demag LLG-Drive Mismatch",
            r"Mean demag-drive mismatch [Oe]",
        ),
    )

    for ax, (key, title, ylabel) in zip(axes.flat, panels):
        ax.plot(hext_range, torque_error_history[key], lw=2.3, label=title)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel(r"External Field $H_{ext}$ [Oe]")
        ax.set_ylabel(ylabel)
        _set_reversed_hext_axis(ax, hext_range)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(
        os.path.join(folder, "demag_torque_error_summary_vs_hext.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_loop_change_error_overlays(
    general_title_summary,
    save_path_summary,
    hext_range,
    error_histories: Mapping[str, Sequence[float]],
    loop_change_fft: Mapping[str, Sequence[float]],
    loop_change_unet: Mapping[str, Sequence[float]],
):
    """Create exact four-panel rate/error overlays for all retained signals."""
    folder = _output_folder(save_path_summary)

    error_panels = (
        ("hd", r"$H_{demag}$ Error", r"$H_{demag}$ MAE [Oe]"),
        ("spin", "Magnetization Trajectory Error", "Magnetization MAE"),
        ("he", r"$H_{ex}$ Error", r"$H_{ex}$ MAE [Oe]"),
        ("ha", r"$H_{anis}$ Error", r"$H_{anis}$ MAE [Oe]"),
    )

    signal_specs = (
        (
            "gradient_loop_abs_change_per_oe",
            "Gradient-Magnitude Rate Over Error Curves",
            r"Mean $|\Delta|\nabla m||/|\Delta H_{ext}|$ [cell$^{-1}$/Oe]",
            "gradient_rate_per_oe_over_errors.png",
        ),
        (
            "gradient_tensor_loop_abs_change_per_oe",
            "Full Gradient-Tensor Rate Over Error Curves",
            r"Mean $||\nabla(m_i-m_{i-1})||_F/|\Delta H_{ext}|$ "
            r"[cell$^{-1}$/Oe]",
            "gradient_tensor_rate_per_oe_over_errors.png",
        ),
        (
            "exchange_energy_density_loop_abs_change_per_oe",
            "Exchange-Energy-Density Rate Over Error Curves",
            r"Mean $|\Delta\epsilon_{ex}|/|\Delta H_{ext}|$ "
            r"[erg cm$^{-3}$/Oe]",
            "exchange_energy_density_rate_per_oe_over_errors.png",
        ),
        (
            "exchange_field_vector_loop_abs_change_per_oe",
            "Exchange-Field Vector Rate Over Error Curves",
            r"Mean $||\Delta H_{ex}||_2/|\Delta H_{ext}|$ [Oe/Oe]",
            "exchange_field_vector_rate_per_oe_over_errors.png",
        ),
        (
            "exchange_torque_vector_loop_abs_change_per_oe",
            "Exchange-Torque Vector Rate Over Error Curves",
            r"Mean $||\Delta(m\times H_{ex})||_2/|\Delta H_{ext}|$ "
            r"[Oe/Oe]",
            "exchange_torque_vector_rate_per_oe_over_errors.png",
        ),
        (
            "demag_torque_vector_loop_abs_change_per_oe",
            "Demagnetizing-Torque Vector Rate Over Error Curves",
            r"Mean $||\Delta(m\times H_d)||_2/|\Delta H_{ext}|$ "
            r"[Oe/Oe]",
            "demag_torque_vector_rate_per_oe_over_errors.png",
        ),
        (
            "total_llg_drive_vector_loop_abs_change_per_oe",
            "Total Reduced-LLG-Drive Rate Over Error Curves",
            r"Mean $||\Delta F_{LLG}||_2/|\Delta H_{ext}|$ [Oe/Oe]",
            "total_llg_drive_vector_rate_per_oe_over_errors.png",
        ),
        (
            "winding_map_loop_abs_change_per_oe",
            "Winding-Density Rate Over Error Curves",
            r"Mean $|\Delta w|/|\Delta H_{ext}|$ [Oe$^{-1}$]",
            "winding_density_rate_per_oe_over_errors.png",
        ),
    )

    for signal_key, figure_title, signal_ylabel, filename in signal_specs:
        fig, axes = plt.subplots(2, 2, figsize=(16, 11), sharex=True)
        fig.suptitle(
            figure_title + "\n\n" + general_title_summary,
            fontsize=13,
            fontweight="bold",
        )

        for ax, (error_key, panel_title, error_ylabel) in zip(
            axes.flat, error_panels
        ):
            error_values = np.asarray(error_histories[error_key], dtype=float)
            fft_values = np.asarray(loop_change_fft[signal_key], dtype=float)
            unet_values = np.asarray(loop_change_unet[signal_key], dtype=float)

            ax.plot(
                hext_range,
                error_values,
                linewidth=2.1,
                color="black",
                label=panel_title,
            )
            ax.set_title(panel_title, fontsize=11, fontweight="bold")
            ax.set_xlabel(r"External Field $H_{ext}$ [Oe]")
            ax.set_ylabel(error_ylabel)
            ax.grid(True, linestyle="--", alpha=0.35)
            _set_reversed_hext_axis(ax, hext_range)

            # MAE values cannot be negative. Matplotlib otherwise places
            # artificial negative ticks around an all-zero history, such as
            # H_anis error when Ku = 0.
            finite_error = error_values[np.isfinite(error_values)]
            if finite_error.size:
                error_max = float(np.max(finite_error))
                if error_max > 0.0:
                    ax.set_ylim(0.0, 1.05 * error_max)
                else:
                    ax.set_ylim(0.0, 1.0)

            twin = ax.twinx()
            twin.plot(
                hext_range,
                fft_values,
                linewidth=2.0,
                color="tab:blue",
                label="FFT rate",
            )
            twin.plot(
                hext_range,
                unet_values,
                linewidth=2.0,
                linestyle="--",
                color="tab:red",
                label="UNet rate",
            )
            twin.set_ylabel(signal_ylabel)
            _combined_legend(ax, twin)

        fig.tight_layout()
        fig.savefig(
            os.path.join(folder, filename),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)



