# -*- coding: utf-8 -*-
"""
Created on Thurs July 09 10:30:00 2026
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.colors as colors
import argparse
import torch
import seaborn as sns
import time
from scipy.stats import linregress, pearsonr
import csv
import pandas as pd
from scipy.stats import pearsonr

from libs.misc import Culist, MaskTp, spin_prepare, winding_density
import libs.MAG2305 as MAG2305
from libs.Unet import UNet

# ============================================================================
# M-H TEXTURE DIAGNOSTICS
# ============================================================================

def _layer0_training_tensor(spin):
    """Return layer 0 as ``(batch, 3, Nx, Ny)`` without changing values.

    The gradient and winding losses were trained on the first three input
    channels, which correspond to ``(m_x, m_y, m_z)`` of layer 0.
    """
    tensor = spin if isinstance(spin, torch.Tensor) else torch.as_tensor(spin)

    if tensor.ndim != 4:
        raise ValueError(
            "spin must have shape (Nx, Ny, Nz, 3) or (batch, channels, Nx, Ny); "
            f"received {tuple(tensor.shape)}."
        )

    # Already channel-first: (batch, channels, Nx, Ny).
    if tensor.shape[1] >= 3 and tensor.shape[-1] != 3:
        return tensor[:, :3]

    # MAG2305 layout: (Nx, Ny, Nz, 3).
    if tensor.shape[-1] == 3:
        if tensor.shape[2] < 1:
            raise ValueError("The spin tensor contains no magnetic layers.")
        return tensor[:, :, 0, :].permute(2, 0, 1).unsqueeze(0)

    # Covers channel-first grids whose Ny happens to equal 3.
    if tensor.shape[1] >= 3:
        return tensor[:, :3]

    raise ValueError(
        "Could not identify the three magnetization channels in spin with "
        f"shape {tuple(tensor.shape)}."
    )


def _training_gradient_map(spin_channel_first):
    """Reproduce the exact finite-difference map used by ``train.py``.

    This intentionally uses grid-cell units (dx = dy = 1), centered
    differences, and replicated values only at the outer rectangular array
    boundary. It does not mask sample-shape boundaries before differentiating.
    """
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


def _strict_interior_mask(active_mask):
    """Select magnetic cells whose four in-plane neighbors are magnetic.

    The rectangular array edge is always excluded. For masked samples, this
    also removes the one-cell-thick geometric boundary around holes, triangles,
    convex hulls, and other nonmagnetic regions.
    """
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


def _training_winding_map(spin_channel_first):
    """Reproduce the local winding-density formula used during training."""
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

    for plus, minus, original, axis in (
        (mx_xp, mx_xm, mx, "x"),
        (my_xp, my_xm, my, "x"),
    ):
        plus[:, -1, :] = original[:, -1, :]
        minus[:, 0, :] = original[:, 0, :]

    for plus, minus, original, axis in (
        (mx_yp, mx_ym, mx, "y"),
        (my_yp, my_ym, my, "y"),
    ):
        plus[:, :, -1] = original[:, :, -1]
        minus[:, :, 0] = original[:, :, 0]

    dmx_dx = (mx_xp - mx_xm) / 2.0
    dmx_dy = (mx_yp - mx_ym) / 2.0
    dmy_dx = (my_xp - my_xm) / 2.0
    dmy_dy = (my_yp - my_ym) / 2.0
    return (dmx_dx * dmy_dy - dmy_dx * dmx_dy) / np.pi


def _selected_mean(values, mask):
    selected = values[mask]
    return float(selected.mean().item()) if selected.numel() else float("nan")


def _selected_max(values, mask):
    selected = values[mask]
    return float(selected.max().item()) if selected.numel() else float("nan")


@torch.no_grad()
def compute_training_texture_metrics(spin, exchange_energy=None, active_threshold=1.0e-12):
    """Compute the scalar histories used by the new M-H summary plots.

    Parameters
    ----------
    spin:
        MAG2305 spin tensor ``(Nx, Ny, Nz, 3)`` or a channel-first training
        tensor ``(batch, channels, Nx, Ny)``.
    exchange_energy:
        Optional MAG2305 ``Energy_excha`` value. Because MAG2305 sums a
        per-cell cgs exchange-energy density, dividing by the number of active
        cells gives the mean exchange-energy density in erg/cm^3.

    Returns
    -------
    dict
        Exact training-map summaries, boundary-controlled gradient summaries,
        winding-density summaries, and exchange-energy summaries.
    """
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
        if isinstance(exchange_energy, torch.Tensor):
            exchange_value = float(exchange_energy.detach().cpu().item())
        else:
            exchange_value = float(exchange_energy)
        exchange_density = exchange_value / full_active_count

    return {
        # Exact map used by the original gradient-weighted training loss.
        "gradient_training_grid_mean": float(gradient.mean().item()),
        # Same map, but average only over magnetic center cells.
        "gradient_active_mean": _selected_mean(gradient, active),
        # Same map and stencil, restricted to cells with four magnetic neighbors.
        "gradient_interior_mean": _selected_mean(gradient, interior),
        # Exact proxy used by loss_type == "exchange_energy" in train.py.
        "exchange_proxy_training_mean": float(gradient.square().mean().item()),
        # Local winding density used by loss_type == "winding" in train.py.
        "winding_training_abs_mean": _selected_mean(winding_abs, active),
        "winding_training_abs_max": _selected_max(winding_abs, active),
        # MAG2305 physical exchange-energy density averaged over all active layers.
        "exchange_energy_density": exchange_density,
        "active_cell_count": full_active_count,
        "interior_layer0_cell_count": int(interior.sum().item()),
    }


def _set_reversed_hext_axis(ax, Hext_range):
    values = np.asarray(Hext_range, dtype=float)
    max_h = float(np.nanmax(values))
    min_h = float(np.nanmin(values))
    pad = 0.05 * (max_h - min_h if max_h != min_h else 1.0)
    ax.set_xlim(max_h + pad, min_h - pad)


def plot_magnetization_gradient_vs_hext(
    general_title_summary,
    save_path_summary,
    Hext_range,
    gradient_fft,
    gradient_unet,
):
    """Plot three boundary treatments of the training gradient map vs Hext."""
    folder = os.path.join(save_path_summary, "summary_plots")
    os.makedirs(folder, exist_ok=True)

    fig, axs = plt.subplots(1, 3, figsize=(19, 6.2), sharex=True)
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
            "Removes empty centers but retains magnetic boundary cells.",
        ),
        (
            "gradient_interior_mean",
            "Strict Magnetic Interior",
            r"Mean $|\nabla m|$ [cell$^{-1}$]",
            "Uses only cells with magnetic ±x and ±y neighbors.",
        ),
    )

    for ax, (key, title, ylabel, subtitle) in zip(axs, panels):
        ax.plot(Hext_range, gradient_fft[key], lw=2.3, label="FFT/LLG")
        ax.plot(Hext_range, gradient_unet[key], lw=2.3, label="UNet/LLG")
        ax.set_title(title + "\n" + subtitle, fontsize=10.5, fontweight="bold")
        ax.set_xlabel(r"External Field $H_{ext}$ [Oe]")
        ax.set_ylabel(ylabel)
        _set_reversed_hext_axis(ax, Hext_range)
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
    Hext_range,
    winding_fft,
    winding_unet,
):
    """Plot scalar summaries of the exact local winding map used in training."""
    folder = os.path.join(save_path_summary, "summary_plots")
    os.makedirs(folder, exist_ok=True)

    fig, axs = plt.subplots(1, 2, figsize=(14.5, 6.2), sharex=True)
    fig.suptitle(
        "Training Winding-Density Signal Across the M-H Sweep\n\n"
        + general_title_summary,
        fontsize=13,
        fontweight="bold",
    )

    panels = (
        (
            "winding_training_abs_mean",
            r"Mean Local Training Weight Signal $\langle |w| \rangle$",
            r"Mean $|w|$ on magnetic cells",
        ),
        (
            "winding_training_abs_max",
            r"Peak Local Training Weight Signal $\max |w|$",
            r"Maximum $|w|$ on magnetic cells",
        ),
    )

    for ax, (key, title, ylabel) in zip(axs, panels):
        ax.plot(Hext_range, winding_fft[key], lw=2.3, label="FFT/LLG")
        ax.plot(Hext_range, winding_unet[key], lw=2.3, label="UNet/LLG")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel(r"External Field $H_{ext}$ [Oe]")
        ax.set_ylabel(ylabel)
        _set_reversed_hext_axis(ax, Hext_range)
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
    Hext_range,
    exchange_fft,
    exchange_unet,
):
    """Compare the training exchange proxy with MAG2305 exchange density."""
    folder = os.path.join(save_path_summary, "summary_plots")
    os.makedirs(folder, exist_ok=True)

    fig, axs = plt.subplots(1, 2, figsize=(14.5, 6.2), sharex=True)
    fig.suptitle(
        "Exchange-Energy Density Across the M-H Sweep\n\n"
        + general_title_summary,
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

    for ax, (key, title, ylabel) in zip(axs, panels):
        ax.plot(Hext_range, exchange_fft[key], lw=2.3, label="FFT/LLG")
        ax.plot(Hext_range, exchange_unet[key], lw=2.3, label="UNet/LLG")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel(r"External Field $H_{ext}$ [Oe]")
        ax.set_ylabel(ylabel)
        _set_reversed_hext_axis(ax, Hext_range)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(
        os.path.join(folder, "exchange_energy_density_vs_hext.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

# =======================================================================
# SUMMARY PLOTS (FINAL ENERGY PROFILES, PERFORMANCE METRICS, ETC.) 
# =======================================================================

def plot_full_energy_summary(general_title_summary, save_path_summary,full_data_fft, full_data_un, Hext_range):
    """
    Generates a final 2x2 multi-panel graph charting equilibrium energy components 
    across the entire completed external field sweep loop range.
    """
    full_energy_folder = os.path.join(save_path_summary, "summary_plots")
    os.makedirs(full_energy_folder, exist_ok=True)
    
    fig, axs = plt.subplots(2, 2, figsize=(15, 11))
  
    fig.suptitle("Full MH Curve Energy Summary\n\n"+ general_title_summary, fontsize=13, fontweight="bold")
    
    # Calculate uniform X-axis padding based on external field bounds
    max_h, min_h = max(Hext_range), min(Hext_range)
    h_range = max_h - min_h
    xmax_padded = max_h + (h_range * 0.05)
    xmin_padded = min_h - (h_range * 0.05)
    
    plot_map = [('demag', 'Equilibrium Demagnetizing Energy ($E_{demag}$)', axs[0, 0]),
                ('anis', 'Equilibrium Anisotropy Energy ($E_{anis}$)', axs[0, 1]),
                ('excha', 'Equilibrium Exchange Energy ($E_{excha}$)', axs[1, 0]),
                ('exter', 'Equilibrium exter Energy ($E_{exter}$)', axs[1, 1])]
    
    for key, panel_title, ax in plot_map:
        # Plot full profiles against the external field tracking array
        ax.plot(Hext_range, full_data_fft[key], color='blue', lw=2, linestyle='-', label='FFT Engine Profile')
        ax.plot(Hext_range, full_data_un[key], color='red', lw=2, linestyle='-', label='UNet Model Profile')
        ax.set_title(panel_title, fontsize=11, fontweight='bold')
        ax.set_xlabel('External Field $H_{ext}$ [Oe]', fontsize=10)
        ax.set_ylabel('Energy [Joules]', fontsize=10)
        
        combined_vals = list(full_data_fft[key]) + list(full_data_un[key])
        if len(combined_vals) > 0:
            max_v, min_v = max(combined_vals), min(combined_vals)
            v_range = max_v - min_v if max_v != min_v else 1.0
            ymax = max_v + (v_range * 0.05)
            ymin = -0.05 * max_v if min_v == 0.0 and max_v != 0.0 else min_v - (v_range * 0.05)
            ax.set_ylim(ymin, ymax)
            
        ax.set_xlim(xmax_padded, xmin_padded) # Keeps standard reversing sweep profile view orientation
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.legend(loc='upper right', fontsize=9)
        
    plt.tight_layout()
    plt.savefig(os.path.join(full_energy_folder, 'full_equilibrium_energy_summary.png'), dpi=200)
    plt.close()

    fig_tot, ax_tot = plt.subplots(figsize=(9, 6))
    fig_tot.suptitle(f"Total System Energy Profile Across M-H Sweep\n{general_title_summary}", fontsize=11, fontweight='bold')
    
    ax_tot.plot(Hext_range, full_data_fft['total'], color='blue', lw=2.5, linestyle='-', label='FFT Engine Profile')
    ax_tot.plot(Hext_range, full_data_un['total'], color='red', lw=2.5, linestyle='-', label='UNet Model Profile')
    ax_tot.set_title('Equilibrium Total System Energy ($E_{total}$)', fontsize=12, fontweight='bold')
    ax_tot.set_xlabel('External Field $H_{ext}$ [Oe]', fontsize=11)
    ax_tot.set_ylabel('Total Energy [Joules]', fontsize=11) 
    
    combined_tot = list(full_data_fft['total']) + list(full_data_un['total'])
    if len(combined_tot) > 0:
        max_v, min_v = max(combined_tot), min(combined_tot)
        v_range = max_v - min_v if max_v != min_v else 1.0
        ax_tot.set_ylim(min_v - (v_range * 0.05), max_v + (v_range * 0.05))
        
    ax_tot.set_xlim(xmax_padded, xmin_padded)
    ax_tot.grid(True, linestyle='--', alpha=0.4)
    ax_tot.legend(loc='upper right', fontsize=10)
    
    plt.tight_layout()
    fig_tot.subplots_adjust(top=0.85)
    plt.savefig(os.path.join(full_energy_folder, 'macro_total_energy_summary.png'), dpi=200)
    plt.close()

def plot_performance_summary(general_title_summary, save_path_summary, performance_fft, performance_un, Hext_range):
    """
    Generates a final 2x2 multi-panel chart compiling global optimization metrics,
    topological structures, and execution times across the full Hext range.
    """
    performance_folder = os.path.join(save_path_summary, "summary_plots")
    os.makedirs(performance_folder, exist_ok=True)
    
    fig, axs = plt.subplots(2, 2, figsize=(15, 11))

    fig.suptitle("Performance Summary\n\n" + general_title_summary, fontsize=13, fontweight='bold')
    
    # Calculate uniform X-axis bounds with  5% padding
    max_h, min_h = max(Hext_range), min(Hext_range)
    h_range = max_h - min_h
    xmax_padded = max_h + (h_range * 0.05)
    xmin_padded = min_h - (h_range * 0.05)
    
    plot_map = [('iters', 'Solver Iterations Per Loop', 'Total Iteration Count/Hext Step', axs[0, 0]),
                ('vortices', 'Topological Vortex Count', 'Absolute Vortex Population Count', axs[0, 1]),
                ('mz', 'Mean Out-of-Plane Magnetization ($|M_z|$)', 'Average Absolute Magnitude $|M_z|$', axs[1, 0]),
                ('time', 'Real-World Total Execution Time', 'Compute Duration [Seconds]', axs[1, 1])]
    
    for key, panel_title, y_label, ax in plot_map:
        ax.plot(Hext_range, performance_fft[key], color='blue', lw=2, linestyle='-', label='FFT Engine Profile')
        ax.plot(Hext_range, performance_un[key], color='red', lw=2, linestyle='-', label='UNet Model Profile')
        ax.set_title(panel_title, fontsize=11, fontweight='bold')
        ax.set_xlabel('External Field $H_{ext}$ [Oe]', fontsize=10)
        ax.set_ylabel(y_label, fontsize=10)
        
        combined_vals = list(performance_fft[key]) + list(performance_un[key])
        if len(combined_vals) > 0:
            max_v, min_v = max(combined_vals), min(combined_vals)
            v_range = max_v - min_v if max_v != min_v else 1.0
            ymax = max_v + (v_range * 0.05)
            
            # check for fields like vortex counts or Mz that sit flat at 0.0
            ymin = -0.05 * max_v if min_v == 0.0 and max_v != 0.0 else min_v - (v_range * 0.05)
            ax.set_ylim(ymin, ymax)
            
        ax.set_xlim(xmax_padded, xmin_padded) # Keeps standard reversing sweep profile view orientation
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.legend(loc='upper right', fontsize=9)
        
    plt.tight_layout()
    plt.savefig(os.path.join(performance_folder, 'performance_summary.png'), dpi=200)
    plt.close()

def plot_error_summary(general_title_summary, save_path_summary, Hext_range, inst_hd_mae, traj_shift_mae, hex_err_mae, hanis_err_mae, y_limits=None):
    """
    Generates a final 2x2 multi-panel master report compiling all local field approximations,
    historical path tracking drift, and intrinsic field deviations across the Hext sweep.
    """
    error_summary_folder = os.path.join(save_path_summary, "summary_plots")
    os.makedirs(error_summary_folder, exist_ok=True)
    
    print("Generating comprehensive 4-panel error tracking analysis...")
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    
    default_y_limits = {"hd": (0.0, 400.0),
                        "trajectory": (0.0, 0.75),
                        "exchange": (0.0, 400.0),
                        "anisotropy": (0.0, 400.0),}
    
    fixed_y_limits = default_y_limits.copy()
    if y_limits is not None:
        unknown_keys = set(y_limits) - set(default_y_limits)
        if unknown_keys:
            raise ValueError(
                "Unknown plot_error_summary y-limit key(s): "
                + ", ".join(sorted(unknown_keys))
            )
        fixed_y_limits.update(y_limits)

    for name, limits in fixed_y_limits.items():
        if len(limits) != 2 or limits[0] >= limits[1]:
            raise ValueError(
                f"Invalid y-axis limits for {name}: {limits}. "
                "Expected (minimum, maximum) with minimum < maximum."
            )

    fig.suptitle("Field Component Error Summary\n\n" + general_title_summary, fontsize=13, fontweight='bold')
    
    # Calculate uniform X-axis limits with standard 5% padding while maintaining the reversed sweep
    max_h, min_h = max(Hext_range), min(Hext_range)
    h_range = max_h - min_h if max_h != min_h else 1.0
    xmax_padded = max_h + (h_range * 0.05)
    xmin_padded = min_h - (h_range * 0.05)
    
    # Structural Mapping Matrix to cycle configurations cleanly
    plot_map = [(inst_hd_mae, 'darkorange', 'Total Unet Model $H_{demag}$ Approximation Error', '$H_{demag}$ Field Prediction Error', 'Instantaneous $H_{demag}$ MAE [Oe]', axs[0, 0], 'hd'),
                (traj_shift_mae, 'crimson', 'Magnetization Trajectory Drift (Accumulated Error)', 'Predicted Magnetization Error', 'Cumulative Spin $\\vec{m}$ MAE', axs[0, 1], 'trajectory'),
                (hex_err_mae, 'purple', 'Total Exchange Field ($H_{ex}$) Error Accumulation', '$H_{ex}$ Prediction Error', 'Exchange Field MAE [Oe]', axs[1, 0], 'exchange'),
                (hanis_err_mae, 'teal', 'Total Anisotropy Field ($H_{anis}$) Error Accumulation', '$H_{anis}$ Prediction Error', 'Anisotropy Field MAE [Oe]', axs[1, 1], 'anisotropy')]
    
    for data, color, subtitle, label, y_label, ax, limit_key in plot_map:
        data_array = np.asarray(data, dtype=float)
        ymin, ymax = fixed_y_limits[limit_key]

        ax.plot(Hext_range, data_array, color=color, lw=2, linestyle="-", label=label)
        ax.set_title(subtitle, fontsize=11, fontweight="bold")
        ax.set_xlabel("External Magnetic Field $H_{ext}$ [Oe]", fontsize=10)
        ax.set_ylabel(y_label, fontsize=10)
        ax.set_xlim(xmax_padded, xmin_padded)
        ax.set_ylim(ymin, ymax)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc="upper right", fontsize=9)

        # Fixed limits can clip an unusually large run. Warn rather than
        # silently hiding that fact.
        finite_data = data_array[np.isfinite(data_array)]
        if finite_data.size:
            observed_min = float(np.min(finite_data))
            observed_max = float(np.max(finite_data))
            if observed_min < ymin or observed_max > ymax:
                print(
                    f"[plot_error_summary] WARNING: {limit_key} data range "
                    f"({observed_min:.6g}, {observed_max:.6g}) exceeds the "
                    f"fixed y-axis range ({ymin:.6g}, {ymax:.6g})."
                )

    plt.tight_layout()
    plt.savefig(
        os.path.join(error_summary_folder, "comprehensive_error_analysis.png"),
        dpi=300,
    )
    plt.close()

def plot_fields_summary(general_title_summary, save_path_summary, Hext_range, hex_mm, hex_un, hanis_mm, hanis_un, hd_mm, hd_un, heff_mm, heff_un):
    """
    Generates a 2x2 panel graph chart recording equilibrium 
    magnitudes of all internal fields across the completed Hext sweep range.
    """
    fields_summary_folder = os.path.join(save_path_summary, "summary_plots")
    os.makedirs(fields_summary_folder, exist_ok=True)
    
    print("Generating final 4-panel physical field summary plot...")
    fig, axs = plt.subplots(2, 2, figsize=(15, 11))

    fig.suptitle("Field Component Over Full MH Curve Summary\n\n" + general_title_summary, fontsize=13, fontweight='bold')
    
    max_h, min_h = max(Hext_range), min(Hext_range)
    h_range = max_h - min_h if max_h != min_h else 1.0
    xmax_padded = max_h + (h_range * 0.05)
    xmin_padded = min_h - (h_range * 0.05)
    
    plot_map = [(hex_mm, hex_un, 'Exchange Field ($H_{ex}$)', 'Mean $H_{ex}$ Magnitude [Oe]', axs[0, 0]),
                (hanis_mm, hanis_un, 'Anisotropy Field ($H_{anis}$)', 'Mean $H_{anis}$ Magnitude [Oe]', axs[0, 1]),
                (hd_mm, hd_un, 'Demagnetizing Field ($H_{demag}$)', 'Mean $H_{demag}$ Magnitude [Oe]', axs[1, 0]),
                (heff_mm, heff_un, 'Total Effective Field ($H_{eff}$)', 'Mean $H_{eff}$ Magnitude [Oe]', axs[1, 1])]
    
    for data_mm, data_un, panel_title, y_label, ax in plot_map:
        ax.plot(Hext_range, data_mm, color='blue', lw=2.5, linestyle='-', label='FFT Simulator (mm)')
        ax.plot(Hext_range, data_un, color='red', lw=2.5, linestyle='-', label='UNet Model (un)')
        
        ax.set_title(panel_title, fontsize=11, fontweight='bold')
        ax.set_xlabel('External Field $H_{ext}$ [Oe] Summary', fontsize=10)
        ax.set_ylabel(y_label, fontsize=10)
        
        combined_vals = list(data_mm) + list(data_un)
        if len(combined_vals) > 0:
            max_v, min_v = max(combined_vals), min(combined_vals)
            v_range = max_v - min_v if max_v != min_v else 1.0
            ymax = max_v + (v_range * 0.05)
            
            # check for fields like Anisotropy that are set to at 0.0
            ymin = -0.05 * max_v if min_v == 0.0 and max_v != 0.0 else min_v - (v_range * 0.05)
            ax.set_ylim(ymin, ymax)
            
        ax.set_xlim(xmax_padded, xmin_padded) # Reverses axis to match physical sweep direction
        ax.grid(True, linestyle='-.', alpha=0.5)
        ax.legend(loc='upper right', fontsize=9)
        
    plt.tight_layout()
    plt.savefig(os.path.join(fields_summary_folder, 'all_internal_fields_mh_sweep.png'), dpi=300)
    plt.close()

def plot_error_correlations(general_title_summary, save_path_summary,hd_error, hex_error, hanis_error, traj_error, Hext_range):
    """
    Generates scatter plots comparing each internal field error to the
    trajectory error over the entire hysteresis sweep.
    """
    fields_summary_folder = os.path.join(save_path_summary, "summary_plots")
    os.makedirs(fields_summary_folder, exist_ok=True)

    fig, axs = plt.subplots(1, 3, figsize=(16, 5.5), sharey=True, constrained_layout=True)

    datasets = [(hd_error, "Demagnetizing Field $H_{demag}$ [Oe] Error"), 
                (hex_error, "Exchange Field $H_{ex}$ [Oe] Error"), 
                (hanis_error, "Anisotropy Field $H_{anis}$ [Oe] Error")]

    for ax, (x, xlabel) in zip(axs, datasets):
        x = np.asarray(x)
        y = np.asarray(traj_error)                 #TODO: check after if values need ax.set_xscale("log") ax.set_yscale("log")
        result = linregress(x, y)
        xx = np.linspace(x.min(), x.max(), 200)

        slope = result.slope
        intercept = result.intercept
        r = result.rvalue
        p = result.pvalue
        r2 = r**2

        if Hext_range is not None:
            sc = ax.scatter(x, y, s=25, alpha=0.85, c=Hext_range, cmap="coolwarm")
        else:
            sc = ax.scatter(x, y, s=25, alpha=0.75)
        
        ax.plot(xx, slope * xx + intercept, '--', linewidth=2, color='black')
        ax.set_xlabel(xlabel)
        ax.grid(alpha=0.3)

        textbox = (f"r = {r:.3f}\n"
                   f"$R^2$ = {r2:.3f}\n"
                   f"p = {p:.2e}")

        ax.text(0.04, 0.96, textbox, transform=ax.transAxes, va='top',fontsize=10, bbox=dict(facecolor='white', alpha=0.9))

    if Hext_range is not None:
        fig.colorbar(sc, ax=axs, label="$H_{ext}$ [Oe]", shrink=0.8)

    axs[0].set_ylabel("Trajectory Error")
    fig.suptitle("Correlation Between Internal Field Errors and Trajectory Error\n\n" + general_title_summary, fontsize=13, fontweight='bold')

    plt.savefig(os.path.join(fields_summary_folder, "error_correlations.png"), dpi=300, bbox_inches="tight")
    plt.close()


def plot_error_vs_transition_proximity(general_title_summary, save_path_summary,trajectory_error, vortex_count, max_window=15, event_type='both'):
    """
    Bin trajectory error by "frames since nearest topological event"
    (vortex nucleation or annihilation, detected as a change in vortex
    count between consecutive Hext steps) and plot the resulting decay/
    rise curve.
    """
    folder = os.path.join(save_path_summary, "summary_plots")
    os.makedirs(folder, exist_ok=True)
 
    error = np.asarray(trajectory_error, dtype=float)
    vortices = np.asarray(vortex_count, dtype=float)
    dv = np.diff(vortices)
 
    if event_type == 'nucleation':
        event_indices = np.where(dv > 0)[0] + 1
    elif event_type == 'annihilation':
        event_indices = np.where(dv < 0)[0] + 1
    else:
        event_indices = np.where(dv != 0)[0] + 1
 
    if len(event_indices) == 0:
        print("[plot_error_vs_transition_proximity] No topological events "
              "(vortex count changes) detected; skipping.")
        return None
 
    # Align a window of error values around every event, padding with NaN
    # at the sweep edges so events near the boundary don't bias the mean.
    n = len(error)
    aligned = np.full((len(event_indices), 2 * max_window + 1), np.nan)
    for row, ev in enumerate(event_indices):
        for offset in range(-max_window, max_window + 1):
            idx = ev + offset
            if 0 <= idx < n:
                aligned[row, offset + max_window] = error[idx]
 
    mean_curve = np.nanmean(aligned, axis=0)
    std_curve = np.nanstd(aligned, axis=0)
    x = np.arange(-max_window, max_window + 1)
 
    fig, ax = plt.subplots(figsize=(9, 6))
 
    fig.suptitle("Trajectory Error Aligned to Topological Events\n\n" + general_title_summary, fontsize=12, fontweight='bold')
 
    ax.plot(x, mean_curve, color='crimson', lw=2.5, label='Mean trajectory error')
    ax.fill_between(x, mean_curve - std_curve, mean_curve + std_curve, color='crimson', alpha=0.2, label='+/- 1 std')
    ax.axvline(0, color='black', linestyle='--', lw=1, alpha=0.7, label='Event (vortex count change)')
 
    ax.set_xlabel('Hext steps relative to event')
    ax.set_ylabel('Trajectory error (MAE)')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
 
    plt.tight_layout()
    plt.savefig(os.path.join(folder, f'error_vs_transition_proximity_{event_type}.png'), dpi=250)
    plt.close()
 
    return {'x': x, 'mean_curve': mean_curve, 'std_curve': std_curve,
            'n_events': len(event_indices), 'event_indices': event_indices}
 
 
def plot_ablation_comparison_table(general_title_summary, ablation_results, base_path):
    """
    Creates a summary table (as a saved figure + CSV) comparing peak error
    and total accumulated error across different model variants, e.g.
    full FFT vs. woHd ablation vs. UNet-Hd 
    """
    folder = os.path.join(base_path, "summary_plots")
    os.makedirs(folder, exist_ok=True)
 
    rows = []
    for name, data in ablation_results.items():
        err = np.asarray(data["trajectory_error"], dtype=float)
        rows.append({
            "Variant": name,
            "Peak Error": float(np.max(err)),
            "Mean Error": float(np.mean(err)),
            "Total Accumulated Error": float(np.sum(err)),
            "Std Error": float(np.std(err)),
        })
 
    table_df = pd.DataFrame(rows).sort_values("Total Accumulated Error").reset_index(drop=True)
 
    csv_path = os.path.join(folder, "ablation_comparison.csv")
    table_df.to_csv(csv_path, index=False)
 
    fig, ax = plt.subplots(figsize=(10, 1.2 + 0.5 * len(table_df)))
    ax.axis('off')
 
    ax.set_title("Model Variant Ablation Comparison\n" + general_title_summary, fontsize=12, fontweight='bold', pad=20)
 
    display_df = table_df.copy()
    for col in ["Peak Error", "Mean Error", "Total Accumulated Error", "Std Error"]:
        display_df[col] = display_df[col].map(lambda v: f"{v:.4e}")
 
    tbl = ax.table(cellText=display_df.values, colLabels=display_df.columns,
                   cellLoc='center', loc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.6)
 
    for col_idx in range(len(display_df.columns)):
        tbl[0, col_idx].set_facecolor('#4C72B0')
        tbl[0, col_idx].set_text_props(color='white', fontweight='bold')
 
    plt.tight_layout()
    plt.savefig(os.path.join(folder, "ablation_comparison_table.png"), dpi=200, bbox_inches='tight')
    plt.close()
 
    print("Ablation comparison saved.")
    print(csv_path)
 
    return table_df
 

def plot_hd_error_vs_vortex_cores(general_title_summary, save_path_summary,film1, film2, nloop, core_threshold=0.5):
    """
    Overlay the spatial Hd (demag field) error map with the locations of
    vortex cores, identified from winding density. This directly tests
    whether U-Net Hd error is spatially co-located with topological defects,
    rather than just temporally correlated with a scalar vortex count.
 
    core_threshold : float
        Minimum |winding density| (per-cell, already normalized by pi in
        misc.winding_density) to call a cell part of a vortex core. Cores
        are usually only a few cells wide with |winding density| close to
        its local extremum, so 0.3-0.6 is a reasonable starting point;
        tune by eye against a few spatial_topology_loop_*.png plots first.
    """
    folder = os.path.join(save_path_summary, "hd_error_vs_cores")
    os.makedirs(folder, exist_ok=True)
 
    Hd_mm = film1.Hd.detach().cpu().numpy()[:, :, 0, :]
    Hd_un = film2.Hd.detach().cpu().numpy()[:, :, 0, :]
    hd_error_map = np.linalg.norm(Hd_un - Hd_mm, axis=-1)
 
    spin_fft_tensor = film1.Spin.permute(3, 0, 1, 2)[:, :, :, 0].unsqueeze(0)
    topo_fft_raw, winding_abs_fft, _ = winding_density(spin_fft_tensor)
    topo_fft = topo_fft_raw.squeeze().detach().cpu().numpy()
 
    core_mask = np.abs(topo_fft) > core_threshold
    core_ys, core_xs = np.where(core_mask)
 
    fig, ax = plt.subplots(figsize=(8, 7))

    fig.suptitle("Demag Field Error vs. Vortex Core Locations\n\n" + general_title_summary, fontsize=12, fontweight='bold')
 
    im = ax.imshow(hd_error_map, cmap='hot', origin='lower')
    fig.colorbar(im, ax=ax, label='$|H_{demag,un} - H_{demag,mm}|$ [Oe]')
 
    if len(core_xs) > 0:
        ax.scatter(core_xs, core_ys, s=40, facecolors='none', edgecolors='cyan',
                   linewidths=1.5, label=f'Vortex core cells (n={len(core_xs)})')
        ax.legend(loc='upper right', fontsize=9)
 
    ax.set_xlabel('x [cell index]')
    ax.set_ylabel('y [cell index]')
 
    plt.tight_layout()
    plt.savefig(os.path.join(folder, f'hd_error_vs_cores_loop_{nloop}.png'), dpi=150)
    plt.close()
 
    # Return a simple quantitative co-localization metric: mean Hd error
    # inside vs. outside the vortex-core mask. If error is concentrated at
    # cores, mean_error_at_cores should be substantially larger.
    if core_mask.sum() > 0:
        mean_error_at_cores = hd_error_map[core_mask].mean()
        mean_error_elsewhere = hd_error_map[~core_mask].mean() if (~core_mask).sum() > 0 else np.nan
    else:
        mean_error_at_cores = np.nan
        mean_error_elsewhere = hd_error_map.mean()
 
    return {'mean_error_at_cores': float(mean_error_at_cores),
            'mean_error_elsewhere': float(mean_error_elsewhere),
            'n_core_cells': int(core_mask.sum())}

def plot_colocalization_summary(general_title_summary, save_path_summary, Hext_range, coloc_rcd):
    """
    Sweep-level view of plot_hd_error_vs_vortex_cores: mean Hd error at vortex
    cores vs. elsewhere, as a function of Hext, to see whether spatial
    co-localization strengthens near switching fields.
    """
    folder = os.path.join(save_path_summary, "summary_plots")
    os.makedirs(folder, exist_ok=True)

    at_cores = np.array([r['mean_error_at_cores'] for r in coloc_rcd])
    elsewhere = np.array([r['mean_error_elsewhere'] for r in coloc_rcd])
    n_cells = np.array([r['n_core_cells'] for r in coloc_rcd])

    fig, axs = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    axs[0].plot(Hext_range, at_cores, color='crimson', lw=2, label='Mean error at vortex cores')
    axs[0].plot(Hext_range, elsewhere, color='steelblue', lw=2, label='Mean error elsewhere')
    axs[0].set_ylabel('Mean $H_{demag}$ error [Oe]')
    axs[0].legend(fontsize=9)
    axs[0].grid(alpha=0.3)

    axs[1].plot(Hext_range, n_cells, color='black', lw=1.5)
    axs[1].set_ylabel('# vortex-core cells')
    axs[1].set_xlabel('$H_{ext}$ [Oe]')
    axs[1].grid(alpha=0.3)

    fig.suptitle("Hd Error / Vortex-Core Co-localization Across Sweep\n" + general_title_summary, fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(folder, "colocalization_summary.png"), dpi=250)
    plt.close()


def plot_temporal_variance_vs_error(general_title_summary, save_path_summary, Hext_range, temporal_var_rcd, trajectory_error):
    """
    Tests whether compute_temporal_hd_variance (a model-internal signal,
    no ground truth needed) tracks actual trajectory error -- i.e. whether
    it's a usable uncertainty proxy at inference time.
    """
    folder = os.path.join(save_path_summary, "summary_plots")
    os.makedirs(folder, exist_ok=True)

    Hext_range = np.asarray(Hext_range, dtype=float)
    var = np.asarray(temporal_var_rcd, dtype=float)
    err = np.asarray(trajectory_error, dtype=float)
    valid = ~np.isnan(var) & ~np.isnan(err)

    fig, axs = plt.subplots(1, 2, figsize=(13, 5.5))

    line1, = axs[0].plot(Hext_range, var, color='darkorange', lw=2, label='Temporal Hd variance (model-internal)')
    ax_twin = axs[0].twinx()
    line2, = ax_twin.plot(Hext_range, err, color='crimson', lw=1.5, alpha=0.6, label='Trajectory error')
    axs[0].set_xlabel('$H_{ext}$ [Oe]')
    axs[0].set_ylabel('Temporal Hd variance', color='darkorange')
    ax_twin.set_ylabel('Trajectory error', color='crimson')
    axs[0].tick_params(axis='y', labelcolor='darkorange')
    ax_twin.tick_params(axis='y', labelcolor='crimson')
    axs[0].grid(alpha=0.3)
    axs[0].legend(handles=[line1, line2], loc='upper right', fontsize=9)

    r = np.corrcoef(var[valid], err[valid])[0, 1] if valid.sum() > 1 else np.nan
    sc = axs[1].scatter(var[valid], err[valid], s=20, alpha=0.7, c=Hext_range[valid], cmap='coolwarm')
    axs[1].set_xlabel('Temporal Hd variance')
    axs[1].set_ylabel('Trajectory error')
    axs[1].text(0.05, 0.95, f"r = {r:.3f}", transform=axs[1].transAxes, va='top',
                bbox=dict(facecolor='white', alpha=0.9))
    axs[1].grid(alpha=0.3)
    fig.colorbar(sc, ax=axs[1], label='$H_{ext}$ [Oe]')

    fig.suptitle("Temporal Hd Variance as an Uncertainty Proxy\n" + general_title_summary, fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(folder, "temporal_variance_vs_error.png"), dpi=250)
    plt.close()

def compute_temporal_hd_variance(hd_history_buffer):
    """
    Given a short history of the U-Net's own Hd predictions at this cell,
    compute the per-cell variance across that history as a proxy for "the model
    itself is uncertain here". 

    Returns
    -------
    variance_map : ndarray, shape (W, W) or (W, W, D)
        Per-cell variance of |Hd| across the k history steps.
    mean_variance : float
        Spatial mean of variance_map -- use this as a scalar predictor,
        the same way instantaneous_hd_mae etc. are used in `predictors`.
    """
    hd_history_buffer = np.asarray(hd_history_buffer)
    hd_mag_history = np.linalg.norm(hd_history_buffer, axis=-1)  # (k, W, W[, D])
    variance_map = np.var(hd_mag_history, axis=0)
    mean_variance = float(np.mean(variance_map))
    return variance_map, mean_variance