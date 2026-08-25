# -*- coding: utf-8 -*-
"""
Streamlined NeuralMAG M-H evaluation.

- original one-figure-per-field-step diagnostics
- error, energy, field, performance, torque, and torque-error summaries
- magnetization-gradient, exchange-energy-density, and winding summaries
- gradient-magnitude, full gradient-tensor, and exchange rate-of-change summary
- exact gradient-magnitude, gradient-tensor, exchange-energy-density, and
  winding-rate overlays over all four error histories
- physics_snapshots.csv and the original compact .npy histories

"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import Normalize

from egs.NMI.MH_evaluate.searcher import PhysicsRecorder, analyze_winding_components
from libs.misc import Culist, MaskTp, spin_prepare, winding_density
import libs.MAG2305 as MAG2305
from libs.Unet import UNet
from plots import (
    compute_exact_loop_change_metrics,
    compute_gradient_spatial_variation_metrics,
    compute_training_texture_metrics,
    plot_error_summary,
    plot_exchange_energy_density_vs_hext,
    plot_fields_summary,
    plot_full_energy_summary,
    plot_gradient_change_rate_summary,
    plot_loop_change_error_overlays,
    plot_magnetization_gradient_vs_hext,
    plot_performance_summary,
    plot_physics_vector_rate_summary,
    plot_torque_error_summary,
    plot_torque_summary,
    plot_training_winding_density_vs_hext,
    plot_training_gradient_tensor_rate_vs_hext,
    plot_training_weight_error_overlays,
)


def load_unet_model(args: argparse.Namespace, device: torch.device) -> Path:
    model = UNet(
        kc=args.krn,
        inc=args.layers * 3,
        ouc=args.layers * 3,
    ).eval().to(device)

    checkpoint = Path("../ckpt") / f"k{args.krn}" / args.model_name
    if not checkpoint.is_file():
        raise FileNotFoundError(f"UNet checkpoint was not found: {checkpoint.resolve()}")

    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    MAG2305.load_model(model)
    print(f"UNet model loaded from {checkpoint}")
    return checkpoint.resolve()


def initialize_models(args: argparse.Namespace, device: torch.device):
    common = dict(
        types="bulk",
        size=(args.w, args.w, args.layers),
        cell=(args.cell_size, args.cell_size, args.cell_size),
        Ms=args.Ms,
        Ax=args.Ax,
        Ku=args.Ku,
        Kvec=args.Kvec,
        device=str(device),
    )

    film_fft = MAG2305.mmModel(**common)
    film_unet = MAG2305.mmModel(**common)
    print(f"Creating {args.layers} layer models")
    film_fft.DemagInit()
    print("FFT demagnetization matrix initialized")
    checkpoint = load_unet_model(args, device)
    return film_fft, film_unet, checkpoint


def prepare_spin_state(film_fft, film_unet, args: argparse.Namespace):
    spin = spin_prepare(args.spin_split, film_fft, args.rand_seed, mask=args.mask)
    film_fft.SpinInit(spin)
    film_unet.SpinInit(spin)
    cell_count = int((np.linalg.norm(spin, axis=-1) > 0).sum())
    return spin, cell_count


def update_spin_fft(model, hext: np.ndarray, args: argparse.Namespace):
    error = 1.0
    iteration = 0
    error_record = []

    while iteration < args.max_iter and error > args.error_min:
        error = model.SpinLLG_RK4(
            Hext=hext,
            dtime=args.dtime,
            damping=args.damping,
        )
        error_record.append(float(error))
        if error <= args.error_min or iteration % 1000 == 0:
            print(f"Iteration: {iteration}\nError_converge FFT: {error:.2e}")
        iteration += 1

    return np.asarray(error_record, dtype=float), iteration


def update_spin_unet(model, hext: np.ndarray, args: argparse.Namespace):
    error = 1.0
    iteration = 0
    error_record = []

    while iteration < args.max_iter and error > args.error_min:
        error = model.SpinLLG_RK4_unetHd(
            Hext=hext,
            dtime=args.dtime,
            damping=args.damping,
        )
        error_record.append(float(error))

        if (
            iteration > args.unet_stagnation_start
            and len(error_record) >= args.unet_stagnation_long_window
        ):
            long_mean = float(
                np.mean(error_record[-args.unet_stagnation_long_window :])
            )
            short_mean = float(
                np.mean(error_record[-args.unet_stagnation_short_window :])
            )
            fluctuation = abs(long_mean - short_mean) / max(abs(long_mean), 1.0e-30)
            if (
                fluctuation < args.unet_stagnation_fraction
                and error < args.unet_stagnation_error
            ):
                print("UNet convergence error has stagnated; ending this relaxation.")
                break

        if error <= args.error_min or iteration % 1000 == 0:
            print(f"Iteration: {iteration}\nError_converge UNet: {error:.2e}")
        iteration += 1

    return np.asarray(error_record, dtype=float), iteration


def _mean_field_magnitude(field: torch.Tensor, spin: torch.Tensor) -> float:
    active = torch.linalg.vector_norm(spin, dim=-1) > 1.0e-12
    magnitude = torch.linalg.vector_norm(field, dim=-1)
    selected = magnitude[active]
    return float(selected.mean().item()) if selected.numel() else float("nan")


def _mean_torque_magnitude(spin: torch.Tensor, field: torch.Tensor) -> float:
    active = torch.linalg.vector_norm(spin, dim=-1) > 1.0e-12
    torque = torch.linalg.vector_norm(torch.cross(spin, field, dim=-1), dim=-1)
    selected = torque[active]
    return float(selected.mean().item()) if selected.numel() else float("nan")


@torch.no_grad()
def _exchange_energy_density_map(model) -> torch.Tensor:
    """MAG2305 per-cell exchange-energy density, in erg/cm^3."""
    return -0.5 * model.Msmx * torch.sum(model.Spin * model.He, dim=-1)


def _exchange_density_loop_change(
    current_map: torch.Tensor,
    previous_map: Optional[torch.Tensor],
    current_active_mask: torch.Tensor,
    previous_active_mask: Optional[torch.Tensor],
    delta_hext_oe: Optional[float],
) -> Dict[str, float]:
    result = {
        "exchange_energy_density_loop_abs_change_mean": float("nan"),
        "exchange_energy_density_loop_abs_change_per_oe": float("nan"),
    }

    if previous_map is None or previous_active_mask is None:
        return result

    compare_mask = current_active_mask & previous_active_mask
    selected = torch.abs(current_map - previous_map)[compare_mask]
    if selected.numel() == 0:
        return result

    change = float(selected.mean().item())
    result["exchange_energy_density_loop_abs_change_mean"] = change

    if delta_hext_oe is not None and abs(float(delta_hext_oe)) > 0.0:
        result["exchange_energy_density_loop_abs_change_per_oe"] = (
            change / abs(float(delta_hext_oe))
        )

    return result


def _vector_map_loop_change(
    current_map: torch.Tensor,
    previous_map: Optional[torch.Tensor],
    current_active_mask: torch.Tensor,
    previous_active_mask: Optional[torch.Tensor],
    delta_hext_oe: Optional[float],
    metric_name: str,
) -> Dict[str, float]:
    """Exact loop-to-loop change of a vector-valued spatial map.

    The vector difference is taken cell by cell first, then its Euclidean
    magnitude is spatially averaged over cells active in both states.
    """
    mean_key = f"{metric_name}_loop_abs_change_mean"
    per_oe_key = f"{metric_name}_loop_abs_change_per_oe"
    result = {mean_key: float("nan"), per_oe_key: float("nan")}

    if previous_map is None or previous_active_mask is None:
        return result

    compare_mask = current_active_mask & previous_active_mask
    difference_magnitude = torch.linalg.vector_norm(
        current_map - previous_map,
        dim=-1,
    )
    selected = difference_magnitude[compare_mask]
    if selected.numel() == 0:
        return result

    change = float(selected.mean().item())
    result[mean_key] = change

    if delta_hext_oe is not None and abs(float(delta_hext_oe)) > 0.0:
        result[per_oe_key] = change / abs(float(delta_hext_oe))

    return result


@torch.no_grad()
def _physics_vector_maps(model, damping: float) -> Dict[str, torch.Tensor]:
    """Return vector maps used for exact loop-to-loop rate diagnostics.

    ``total_llg_drive_vector`` reproduces MAG2305's field-scale LLG drive
    before multiplication by the gyromagnetic factor and time step:

        damping * ((m x H_eff) x m) - (m x H_eff)

    ``model.Heff`` must already include the external field.
    """
    exchange_torque = torch.cross(model.Spin, model.He, dim=-1)
    demag_torque = torch.cross(model.Spin, model.Hd, dim=-1)
    effective_torque = torch.cross(model.Spin, model.Heff, dim=-1)
    damping_drive = torch.cross(effective_torque, model.Spin, dim=-1)
    total_llg_drive = float(damping) * damping_drive - effective_torque

    return {
        "exchange_field_vector": model.He,
        "exchange_torque_vector": exchange_torque,
        "demag_torque_vector": demag_torque,
        "total_llg_drive_vector": total_llg_drive,
    }


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
    selected = values[mask]
    return float(selected.mean().item()) if selected.numel() else float("nan")


@torch.no_grad()
def _compute_demag_torque_error_metrics(
    film_fft,
    film_unet,
    damping: float,
) -> Dict[str, float]:
    """Compute causal and accumulated demagnetizing torque errors.

    The same-state terms evaluate FFT and UNet Hdemag on the exact same FFT
    magnetization. The trajectory terms compare the already-diverged FFT and
    UNet trajectories.
    """
    fft_active = torch.linalg.vector_norm(film_fft.Spin, dim=-1) > 1.0e-12
    unet_active = torch.linalg.vector_norm(film_unet.Spin, dim=-1) > 1.0e-12
    trajectory_mask = fft_active & unet_active

    # This is the exact scaling used by MAG2305.GetHeff_unetHd().
    hd_unet_on_fft = MAG2305.MFNN(film_fft.Spin) * film_fft.Ms[0] / 1000
    delta_hd_same_state = hd_unet_on_fft - film_fft.Hd

    same_state_vector_error = torch.linalg.vector_norm(
        delta_hd_same_state,
        dim=-1,
    )
    same_state_torque_error = torch.linalg.vector_norm(
        torch.cross(film_fft.Spin, delta_hd_same_state, dim=-1),
        dim=-1,
    )

    fft_precession = torch.cross(film_fft.Spin, film_fft.Hd, dim=-1)
    unet_precession = torch.cross(film_unet.Spin, film_unet.Hd, dim=-1)
    trajectory_torque_mismatch = torch.linalg.vector_norm(
        unet_precession - fft_precession,
        dim=-1,
    )

    # MAG2305 demag-only LLG drive: damping * ((m x Hd) x m) - (m x Hd).
    fft_damping = torch.cross(fft_precession, film_fft.Spin, dim=-1)
    unet_damping = torch.cross(unet_precession, film_unet.Spin, dim=-1)
    fft_drive = float(damping) * fft_damping - fft_precession
    unet_drive = float(damping) * unet_damping - unet_precession
    trajectory_drive_mismatch = torch.linalg.vector_norm(
        unet_drive - fft_drive,
        dim=-1,
    )

    return {
        "same_state_hd_vector_error_mean": _masked_mean(
            same_state_vector_error, fft_active
        ),
        "same_state_torque_error_mean": _masked_mean(
            same_state_torque_error, fft_active
        ),
        "trajectory_demag_torque_mismatch_mean": _masked_mean(
            trajectory_torque_mismatch, trajectory_mask
        ),
        "trajectory_demag_llg_drive_mismatch_mean": _masked_mean(
            trajectory_drive_mismatch, trajectory_mask
        ),
    }


def plot_results(
    nloop,
    spin_mm,
    spin_un,
    itern1,
    itern2,
    hd_mm,
    hd_un,
    x_plot,
    y1_plot,
    y2_plot,
    hext_range,
    error1_rcd,
    error2_rcd,
    save_path_iteration,
    general_title_iteration,
):
    """Original one-figure-per-field-step diagnostic."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(general_title_iteration, fontsize=13, fontweight="bold")

    spin = (spin_mm + 1) / 2
    axes[0, 0].imshow(spin[:, :, 0, :].transpose(1, 0, 2), origin="lower")
    axes[0, 0].set_title(f"Spin FFT steps: [{itern1}]", fontsize=18)
    axes[0, 0].set_xlabel("x [cell]")
    axes[0, 0].set_ylabel("y [cell]")

    spin = (spin_un + 1) / 2
    axes[0, 1].imshow(spin[:, :, 0, :].transpose(1, 0, 2), origin="lower")
    axes[0, 1].set_title(f"Spin UNet steps: [{itern2}]", fontsize=18)
    axes[0, 1].set_xlabel("x [cell]")
    axes[0, 1].set_ylabel("y [cell]")

    spin_mse = np.square(spin_un - spin_mm).sum(axis=-1).transpose((1, 0, 2))[:, :, 0]
    image = axes[0, 2].imshow(spin_mse, cmap="hot", origin="lower")
    axes[0, 2].set_title(
        f"Spin FFT/UNet MSE\nMean: {spin_mse.mean():.4f}",
        fontsize=16,
    )
    fig.colorbar(image, ax=axes[0, 2])

    axes[0, 3].plot(x_plot, y1_plot, lw=1.5, label="FFT")
    axes[0, 3].plot(x_plot, y2_plot, lw=1.5, label="UNet")
    axes[0, 3].legend(fontsize=14, loc="upper left")
    axes[0, 3].set_title("M-H data", fontsize=16)
    axes[0, 3].set_xlabel(r"$H_{ext}$ [Oe]", fontsize=16)
    axes[0, 3].set_ylabel(
        r"Reduced longitudinal magnetization $m_{\parallel}=M_{\parallel}/M_s$",
        fontsize=14,
    )
    axes[0, 3].set_xlim(min(hext_range) * 1.1, max(hext_range) * 1.1)
    axes[0, 3].set_ylim(-1.1, 1.1)
    axes[0, 3].grid(True, lw=0.5, ls="-.")

    hd_mm_norm = Normalize(
        vmin=hd_mm[:, :, 0, :].min(),
        vmax=hd_mm[:, :, 0, :].max(),
    )(hd_mm[:, :, 0, :])
    axes[1, 0].imshow(hd_mm_norm.transpose(1, 0, 2), origin="lower")
    axes[1, 0].set_title(f"Hd FFT steps: [{itern1}]", fontsize=18)
    axes[1, 0].set_xlabel("x [cell]")
    axes[1, 0].set_ylabel("y [cell]")

    hd_un_norm = Normalize(
        vmin=hd_un[:, :, 0, :].min(),
        vmax=hd_un[:, :, 0, :].max(),
    )(hd_un[:, :, 0, :])
    axes[1, 1].imshow(hd_un_norm.transpose(1, 0, 2), origin="lower")
    axes[1, 1].set_title(f"Hd UNet steps: [{itern2}]", fontsize=18)
    axes[1, 1].set_xlabel("x [cell]")
    axes[1, 1].set_ylabel("y [cell]")

    hd_mse = np.square(hd_mm - hd_un).sum(axis=-1).transpose((1, 0, 2))[:, :, 0]
    image = axes[1, 2].imshow(hd_mse, cmap="hot", origin="lower")
    axes[1, 2].set_title(
        f"Hd FFT/UNet MSE\nMean: {hd_mse.mean():.1e}",
        fontsize=16,
    )
    fig.colorbar(image, ax=axes[1, 2])

    axes[1, 3].plot(np.arange(len(error1_rcd)), error1_rcd, label="FFT")
    axes[1, 3].plot(np.arange(len(error2_rcd)), error2_rcd, label="UNet")
    axes[1, 3].set_title("Convergence Error", fontsize=16)
    axes[1, 3].set_xlabel("Iteration", fontsize=16)
    axes[1, 3].set_ylabel(r"Maximal $\Delta m$", fontsize=16)
    axes[1, 3].set_yscale("log")
    axes[1, 3].legend(fontsize=14)

    fig.tight_layout()
    fig.savefig(
        os.path.join(save_path_iteration, f"loop_{nloop}.png"),
        dpi=300,
        bbox_inches="tight",
    )

    # The final field step contains the complete M-H curve. Save a second,
    # obvious filename so it is easy to find.
    if nloop == len(hext_range) - 1:
        fig.savefig(
            os.path.join(save_path_iteration, "final_loop_full_mh_diagnostic.png"),
            dpi=300,
            bbox_inches="tight",
        )

    plt.close(fig)


def plot_final_mh_curves(
    general_title_summary,
    save_path_summary,
    hext_values,
    m_fft,
    m_unet,
):
    """Save full and zoomed standalone M-H curves at the end of the sweep.

    The recorded quantity is the projection of the mean reduced magnetization
    onto the applied-field sweep direction, so the physically precise label is

        m_parallel = M_parallel / M_s.
    """
    folder = Path(save_path_summary)
    folder.mkdir(parents=True, exist_ok=True)

    hext_values = np.asarray(hext_values, dtype=float)
    m_fft = np.asarray(m_fft, dtype=float)
    m_unet = np.asarray(m_unet, dtype=float)

    def _save(filename, title, x_limits):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(hext_values, m_fft, lw=2.4, label="FFT/LLG")
        ax.plot(hext_values, m_unet, lw=2.4, label="UNet/LLG")
        ax.set_title(
            title + "\n" + general_title_summary,
            fontsize=11,
            fontweight="bold",
        )
        ax.set_xlabel(r"External Field $H_{ext}$ [Oe]")
        ax.set_ylabel(
            r"Reduced longitudinal magnetization "
            r"$m_{\parallel}=M_{\parallel}/M_s$"
        )
        ax.set_xlim(*x_limits)
        ax.set_ylim(-1.1, 1.1)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(
            folder / filename,
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

    # Conventional axis orientation: negative Hext on the left,
    # positive Hext on the right.
    _save(
        "mh_curve_full.png",
        "Full M-H Curve",
        (float(np.nanmin(hext_values)), float(np.nanmax(hext_values))),
    )

    # Requested transition-region view. With an ascending x axis this is
    # displayed from -750 Oe on the left to +250 Oe on the right.
    _save(
        "mh_curve_zoom_minus750_to_plus250.png",
        r"Zoomed M-H Curve ($-750 \leq H_{ext} \leq 250$ Oe)",
        (-750.0, 250.0),
    )



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Streamlined NeuralMAG M-H evaluation")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--krn", type=int, default=16)
    parser.add_argument("--w", type=int, default=32)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--cell_size", type=float, default=3.0)
    parser.add_argument("--Ms", type=float, default=1000)
    parser.add_argument("--Ax", type=float, default=0.5e-6)
    parser.add_argument("--Ku", type=float, default=0.0)
    parser.add_argument("--Kvec", type=Culist, default=(0, 0, 1))
    parser.add_argument("--damping", type=float, default=0.1)
    parser.add_argument("--dtime", type=float, default=1.0e-13)
    parser.add_argument("--error_min", type=float, default=1.0e-5)
    parser.add_argument("--max_iter", type=int, default=100000)
    parser.add_argument("--mask", type=MaskTp, default=False)
    parser.add_argument("--loss_type", type=str, default="baseline")
    parser.add_argument("--model_name", type=str, default="model.pt")
    parser.add_argument("--spin_split", type=int, default=8)
    parser.add_argument("--rand_seed", type=int, default=1234)
    parser.add_argument("--hext_start", type=float, default=1000.0)
    parser.add_argument("--hext_end", type=float, default=-1000.0)
    parser.add_argument("--hext_steps", type=int, default=201)
    parser.add_argument("--field_angle_radians", type=float, default=0.01)
    parser.add_argument("--weight_alpha", type=float, default=0.5, help="alpha used in diagnostic training weights w = 1 + alpha*R")

    parser.add_argument("--unet_stagnation_start", type=int, default=20000)
    parser.add_argument("--unet_stagnation_long_window", type=int, default=2000)
    parser.add_argument("--unet_stagnation_short_window", type=int, default=500)
    parser.add_argument("--unet_stagnation_fraction", type=float, default=0.02)
    parser.add_argument("--unet_stagnation_error", type=float, default=1.0e-4)

    parser.add_argument("--core_relative_threshold", type=float, default=0.25)
    parser.add_argument("--core_absolute_threshold", type=float, default=0.02)
    parser.add_argument("--core_min_cells", type=int, default=1)
    parser.add_argument("--core_min_abs_charge", type=float, default=0.05)

    parser.add_argument("--skip_original_plots", action="store_true")
    parser.add_argument("--skip_summary_plots", action="store_true")
    # Compatibility with older bash scripts. It is now a no-op.
    parser.add_argument("--skip_parts_2_to_5", action="store_true", help=argparse.SUPPRESS)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.hext_steps < 2:
        raise ValueError("--hext_steps must be at least 2.")

    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    )
    film_fft, film_unet, _ = initialize_models(args, device)
    _, cell_count = prepare_spin_state(film_fft, film_unet, args)

    output_dir = Path(
        f"./figs_k{args.krn}/model_{args.loss_type}/shape_{args.mask}/"
        f"size{args.w}_Ms{args.Ms}_Ax{args.Ax}_Ku{args.Ku}_dtime{args.dtime}_"
        f"split{args.spin_split}_seed{args.rand_seed}_Layers{args.layers}/"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_dir = output_dir / "final_plots"
    summary_dir.mkdir(parents=True, exist_ok=True)
    original_plot_dir = output_dir / "original_iteration_plots"
    original_plot_dir.mkdir(parents=True, exist_ok=True)

    hext_range = np.linspace(args.hext_start, args.hext_end, args.hext_steps)
    sweep_direction = np.array(
        [
            np.cos(args.field_angle_radians),
            np.sin(args.field_angle_radians),
            0.0,
        ],
        dtype=float,
    )

    recorder = PhysicsRecorder()

    x_plot: list[float] = []
    y_fft: list[float] = []
    y_unet: list[float] = []
    hd_error_mae: list[float] = []
    spin_error_mae: list[float] = []
    he_error_mae: list[float] = []
    ha_error_mae: list[float] = []
    heff_error_mae: list[float] = []

    field_keys = ("he", "ha", "hd", "heff")
    field_fft: Dict[str, list] = {key: [] for key in field_keys}
    field_unet: Dict[str, list] = {key: [] for key in field_keys}
    torque_fft: Dict[str, list] = {key: [] for key in field_keys}
    torque_unet: Dict[str, list] = {key: [] for key in field_keys}

    full_fft: Dict[str, list] = {
        key: []
        for key in (
            "demag",
            "anis",
            "excha",
            "exter",
            "total",
            "iters",
            "vortices",
            "mz",
            "time",
        )
    }
    full_unet: Dict[str, list] = {key: [] for key in full_fft}

    texture_keys = (
        "gradient_training_grid_mean",
        "gradient_active_mean",
        "gradient_interior_mean",
        "exchange_proxy_training_mean",
        "winding_training_abs_mean",
        "winding_training_abs_max",
        "exchange_energy_density",
        "active_cell_count",
        "interior_layer0_cell_count",
    )
    texture_fft: Dict[str, list] = {key: [] for key in texture_keys}
    texture_unet: Dict[str, list] = {key: [] for key in texture_keys}

    spatial_keys = (
        "gradient_spatial_variation_mean",
        "exchange_proxy_spatial_variation_mean",
    )
    spatial_fft: Dict[str, list] = {key: [] for key in spatial_keys}
    spatial_unet: Dict[str, list] = {key: [] for key in spatial_keys}

    loop_change_keys = (
        "gradient_loop_abs_change_mean",
        "gradient_loop_abs_change_per_oe",
        "gradient_tensor_loop_abs_change_mean",
        "gradient_tensor_loop_abs_change_per_oe",
        "gradient_tensor_training_rate_mean",
        "exchange_torque_training_rate_mean",
        "exchange_proxy_loop_abs_change_mean",
        "exchange_proxy_loop_abs_change_per_oe",
        "exchange_energy_density_loop_abs_change_mean",
        "exchange_energy_density_loop_abs_change_per_oe",
        "exchange_field_vector_loop_abs_change_mean",
        "exchange_field_vector_loop_abs_change_per_oe",
        "exchange_torque_vector_loop_abs_change_mean",
        "exchange_torque_vector_loop_abs_change_per_oe",
        "demag_torque_vector_loop_abs_change_mean",
        "demag_torque_vector_loop_abs_change_per_oe",
        "total_llg_drive_vector_loop_abs_change_mean",
        "total_llg_drive_vector_loop_abs_change_per_oe",
        "winding_map_loop_abs_change_mean",
        "winding_map_loop_abs_change_per_oe",
    )
    loop_change_fft: Dict[str, list] = {key: [] for key in loop_change_keys}
    loop_change_unet: Dict[str, list] = {key: [] for key in loop_change_keys}
    delta_hext_history: list[float] = []

    torque_error_keys = (
        "same_state_hd_vector_error_mean",
        "same_state_torque_error_mean",
        "trajectory_demag_torque_mismatch_mean",
        "trajectory_demag_llg_drive_mismatch_mean",
    )
    torque_error_history: Dict[str, list] = {key: [] for key in torque_error_keys}

    previous_fft_exchange_density_map = None
    previous_unet_exchange_density_map = None
    previous_fft_vector_maps: Optional[Dict[str, torch.Tensor]] = None
    previous_unet_vector_maps: Optional[Dict[str, torch.Tensor]] = None
    previous_fft_active_mask = None
    previous_unet_active_mask = None

    spin_mm = film_fft.Spin.detach().cpu().numpy().copy()
    spin_un = film_unet.Spin.detach().cpu().numpy().copy()

    general_title_summary = (
        f"Film Layers: {args.layers} | Grid Size: {args.w}x{args.w} | "
        f"Split: {args.spin_split} | Seed: {args.rand_seed} | Mask: {args.mask}\n"
        f"Ms: {args.Ms} emu/cc | Ax: {args.Ax} erg/cm | "
        f"Ku: {args.Ku} erg/cc | dtime: {args.dtime} s\n"
    )

    for nloop, hext_scalar in enumerate(hext_range):
        hext_vector = hext_scalar * sweep_direction
        print(f">>>>> loop: {nloop}, Hext: {hext_scalar}")

        previous_fft = spin_mm.copy()
        previous_unet = spin_un.copy()

        start = time.time()
        error_fft, iterations_fft = update_spin_fft(film_fft, hext_vector, args)
        fft_runtime = time.time() - start

        start = time.time()
        error_unet, iterations_unet = update_spin_unet(film_unet, hext_vector, args)
        unet_runtime = time.time() - start

        final_fft_error = float(error_fft[-1]) if len(error_fft) else np.nan
        final_unet_error = float(error_unet[-1]) if len(error_unet) else np.nan

        # Recompute all fields at the actual final normalized spin states.
        film_fft.GetHeff_intrinsic()
        film_unet.GetHeff_unetHd()

        fft_hext_tensor = torch.as_tensor(
            hext_vector,
            device=film_fft.Spin.device,
            dtype=film_fft.Spin.dtype,
        )
        unet_hext_tensor = torch.as_tensor(
            hext_vector,
            device=film_unet.Spin.device,
            dtype=film_unet.Spin.dtype,
        )
        film_fft.Heff = film_fft.Heff + fft_hext_tensor
        film_unet.Heff = film_unet.Heff + unet_hext_tensor

        film_fft.GetEnergy_detailed(Hext=hext_vector)
        film_unet.GetEnergy_detailed(Hext=hext_vector)

        fft_texture = compute_training_texture_metrics(
            film_fft.Spin,
            exchange_energy=film_fft.Energy_excha,
        )
        unet_texture = compute_training_texture_metrics(
            film_unet.Spin,
            exchange_energy=film_unet.Energy_excha,
        )
        for key in texture_keys:
            texture_fft[key].append(fft_texture[key])
            texture_unet[key].append(unet_texture[key])

        fft_spatial = compute_gradient_spatial_variation_metrics(film_fft.Spin)
        unet_spatial = compute_gradient_spatial_variation_metrics(film_unet.Spin)
        for key in spatial_keys:
            spatial_fft[key].append(fft_spatial[key])
            spatial_unet[key].append(unet_spatial[key])

        delta_hext_oe = (
            abs(float(hext_scalar) - float(hext_range[nloop - 1]))
            if nloop > 0
            else float("nan")
        )
        delta_hext_history.append(delta_hext_oe)

        previous_fft_tensor = None
        previous_unet_tensor = None
        if nloop > 0:
            previous_fft_tensor = torch.as_tensor(
                previous_fft,
                device=film_fft.Spin.device,
                dtype=film_fft.Spin.dtype,
            )
            previous_unet_tensor = torch.as_tensor(
                previous_unet,
                device=film_unet.Spin.device,
                dtype=film_unet.Spin.dtype,
            )

        fft_loop_change = compute_exact_loop_change_metrics(
            current_spin=film_fft.Spin,
            previous_spin=previous_fft_tensor,
            delta_hext_oe=delta_hext_oe if nloop > 0 else None,
            # Exact constants used by training utils.exchange_torque_rate().
            # Do NOT substitute the evaluation geometry's Ax here: the trained
            # loss called exchange_torque_rate(x, x_prev) with its defaults.
            Ms=1000.0,
            Ax=0.5e-6,
            cell_nm=(3.0, 3.0, 3.0),
        )
        unet_loop_change = compute_exact_loop_change_metrics(
            current_spin=film_unet.Spin,
            previous_spin=previous_unet_tensor,
            delta_hext_oe=delta_hext_oe if nloop > 0 else None,
            # Exact constants used by training utils.exchange_torque_rate().
            # Do NOT substitute the evaluation geometry's Ax here: the trained
            # loss called exchange_torque_rate(x, x_prev) with its defaults.
            Ms=1000.0,
            Ax=0.5e-6,
            cell_nm=(3.0, 3.0, 3.0),
        )

        current_fft_exchange_density_map = _exchange_energy_density_map(film_fft)
        current_unet_exchange_density_map = _exchange_energy_density_map(film_unet)
        current_fft_vector_maps = _physics_vector_maps(
            film_fft,
            damping=args.damping,
        )
        current_unet_vector_maps = _physics_vector_maps(
            film_unet,
            damping=args.damping,
        )
        current_fft_active_mask = (
            torch.linalg.vector_norm(film_fft.Spin, dim=-1) > 1.0e-12
        )
        current_unet_active_mask = (
            torch.linalg.vector_norm(film_unet.Spin, dim=-1) > 1.0e-12
        )

        fft_loop_change.update(
            _exchange_density_loop_change(
                current_map=current_fft_exchange_density_map,
                previous_map=previous_fft_exchange_density_map,
                current_active_mask=current_fft_active_mask,
                previous_active_mask=previous_fft_active_mask,
                delta_hext_oe=delta_hext_oe if nloop > 0 else None,
            )
        )
        unet_loop_change.update(
            _exchange_density_loop_change(
                current_map=current_unet_exchange_density_map,
                previous_map=previous_unet_exchange_density_map,
                current_active_mask=current_unet_active_mask,
                previous_active_mask=previous_unet_active_mask,
                delta_hext_oe=delta_hext_oe if nloop > 0 else None,
            )
        )

        for metric_name, current_map in current_fft_vector_maps.items():
            previous_map = (
                None
                if previous_fft_vector_maps is None
                else previous_fft_vector_maps[metric_name]
            )
            fft_loop_change.update(
                _vector_map_loop_change(
                    current_map=current_map,
                    previous_map=previous_map,
                    current_active_mask=current_fft_active_mask,
                    previous_active_mask=previous_fft_active_mask,
                    delta_hext_oe=delta_hext_oe if nloop > 0 else None,
                    metric_name=metric_name,
                )
            )

        for metric_name, current_map in current_unet_vector_maps.items():
            previous_map = (
                None
                if previous_unet_vector_maps is None
                else previous_unet_vector_maps[metric_name]
            )
            unet_loop_change.update(
                _vector_map_loop_change(
                    current_map=current_map,
                    previous_map=previous_map,
                    current_active_mask=current_unet_active_mask,
                    previous_active_mask=previous_unet_active_mask,
                    delta_hext_oe=delta_hext_oe if nloop > 0 else None,
                    metric_name=metric_name,
                )
            )

        previous_fft_exchange_density_map = (
            current_fft_exchange_density_map.detach().clone()
        )
        previous_unet_exchange_density_map = (
            current_unet_exchange_density_map.detach().clone()
        )
        previous_fft_vector_maps = {
            key: value.detach().clone()
            for key, value in current_fft_vector_maps.items()
        }
        previous_unet_vector_maps = {
            key: value.detach().clone()
            for key, value in current_unet_vector_maps.items()
        }
        previous_fft_active_mask = current_fft_active_mask.detach().clone()
        previous_unet_active_mask = current_unet_active_mask.detach().clone()

        for key in loop_change_keys:
            loop_change_fft[key].append(float(fft_loop_change[key]))
            loop_change_unet[key].append(float(unet_loop_change[key]))

        fft_spin_for_winding = (
            film_fft.Spin.permute(3, 0, 1, 2)[:, :, :, 0].unsqueeze(0)
        )
        unet_spin_for_winding = (
            film_unet.Spin.permute(3, 0, 1, 2)[:, :, :, 0].unsqueeze(0)
        )
        fft_winding_map, fft_winding_abs, fft_winding_sum = winding_density(
            fft_spin_for_winding
        )
        unet_winding_map, unet_winding_abs, unet_winding_sum = winding_density(
            unet_spin_for_winding
        )

        fft_topology = analyze_winding_components(
            fft_winding_map,
            relative_threshold=args.core_relative_threshold,
            absolute_threshold=args.core_absolute_threshold,
            min_cells=args.core_min_cells,
            min_abs_charge=args.core_min_abs_charge,
        )
        unet_topology = analyze_winding_components(
            unet_winding_map,
            relative_threshold=args.core_relative_threshold,
            absolute_threshold=args.core_absolute_threshold,
            min_cells=args.core_min_cells,
            min_abs_charge=args.core_min_abs_charge,
        )

        fields = {
            "he": (film_fft.He, film_unet.He),
            "ha": (film_fft.Ha, film_unet.Ha),
            "hd": (film_fft.Hd, film_unet.Hd),
            "heff": (film_fft.Heff, film_unet.Heff),
        }
        for key, (fft_field_value, unet_field_value) in fields.items():
            field_fft[key].append(
                _mean_field_magnitude(fft_field_value, film_fft.Spin)
            )
            field_unet[key].append(
                _mean_field_magnitude(unet_field_value, film_unet.Spin)
            )
            torque_fft[key].append(
                _mean_torque_magnitude(film_fft.Spin, fft_field_value)
            )
            torque_unet[key].append(
                _mean_torque_magnitude(film_unet.Spin, unet_field_value)
            )

        torque_error_metrics = _compute_demag_torque_error_metrics(
            film_fft,
            film_unet,
            damping=args.damping,
        )
        for key in torque_error_keys:
            torque_error_history[key].append(torque_error_metrics[key])

        snapshot = recorder.capture(
            mh_step=nloop,
            hext_scalar=hext_scalar,
            hext_vector=hext_vector,
            projection_direction=sweep_direction,
            film_fft=film_fft,
            film_unet=film_unet,
            fft_winding_abs=fft_winding_abs,
            fft_winding_sum=fft_winding_sum,
            unet_winding_abs=unet_winding_abs,
            unet_winding_sum=unet_winding_sum,
            fft_topology=fft_topology,
            unet_topology=unet_topology,
            fft_iterations=iterations_fft,
            unet_iterations=iterations_unet,
            fft_final_convergence_error=final_fft_error,
            unet_final_convergence_error=final_unet_error,
            fft_runtime_seconds=fft_runtime,
            unet_runtime_seconds=unet_runtime,
            cell_count=cell_count,
        )

        spin_mm = film_fft.Spin.detach().cpu().numpy().copy()
        spin_un = film_unet.Spin.detach().cpu().numpy().copy()
        hd_mm = film_fft.Hd.detach().cpu().numpy().copy()
        hd_un = film_unet.Hd.detach().cpu().numpy().copy()

        x_plot.append(float(hext_scalar))
        y_fft.append(float(snapshot["fft_m_projection"]))
        y_unet.append(float(snapshot["unet_m_projection"]))
        hd_error_mae.append(float(snapshot["hd_mae"]))
        spin_error_mae.append(float(snapshot["spin_mae"]))
        he_error_mae.append(float(snapshot["he_mae"]))
        ha_error_mae.append(float(snapshot["ha_mae"]))
        heff_error_mae.append(float(snapshot["heff_mae"]))

        for store, prefix in ((full_fft, "fft"), (full_unet, "unet")):
            store["demag"].append(float(snapshot[f"{prefix}_e_demag"]))
            store["anis"].append(float(snapshot[f"{prefix}_e_anis"]))
            store["excha"].append(float(snapshot[f"{prefix}_e_exchange"]))
            store["exter"].append(float(snapshot[f"{prefix}_e_external"]))
            store["total"].append(float(snapshot[f"{prefix}_e_total"]))
            store["iters"].append(int(snapshot[f"{prefix}_iterations"]))
            store["vortices"].append(float(snapshot[f"{prefix}_total_core_count"]))
            store["mz"].append(float(snapshot[f"{prefix}_mz_abs_mean"]))
            store["time"].append(float(snapshot[f"{prefix}_runtime_seconds"]))

        title = (
            general_title_summary
            + f"Loop: {nloop} | Hext = {hext_scalar:.1f} Oe | "
            f"Iterations: FFT [{iterations_fft}] | UNet [{iterations_unet}]"
        )

        # Skip intermediate original diagnostics when requested, but ALWAYS
        # save the final loop because it contains the complete M-H curve.
        if (not args.skip_original_plots) or (nloop == len(hext_range) - 1):
            plot_results(
                nloop=nloop,
                spin_mm=spin_mm,
                spin_un=spin_un,
                itern1=iterations_fft,
                itern2=iterations_unet,
                hd_mm=hd_mm,
                hd_un=hd_un,
                x_plot=x_plot,
                y1_plot=y_fft,
                y2_plot=y_unet,
                hext_range=hext_range,
                error1_rcd=error_fft,
                error2_rcd=error_unet,
                save_path_iteration=str(original_plot_dir),
                general_title_iteration=title,
            )

        if np.isclose(hext_scalar, 0.0):
            np.save(output_dir / "Mr_spin_mm.npy", spin_mm)
            np.save(output_dir / "Mr_spin_un.npy", spin_un)

        if previous_fft[..., 0].sum() > 0 and spin_mm[..., 0].sum() <= 0:
            np.save(output_dir / f"Hc{nloop-1}_spin_mm.npy", previous_fft)
            np.save(output_dir / f"Hc{nloop}_spin_mm.npy", spin_mm)

        if previous_unet[..., 0].sum() > 0 and spin_un[..., 0].sum() <= 0:
            np.save(output_dir / f"Hc{nloop-1}_spin_un.npy", previous_unet)
            np.save(output_dir / f"Hc{nloop}_spin_un.npy", spin_un)

    physics_df = recorder.dataframe()

    for prefix, histories in (("fft", texture_fft), ("unet", texture_unet)):
        for key, values in histories.items():
            physics_df[f"{prefix}_{key}"] = np.asarray(values, dtype=float)

    for prefix, histories in (("fft", spatial_fft), ("unet", spatial_unet)):
        for key, values in histories.items():
            physics_df[f"{prefix}_{key}"] = np.asarray(values, dtype=float)

    for prefix, histories in (("fft", loop_change_fft), ("unet", loop_change_unet)):
        for key, values in histories.items():
            physics_df[f"{prefix}_{key}"] = np.asarray(values, dtype=float)

    for key, values in torque_error_history.items():
        physics_df[key] = np.asarray(values, dtype=float)

    physics_df["delta_hext_oe"] = np.asarray(delta_hext_history, dtype=float)

    # Mentor-requested exact training-rate diagnostics on the FFT reference path.
    r_grad = np.asarray(loop_change_fft["gradient_tensor_training_rate_mean"], dtype=float)
    r_torque = np.asarray(loop_change_fft["exchange_torque_training_rate_mean"], dtype=float)
    w_grad = 1.0 + args.weight_alpha * r_grad
    w_torque = 1.0 + args.weight_alpha * r_torque
    physics_df["R_grad"] = r_grad
    physics_df["R_torque"] = r_torque
    physics_df["w_grad"] = w_grad
    physics_df["w_torque"] = w_torque
    physics_df["weight_alpha"] = float(args.weight_alpha)

    physics_df.to_csv(output_dir / "physics_snapshots.csv", index=False)

    np.save(output_dir / "Hext_array.npy", np.asarray(x_plot))
    np.save(output_dir / "Mext_array_mm.npy", np.asarray(y_fft))
    np.save(output_dir / "Mext_array_un.npy", np.asarray(y_unet))
    np.save(output_dir / "instantaneous_hd_mae.npy", np.asarray(hd_error_mae))
    np.save(output_dir / "trajectory_shift_mae.npy", np.asarray(spin_error_mae))
    np.save(output_dir / "exchange_field_mae.npy", np.asarray(he_error_mae))
    np.save(output_dir / "anisotropy_field_mae.npy", np.asarray(ha_error_mae))

    exact_loop_payload = {"hext_scalar": np.asarray(x_plot, dtype=float)}
    for prefix, histories in (("fft", loop_change_fft), ("unet", loop_change_unet)):
        for key, values in histories.items():
            array = np.asarray(values, dtype=float)
            exact_loop_payload[f"{prefix}_{key}"] = array
            # These individual files are redundant with the NPZ and CSV, but
            # make future standalone plotting and spot checks straightforward.
            np.save(output_dir / f"{prefix}_{key}.npy", array)
    exact_loop_payload["delta_hext_oe"] = np.asarray(delta_hext_history, dtype=float)
    np.savez(output_dir / "exact_loop_change_metrics.npz", **exact_loop_payload)

    np.savez(
        output_dir / "mentor_weight_diagnostics.npz",
        hext_scalar=np.asarray(x_plot, dtype=float),
        R_grad=r_grad,
        R_torque=r_torque,
        w_grad=w_grad,
        w_torque=w_torque,
        trajectory_mae=np.asarray(spin_error_mae, dtype=float),
        alpha=np.asarray(float(args.weight_alpha)),
    )

    np.savez(
        output_dir / "demag_torque_error_tracking.npz",
        hext_scalar=np.asarray(x_plot, dtype=float),
        **{
            key: np.asarray(values, dtype=float)
            for key, values in torque_error_history.items()
        },
    )

    print(
        f"Saved {len(physics_df)} converged physics snapshots to "
        f"{output_dir / 'physics_snapshots.csv'}"
    )

    # Always save the two final standalone M-H curves, even when the optional
    # summary-plot suite is disabled.
    plot_final_mh_curves(
        general_title_summary,
        str(summary_dir),
        x_plot,
        y_fft,
        y_unet,
    )

    if args.skip_summary_plots:
        return

    y_limits = {
        "hd": (0.0, 400.0),
        "trajectory": (0.0, 0.75),
        "exchange": (0.0, 400.0),
        "anisotropy": (0.0, 400.0),
    }

    plot_full_energy_summary(
        general_title_summary,
        str(summary_dir),
        full_fft,
        full_unet,
        hext_range,
    )
    plot_performance_summary(
        general_title_summary,
        str(summary_dir),
        full_fft,
        full_unet,
        hext_range,
    )
    plot_error_summary(
        general_title_summary,
        str(summary_dir),
        hext_range,
        hd_error_mae,
        spin_error_mae,
        he_error_mae,
        ha_error_mae,
        y_limits=y_limits,
    )
    plot_fields_summary(
        general_title_summary,
        str(summary_dir),
        hext_range,
        field_fft["he"],
        field_unet["he"],
        field_fft["ha"],
        field_unet["ha"],
        field_fft["hd"],
        field_unet["hd"],
        field_fft["heff"],
        field_unet["heff"],
    )
    plot_magnetization_gradient_vs_hext(
        general_title_summary,
        str(summary_dir),
        hext_range,
        texture_fft,
        texture_unet,
    )
    plot_training_winding_density_vs_hext(
        general_title_summary,
        str(summary_dir),
        hext_range,
        texture_fft,
        texture_unet,
    )
    plot_exchange_energy_density_vs_hext(
        general_title_summary,
        str(summary_dir),
        hext_range,
        texture_fft,
        texture_unet,
    )
    plot_gradient_change_rate_summary(
        general_title_summary,
        str(summary_dir),
        hext_range,
        spatial_fft,
        spatial_unet,
        loop_change_fft,
        loop_change_unet,
    )
    plot_training_gradient_tensor_rate_vs_hext(
        general_title_summary,
        str(summary_dir),
        hext_range,
        loop_change_fft,
        loop_change_unet,
    )
    plot_training_weight_error_overlays(
        general_title_summary,
        str(summary_dir),
        hext_range,
        loop_change_fft,
        spin_error_mae,
        alpha=args.weight_alpha,
    )
    plot_physics_vector_rate_summary(
        general_title_summary,
        str(summary_dir),
        hext_range,
        loop_change_fft,
        loop_change_unet,
    )
    plot_torque_summary(
        general_title_summary,
        str(summary_dir),
        hext_range,
        torque_fft,
        torque_unet,
    )
    plot_torque_error_summary(
        general_title_summary,
        str(summary_dir),
        hext_range,
        torque_error_history,
    )
    plot_loop_change_error_overlays(
        general_title_summary,
        str(summary_dir),
        hext_range,
        error_histories={
            "hd": hd_error_mae,
            "spin": spin_error_mae,
            "he": he_error_mae,
            "ha": ha_error_mae,
        },
        loop_change_fft=loop_change_fft,
        loop_change_unet=loop_change_unet,
    )


if __name__ == "__main__":
    main()



