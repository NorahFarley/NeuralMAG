# -*- coding: utf-8 -*-
"""
NeuralMAG M-H evaluation with FFT-ground-truth transition analysis.

The repository's original per-field ``plot_results`` diagnostic is retained.
Expensive per-LLG-iteration physics-history plots are intentionally removed;
all Parts 1-5 operate on one converged state per external-field value.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import Normalize

from egs.NMI.MH_evaluate.searcher import (
    CrossSweepAggregator,
    LeadingIndicatorAnalyzer,
    ManuscriptOutputGenerator,
    PhysicsRecorder,
    PublicationFigureGenerator,
    TransitionAnalyzer,
    analyze_winding_components,
)
from libs.misc import Culist, MaskTp, spin_prepare, winding_density
import libs.MAG2305 as MAG2305
from libs.Unet import UNet
from plots import (
    plot_error_correlations,
    plot_error_summary,
    plot_error_vs_transition_proximity,
    plot_fields_summary,
    plot_full_energy_summary,
    plot_performance_summary,
)


def load_unet_model(args: argparse.Namespace, device: torch.device) -> Path:
    # load Unet Model
    model = UNet(kc=args.krn, inc=args.layers*3, ouc=args.layers*3).eval().to(device)
    checkpoint = Path("../ckpt") / f"k{args.krn}" / args.model_name
    if not checkpoint.is_file():
        raise FileNotFoundError(f"UNet checkpoint was not found: {checkpoint.resolve()}")
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    MAG2305.load_model(model)
    print(f"UNet model loaded from {checkpoint}")
    return checkpoint.resolve()

def initialize_models(args: argparse.Namespace, device: torch.device):
    common = dict(types="bulk", size=(args.w, args.w, args.layers), cell=(args.cell_size, args.cell_size, args.cell_size), Ms=args.Ms,
                  Ax=args.Ax, Ku=args.Ku, Kvec=args.Kvec, device=str(device),)
    film_fft = MAG2305.mmModel(**common)
    film_unet = MAG2305.mmModel(**common)
    print(f"Creating {args.layers} layer models")
    film_fft.DemagInit()
    print("FFT demagnetization matrix initialized")
    checkpoint = load_unet_model(args, device)
    return film_fft, film_unet, checkpoint

def prepare_spin_state(film1, film2, args: argparse.Namespace):
    """
    Prepare the initial spin state.
    """
    spin_split = 8
    rand_seed  = 1234
    spin = spin_prepare(spin_split, film1, rand_seed, mask=args.mask)
    film1.SpinInit(spin)
    film2.SpinInit(spin)
    cell_count = (np.linalg.norm(spin, axis=-1) > 0).sum()
    return spin_split, cell_count

def update_spin_fft(model, Hext: np.ndarray, args: argparse.Namespace):
    """
    Update the spin state of the FFT model and retain only the convergence-error trajectory.
    """
    error = 1.0
    iteration = 0
    error_record = []
    while iteration < args.max_iter and error > args.error_min:
        error = model.SpinLLG_RK4(Hext=Hext, dtime=args.dtime, damping=args.damping)
        error_record.append(float(error))
        if error <= args.error_min or iteration % 1000 == 0:
            print(f"Iteration: {iteration}\nError_converge FFT: {error:.2e}")
        iteration += 1
    return np.asarray(error_record, dtype=float), iteration

def update_spin_unet(model, Hext: np.ndarray, args: argparse.Namespace):
    """Relax the UNet-driven model and retain only convergence diagnostics."""
    error = 1.0
    iteration = 0
    error_record = []
    while iteration < args.max_iter and error > args.error_min:
        error = model.SpinLLG_RK4_unetHd(Hext=Hext, dtime=args.dtime, damping=args.damping)
        error_record.append(float(error))
        if iteration > args.unet_stagnation_start and len(error_record) >= args.unet_stagnation_long_window:
            long_mean = float(np.mean(error_record[-args.unet_stagnation_long_window:]))
            short_mean = float(np.mean(error_record[-args.unet_stagnation_short_window:]))
            fluctuation = abs(long_mean - short_mean) / max(abs(long_mean), 1.0e-30)
            if fluctuation < args.unet_stagnation_fraction and error < args.unet_stagnation_error:
                print("UNet convergence error has stagnated; ending this field relaxation.")
                break
        if error <= args.error_min or iteration % 1000 == 0:
            print(f"Iteration: {iteration}\nError_converge UNet: {error:.2e}")
        iteration += 1
    return np.asarray(error_record, dtype=float), iteration


def _mean_field_magnitude(field: torch.Tensor, spin: torch.Tensor) -> float:
    active = torch.linalg.vector_norm(spin, dim=-1) > 1.0e-12
    magnitude = torch.linalg.vector_norm(field, dim=-1)
    selected = magnitude[active]
    return float(selected.mean().item()) if selected.numel() else float(magnitude.mean().item())


def plot_results(nloop, spin_mm, spin_un, itern1, itern2, Hd_mm, Hd_un, x_plot, y1_plot, y2_plot, Hext_range, 
                 error1_rcd, error2_rcd, save_path_iteration, general_title_iteration):
    """
    Plot and save the results.
    """
    fig, axs = plt.subplots(2, 4, figsize=(20, 10))

    fig.suptitle(general_title_iteration, fontsize=13, fontweight='bold')
    
    # Plot spin-mm RGB figures
    spin = (spin_mm + 1)/2
    axs[0, 0].imshow(spin[:,:,0,:].transpose(1,0,2), alpha=1.0, origin='lower')
    axs[0, 0].set_title('Spin-mm steps: [{:d}]'.format(itern1), fontsize=18)
    axs[0, 0].set_xlabel('x [nm]')
    axs[0, 0].set_ylabel('y [nm]')

    # Plot spin-mm RGB figures
    spin = (spin_un + 1)/2
    axs[0, 1].imshow(spin[:,:,0,:].transpose(1,0,2), alpha=1.0, origin='lower')
    axs[0, 1].set_title('Spin-un steps: [{:d}]'.format(itern2), fontsize=18)
    axs[0, 1].set_xlabel('x [nm]')
    axs[0, 1].set_ylabel('y [nm]')

    # Plot mse heatmap
    mse = np.square(spin_un - spin_mm).sum(axis=-1)
    mse = mse.transpose((1, 0, 2))[:,:,0]
    im = axs[0, 2].imshow(mse, cmap='hot', origin="lower")
    axs[0, 2].set_title('MSE of spin_mm & spin_un, \nMSE_avg: {:4f}'.format(mse.mean()), fontsize=16)
    cbar1 = fig.colorbar(im, ax=axs[0, 2])

    #MH loop figures
    axs[0, 3].plot(x_plot, y1_plot, lw=1.5, label='mm', marker='o', markersize=0, color='blue',  alpha=0.6)
    axs[0, 3].plot(x_plot, y2_plot, lw=1.5, label='un', marker='o', markersize=0, color='red', alpha=0.6)
    axs[0, 3].legend(fontsize=16, loc='upper left')
    axs[0, 3].set_title('M-H data',fontsize=16)
    axs[0, 3].set_xlabel('Hext [Oe]',fontsize=16)
    axs[0, 3].set_ylabel('Mext/Ms',fontsize=16)
    axs[0, 3].set_xlim(min(Hext_range)*1.1, max(Hext_range)*1.1)
    axs[0, 3].set_ylim(-1.1, 1.1)
    axs[0, 3].grid(True, axis='both', lw=0.5, ls='-.')

    #Hd-mm rgb figures
    Hd_mm_norm = Normalize(vmin=Hd_mm[:,:,0,:].min(), vmax=Hd_mm[:,:,0,:].max())(Hd_mm[:,:,0,:])
    axs[1, 0].imshow(Hd_mm_norm.transpose(1,0,2), alpha=1.0, origin='lower')
    axs[1, 0].set_title('Hd_mm steps: [{}]'.format(itern1), fontsize=18)
    axs[1, 0].set_xlabel('x [nm]')
    axs[1, 0].set_ylabel('y [nm]')

    #Hd-un rgb figures
    Hd_un_norm = Normalize(vmin=Hd_un[:,:,0,:].min(), vmax=Hd_un[:,:,0,:].max())(Hd_un[:,:,0,:])
    axs[1, 1].imshow(Hd_un_norm.transpose(1,0,2), alpha=1.0, origin='lower')
    axs[1, 1].set_title('Hd_un steps: [{}]'.format(itern2), fontsize=18)
    axs[1, 1].set_xlabel('x [nm]')
    axs[1, 1].set_ylabel('y [nm]')

    # Plot mse heatmap
    mse_Hd = np.square(Hd_mm - Hd_un).sum(axis=-1)
    mse_Hd = mse_Hd.transpose((1, 0, 2))[:,:,0]
    im = axs[1, 2].imshow(mse_Hd, cmap='hot', origin="lower")
    axs[1, 2].set_title('MSE of Hd_mm & Hd_un \nMSE_avg{:.1e}'.format(mse_Hd.mean()), fontsize=16)
    cbar2 = fig.colorbar(im, ax=axs[1, 2])

    # Plot error
    x1 = np.arange(len(error1_rcd))
    axs[1, 3].plot(x1, error1_rcd, color='blue', alpha=0.6, label='mm')
    x2 = np.arange(len(error2_rcd))
    axs[1, 3].plot(x2, error2_rcd, color='red',  alpha=0.6, label='un')
    axs[1, 3].set_title('Error plot', fontsize=16)
    axs[1, 3].set_xlabel('Iterations', fontsize=16)
    axs[1, 3].set_ylabel('Maximal $\\Delta$m', fontsize=16)
    axs[1, 3].set_yscale('log')
    axs[1, 3].legend(fontsize=16, loc='upper right')
    
    # save img
    plt.tight_layout()
    plt.savefig(os.path.join(save_path_iteration, f'loop_{nloop}.png'), dpi=300)
    plt.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="NeuralMAG M-H evaluation and transition analysis")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--krn', type=int, default=16)
    parser.add_argument('--w', type=int, default=32)
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--cell_size', type=float, default=3.0, help='Cubic cell size in nm')
    parser.add_argument('--Ms', type=float, default=1000)
    parser.add_argument('--Ax', type=float, default=0.5e-6)
    parser.add_argument('--Ku', type=float, default=0.0)
    parser.add_argument('--Kvec', type=Culist, default=(0, 0, 1))
    parser.add_argument('--damping', type=float, default=0.1)
    parser.add_argument('--dtime', type=float, default=1.0e-13)
    parser.add_argument('--error_min', type=float, default=1.0e-5)
    parser.add_argument('--max_iter', type=int, default=100000)
    parser.add_argument('--mask', type=MaskTp, default=False)
    parser.add_argument('--loss_type', type=str, default='baseline')
    parser.add_argument('--model_name', type=str, default='model.pt')
    parser.add_argument('--spin_split', type=int, default=8)
    parser.add_argument('--rand_seed', type=int, default=1234)
    parser.add_argument('--hext_start', type=float, default=1000.0)
    parser.add_argument('--hext_end', type=float, default=-1000.0)
    parser.add_argument('--hext_steps', type=int, default=201)
    parser.add_argument('--field_angle_radians', type=float, default=0.01)

    parser.add_argument('--unet_stagnation_start', type=int, default=20000)
    parser.add_argument('--unet_stagnation_long_window', type=int, default=2000)
    parser.add_argument('--unet_stagnation_short_window', type=int, default=500)
    parser.add_argument('--unet_stagnation_fraction', type=float, default=0.02)
    parser.add_argument('--unet_stagnation_error', type=float, default=1.0e-4)

    parser.add_argument('--core_relative_threshold', type=float, default=0.25)
    parser.add_argument('--core_absolute_threshold', type=float, default=0.02)
    parser.add_argument('--core_min_cells', type=int, default=1)
    parser.add_argument('--core_min_abs_charge', type=float, default=0.05)

    parser.add_argument('--transition_merge_gap', type=int, default=1)
    parser.add_argument('--transition_winding_tolerance', type=float, default=0.0)
    parser.add_argument('--transition_m_z_threshold', type=float, default=3.0)
    parser.add_argument('--transition_min_m_change', type=float, default=0.02)

    parser.add_argument('--indicator_primary_window', type=int, default=10, choices=(3, 5, 10, 20))
    parser.add_argument('--indicator_max_lag', type=int, default=20)
    parser.add_argument('--indicator_lead_z_threshold', type=float, default=1.5)
    parser.add_argument('--indicator_post_event_exclusion', type=int, default=3)
    parser.add_argument('--indicator_permutations', type=int, default=500)
    parser.add_argument('--indicator_bootstrap', type=int, default=500)
    parser.add_argument('--indicator_top_n', type=int, default=12)
    parser.add_argument('--indicator_random_seed', type=int, default=1234)

    parser.add_argument('--publication_top_n', type=int, default=4)
    parser.add_argument('--publication_pre_steps', type=int, default=20)
    parser.add_argument('--publication_post_steps', type=int, default=10)
    parser.add_argument('--publication_dpi', type=int, default=300)
    parser.add_argument('--publication_formats', type=str, default='png,pdf')

    parser.add_argument('--manuscript_top_n', type=int, default=10)
    parser.add_argument('--manuscript_formats', type=str, default='csv,tex,md')
    parser.add_argument('--run_label', type=str, default='')
    parser.add_argument('--aggregate_root', type=str, default='')
    parser.add_argument('--aggregate_output', type=str, default='')
    parser.add_argument('--aggregate_min_runs', type=int, default=2)

    parser.add_argument('--skip_original_plots', action='store_true')
    parser.add_argument('--skip_summary_plots', action='store_true')
    parser.add_argument('--skip_parts_2_to_5', action='store_true')
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.hext_steps < 2:
        raise ValueError('--hext_steps must be at least 2.')
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    film_fft, film_unet, checkpoint_path = initialize_models(args, device)
    initial_spin, cell_count = prepare_spin_state(film_fft, film_unet, args)

    output_dir = Path(f"./figs_k{args.krn}/model_{args.loss_type}/shape_{args.mask}/"
                      f"size{args.w}_Ms{args.Ms}_Ax{args.Ax}_Ku{args.Ku}_dtime{args.dtime}_"
                      f"split{args.spin_split}_seed{args.rand_seed}_Layers{args.layers}/")
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_dir = output_dir / "summary_plots"
    summary_dir.mkdir(parents=True, exist_ok=True)
    original_plot_dir = output_dir / "original_iteration_plots"
    if not args.skip_original_plots:
        original_plot_dir.mkdir(parents=True, exist_ok=True)
        hext_range = np.linspace(args.hext_start, args.hext_end, args.hext_steps)
    sweep_direction = np.array([np.cos(args.field_angle_radians), np.sin(args.field_angle_radians), 0.0,], dtype=float)

    recorder = PhysicsRecorder()
    x_plot: list[float] = []
    y_fft: list[float] = []
    y_unet: list[float] = []
    hd_error_mae: list[float] = []
    spin_error_mae: list[float] = []
    he_error_mae: list[float] = []
    ha_error_mae: list[float] = []
    heff_error_mae: list[float] = []
    he_fft_plot: list[float] = []
    he_unet_plot: list[float] = []
    ha_fft_plot: list[float] = []
    ha_unet_plot: list[float] = []
    hd_fft_plot: list[float] = []
    hd_unet_plot: list[float] = []
    heff_fft_plot: list[float] = []
    heff_unet_plot: list[float] = []
    full_fft: Dict[str, list] = {key: [] for key in ('demag', 'anis', 'excha', 'exter', 'total', 'iters', 'vortices', 'mz', 'time')}
    full_unet: Dict[str, list] = {key: [] for key in ('demag', 'anis', 'excha', 'exter', 'total', 'iters', 'vortices', 'mz', 'time')}

    spin_mm = initial_spin.copy()
    spin_un = initial_spin.copy()

    general_title_summary = (f"Film Layers: {args.layers} | Grid Size: {args.w}x{args.w} | Split: {args.spin_split} | "
                             f"Seed: {args.rand_seed} | Mask: {args.mask}\n"
                             f"Material Properties — Ms: {args.Ms} emu/cc | Ax: {args.Ax} erg/cm | Ku: {args.Ku} erg/cc\n")

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

        film_fft.GetEnergy_detailed(Hext=hext_vector)
        film_unet.GetEnergy_detailed(Hext=hext_vector)

        fft_spin_for_winding = film_fft.Spin.permute(3, 0, 1, 2)[:, :, :, 0].unsqueeze(0)
        unet_spin_for_winding = film_unet.Spin.permute(3, 0, 1, 2)[:, :, :, 0].unsqueeze(0)
        fft_winding_map, fft_winding_abs, fft_winding_sum = winding_density(fft_spin_for_winding)
        unet_winding_map, unet_winding_abs, unet_winding_sum = winding_density(unet_spin_for_winding)
        fft_topology = analyze_winding_components(fft_winding_map, relative_threshold=args.core_relative_threshold, absolute_threshold=args.core_absolute_threshold,
                                                  min_cells=args.core_min_cells, min_abs_charge=args.core_min_abs_charge)

        unet_topology = analyze_winding_components(unet_winding_map, relative_threshold=args.core_relative_threshold, absolute_threshold=args.core_absolute_threshold,
                                                   min_cells=args.core_min_cells, min_abs_charge=args.core_min_abs_charge)

        snapshot = recorder.capture(mh_step=nloop,
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
                                    cell_count=cell_count)

        spin_mm = film_fft.Spin.detach().cpu().numpy()
        spin_un = film_unet.Spin.detach().cpu().numpy()
        hd_mm = film_fft.Hd.detach().cpu().numpy()
        hd_un = film_unet.Hd.detach().cpu().numpy()

        x_plot.append(float(hext_scalar))
        y_fft.append(float(snapshot['fft_m_projection']))
        y_unet.append(float(snapshot['unet_m_projection']))
        hd_error_mae.append(float(snapshot['hd_mae']))
        spin_error_mae.append(float(snapshot['spin_mae']))
        he_error_mae.append(float(snapshot['he_mae']))
        ha_error_mae.append(float(snapshot['ha_mae']))
        heff_error_mae.append(float(snapshot['heff_mae']))

        he_fft_plot.append(float(snapshot['fft_he_mean']))
        ha_fft_plot.append(float(snapshot['fft_ha_mean']))
        hd_fft_plot.append(float(snapshot['fft_hd_mean']))
        heff_fft_plot.append(float(snapshot['fft_heff_mean']))
        he_unet_plot.append(_mean_field_magnitude(film_unet.He, film_unet.Spin))
        ha_unet_plot.append(_mean_field_magnitude(film_unet.Ha, film_unet.Spin))
        hd_unet_plot.append(_mean_field_magnitude(film_unet.Hd, film_unet.Spin))
        heff_unet_plot.append(_mean_field_magnitude(film_unet.Heff, film_unet.Spin))

        for store, prefix in ((full_fft, 'fft'), (full_unet, 'unet')):
            store['demag'].append(float(snapshot[f'{prefix}_e_demag']))
            store['anis'].append(float(snapshot[f'{prefix}_e_anis']))
            store['excha'].append(float(snapshot[f'{prefix}_e_exchange']))
            store['exter'].append(float(snapshot[f'{prefix}_e_external']))
            store['total'].append(float(snapshot[f'{prefix}_e_total']))
            store['iters'].append(int(snapshot[f'{prefix}_iterations']))
            store['vortices'].append(float(snapshot.get(f'{prefix}_total_core_count', snapshot[f'{prefix}_winding_abs'])))
            store['mz'].append(float(snapshot[f'{prefix}_mz_abs_mean']))
            store['time'].append(float(snapshot[f'{prefix}_runtime_seconds']))

        title = (general_title_summary+ f"Loop: {nloop} | Hext = {hext_scalar:.1f} Oe | Iterations: FFT [{iterations_fft}] | UNet [{iterations_unet}]")

        if not args.skip_original_plots:
            plot_results(nloop=nloop, spin_mm=spin_mm, spin_un=spin_un, itern1=iterations_fft, itern2=iterations_unet, Hd_mm=hd_mm, Hd_un=hd_un,
                         x_plot=x_plot, y1_plot=y_fft, y2_plot=y_unet, Hext_range=hext_range, error1_rcd=error_fft, error2_rcd=error_unet, save_path_iteration=str(original_plot_dir),
                         general_title_iteration=title)

        if np.isclose(hext_scalar, 0.0):
            np.save(output_dir / "Mr_spin_mm.npy", spin_mm)
            np.save(output_dir / "Mr_spin_un.npy", spin_un)
        if previous_fft[..., 0].sum() > 0 and spin_mm[..., 0].sum() <= 0:
            np.save(output_dir / f"Hc{nloop-1}_spin_mm.npy", previous_fft)
            np.save(output_dir / f"Hc{nloop}_spin_mm.npy", spin_mm)
        if previous_unet[..., 0].sum() > 0 and spin_un[..., 0].sum() <= 0:
            np.save(output_dir / f"Hc{nloop-1}_spin_un.npy", previous_unet)
            np.save(output_dir / f"Hc{nloop}_spin_un.npy", spin_un)

    physics_df = recorder.save_csv(output_dir / "physics_snapshots.csv")
    np.save(output_dir / "Hext_array.npy", np.asarray(x_plot))
    np.save(output_dir / "Mext_array_mm.npy", np.asarray(y_fft))
    np.save(output_dir / "Mext_array_un.npy", np.asarray(y_unet))
    np.save(output_dir / "instantaneous_hd_mae.npy", np.asarray(hd_error_mae))
    np.save(output_dir / "trajectory_shift_mae.npy", np.asarray(spin_error_mae))
    print(f"Saved {len(physics_df)} converged physics snapshots to {output_dir / 'physics_snapshots.csv'}")

    if not args.skip_summary_plots:
        plot_full_energy_summary(general_title_summary, str(summary_dir), full_fft, full_unet, hext_range)
        plot_performance_summary(general_title_summary, str(summary_dir), full_fft, full_unet, hext_range)
        plot_error_summary(general_title_summary, str(summary_dir), hext_range, hd_error_mae, spin_error_mae, he_error_mae, ha_error_mae)
        plot_fields_summary(general_title_summary, str(summary_dir), hext_range, he_fft_plot, he_unet_plot, ha_fft_plot, ha_unet_plot, hd_fft_plot, hd_unet_plot, heff_fft_plot, heff_unet_plot)
        plot_error_correlations(general_title_summary, str(summary_dir), hd_error_mae, he_error_mae, ha_error_mae, spin_error_mae, Hext_range=hext_range)

    if args.skip_parts_2_to_5:
        return
    
    transition_directory = summary_dir / "transition_analysis"
    transition_analyzer = TransitionAnalyzer(physics_df, merge_gap=args.transition_merge_gap, winding_tolerance=args.transition_winding_tolerance, magnetization_z_threshold=args.transition_m_z_threshold,
                                             min_magnetization_change=args.transition_min_m_change, pretransition_windows=(3, 5, 10, 20))
    events_df, labeled_df = transition_analyzer.run(transition_directory, general_title_summary)

    if not args.skip_summary_plots and 'fft_winding_abs' in labeled_df:
        for event_type in ('both', 'nucleation', 'annihilation'):
            plot_error_vs_transition_proximity(general_title_summary, str(summary_dir), spin_error_mae, 
                                               labeled_df['fft_winding_abs'].to_numpy(dtype=float), event_type=event_type)

    part3_directory = summary_dir / "leading_indicator_analysis"
    indicator_analyzer = LeadingIndicatorAnalyzer(labeled_df, events_df=events_df, error_targets=('spin_mae', 'hd_mae'), 
                                                  pretransition_windows=(3, 5, 10, 20), primary_window=args.indicator_primary_window, 
                                                  max_lag=args.indicator_max_lag, lead_z_threshold=args.indicator_lead_z_threshold,
                                                  post_event_exclusion=args.indicator_post_event_exclusion, n_permutations=args.indicator_permutations, 
                                                  n_bootstrap=args.indicator_bootstrap, random_seed=args.indicator_random_seed)
    
    indicator_ranking = indicator_analyzer.run(part3_directory, general_title_summary, top_n=args.indicator_top_n)

    publication_formats = tuple(item.strip() for item in args.publication_formats.split(',') if item.strip())
    publication_generator = PublicationFigureGenerator(labeled_df, events_df, indicator_ranking, primary_window=args.indicator_primary_window,
                                                       top_n=args.publication_top_n, pre_steps=args.publication_pre_steps, post_steps=args.publication_post_steps,
                                                       dpi=args.publication_dpi, formats=publication_formats, lead_z_threshold=args.indicator_lead_z_threshold)
    publication_generator.run(summary_dir / "publication_figures", general_title_summary)

    run_metadata: Dict[str, Any] = {
        'run_label': args.run_label,
        'grid_width': args.w,
        'layers': args.layers,
        'cell_size_nm': args.cell_size,
        'Ms_emu_per_cc': args.Ms,
        'Ax_erg_per_cm': args.Ax,
        'Ku_erg_per_cc': args.Ku,
        'Kvec': list(args.Kvec),
        'dtime_seconds': args.dtime,
        'damping': args.damping,
        'error_min': args.error_min,
        'max_iter': args.max_iter,
        'mask': str(args.mask),
        'loss_type': args.loss_type,
        'model_name': args.model_name,
        'spin_split': args.spin_split,
        'rand_seed': args.rand_seed,
        'hext_start_oe': args.hext_start,
        'hext_end_oe': args.hext_end,
        'hext_steps': args.hext_steps,
        'field_angle_radians': args.field_angle_radians,
        'core_relative_threshold': args.core_relative_threshold,
        'core_absolute_threshold': args.core_absolute_threshold,
        'core_min_cells': args.core_min_cells,
        'core_min_abs_charge': args.core_min_abs_charge,
        'transition_merge_gap': args.transition_merge_gap,
        'transition_winding_tolerance': args.transition_winding_tolerance,
        'transition_m_z_threshold': args.transition_m_z_threshold,
        'transition_min_m_change': args.transition_min_m_change,
        'indicator_primary_window': args.indicator_primary_window,
        'indicator_max_lag': args.indicator_max_lag,
        'indicator_permutations': args.indicator_permutations,
        'indicator_bootstrap': args.indicator_bootstrap,
        'checkpoint_path': str(checkpoint_path),
        'evaluation_script_path': str(Path(__file__).resolve()),
        'searcher_path': str((Path(__file__).resolve().parent / 'searcher.py')),}

    manuscript_formats = tuple(item.strip() for item in args.manuscript_formats.split(',') if item.strip())
    manuscript_generator = ManuscriptOutputGenerator(physics_df, labeled_df, events_df, indicator_ranking, part3_directory=part3_directory, 
                                                     run_metadata=run_metadata, primary_window=args.indicator_primary_window, top_n=args.manuscript_top_n, 
                                                     formats=manuscript_formats)
    manuscript_generator.run(summary_dir / "manuscript_outputs")

    if args.aggregate_root:
        aggregate_output = Path(args.aggregate_output) if args.aggregate_output else Path(args.aggregate_root) / "cross_sweep_aggregate"
        aggregator = CrossSweepAggregator(args.aggregate_root, min_runs=args.aggregate_min_runs)
        summary = aggregator.run(aggregate_output)
        print(f"Cross-sweep aggregation complete: {summary}")


if __name__ == '__main__':
    main()

