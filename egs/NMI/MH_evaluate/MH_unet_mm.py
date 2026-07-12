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


def load_unet_model(args):
    # load Unet Model
    model = UNet(kc=args.krn, inc=args.layers*3, ouc=args.layers*3).eval().to(device)
    checkpoint = '../ckpt/k{}/{}'.format(args.krn, args.model_name)
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

def prepare_spin_state(film1, film2, argsargs: argparse.Namespace):
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MH Test')
    parser.add_argument('--gpu',         type=int,    default=0,         help='GPU ID (default: 0)')
    parser.add_argument('--krn',         type=int,    default=16,        help='unet first layer kernels (default: 16)')
    parser.add_argument('--w',           type=int,    default=32,        help='MAG model size (default: 32)')
    parser.add_argument('--layers',      type=int,    default=2,         help='MAG model layers (default: 2)')

    parser.add_argument('--Ms',          type=float,  default=1000,      help='MAG model Ms (default: 1000)')
    parser.add_argument('--Ax',          type=float,  default=0.5e-6,    help='MAG model Ax (default: 0.5e-6)')
    parser.add_argument('--Ku',          type=float,  default=0.0,       help='MAG model Ku (default: 0.0)')
    parser.add_argument('--Kvec',        type=Culist, default=(0,0,1),   help='MAG model Kvec (default: (0,0,1))')
    parser.add_argument('--damping',     type=float,  default=0.1,       help='MAG model damping (default: 0.1)')
    parser.add_argument('--Hext_val',    type=float,  default=0,         help='external field value (default: 0.0)')

    parser.add_argument('--dtime',       type=float,  default=1.0e-13,   help='real time step (default: 1.0e-13)')
    parser.add_argument('--error_min',   type=float,  default=1.0e-5,    help='min error (default: 1.0e-5)')
    parser.add_argument('--max_iter',    type=int,    default=100000,    help='max iteration number (default: 100000)')
    parser.add_argument('--mask',        type=MaskTp, default=False,     help='mask (default: False)')
    parser.add_argument('--loss_type',  type=str,  default='baseline', help='loss weighting method')
    parser.add_argument('--model_name',  type=str,  default='model.pt', help='name of model')

    args = parser.parse_args() 
    
    device = torch.device("cuda:{}".format(args.gpu))

    # create two film models
    film1, film2 = initialize_models(args)
    spin_split, rand_seed, cell_count = prepare_spin_state(film1, film2, args)
    
    # create folder
    filename='./figs_k{}/model_{}/shape_{}/size{}_Ms{}_Ax{}_Ku{}_dtime{}_split{}_seed{}_Layers{}/'.format(
                    args.krn, args.loss_type, args.mask, args.w, 
                    args.Ms, args.Ax, args.Ku, 
                    args.dtime, spin_split, rand_seed, args.layers
                    )
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    x_plot,y1_plot,y2_plot = [],[],[]

    Hext_range = np.linspace(1000,-1000,201)
    Hext_vec = np.array([np.cos(0.01), np.sin(0.01), 0.0])

    spin_mm = np.array([[[[1]]]])
    spin_un = np.array([[[[1]]]])

    hex_mm_plot, hex_un_plot = [], []
    hanis_mm_plot, hanis_un_plot = [], []
    hd_mm_plot, hd_un_plot = [], []
    heff_mm_plot, heff_un_plot = [], []
    

    hex_error_mae = []
    hanis_error_mae = []
    heff_error_mae = []
    hd_error_mae = []   
    trajectory_shift_mae = [] 
    coloc_rcd = []
    candidate_history = []

    full_fft = {'demag': [], 'anis': [], 'excha': [], 'exter': [], 'total': [], 
                'tau_hd': [], 'tau_he': [], 'tau_ha': [], 'tau_heff': [], 
                'iters': [], 'vortices': [], 'mz': [], 'time': []}
    full_un = {'demag': [], 'anis': [], 'excha': [], 'exter': [], 'total': [], 
               'tau_hd': [], 'tau_he': [], 'tau_ha': [], 'tau_heff': [], 
               'iters': [], 'vortices': [], 'mz': [], 'time': []}
    

    general_title_summary = (f"Film Layers: {args.layers} | Grid Size: {args.w}x{args.w} | Split: {spin_split} | Seed: {rand_seed} | Mask: {args.mask}\n"
                            f"Material Properties ── $M_s$: {args.Ms} emu/cc | $A_x$: {args.Ax} pJ/m | $K_u$: {args.Ku} $J/m^3$\n")

    save_path_summary = os.path.join(filename, "summary_plots")
    os.makedirs(save_path_summary, exist_ok=True) 

    save_iteration_plots = os.path.join(filename, "iteration_plots")
    os.makedirs(save_iteration_plots, exist_ok=True)

    physics_recorder = PhysicsRecorder()

    # Main loop
    for nloop, Hext_val in enumerate(Hext_range):
        save_path_iteration = os.path.join(filename, f"iteration_plots_{nloop}")
        os.makedirs(save_path_iteration, exist_ok=True)

        Hext = Hext_val * Hext_vec
        print('>>>>>loop: {} , Hext: {}'.format(nloop, Hext_val))

        spin0_mm = spin_mm
        spin0_un = spin_un

        # Update spin state
        start_fft = time.time()
        error1_rcd, itern1, hist_fft = update_spin_fft(film1, Hext, Hext_vec, cell_count, args)
        time_elapsed_fft = time.time() - start_fft
        start_un = time.time()
        error2_rcd, itern2, hist_un = update_spin_unet(film2, Hext, Hext_vec, cell_count, args)
        time_elapsed_un = time.time() - start_un

        # Extract final convergence values from the error logs
        final_err_fft = error1_rcd[-1] if len(error1_rcd) > 0 else 0.0
        final_err_un  = error2_rcd[-1] if len(error2_rcd) > 0 else 0.0

        predictor_dict = physics_recorder.predictor_dict()

        film1.GetEnergy_detailed(Hext=Hext)
        film2.GetEnergy_detailed(Hext=Hext)

        # Extract topological counts using winding density 
        spin_fft_tensor = film1.Spin.permute(3, 0, 1, 2)[:, :, :, 0].unsqueeze(0)
        spin_un_tensor  = film2.Spin.permute(3, 0, 1, 2)[:, :, :, 0].unsqueeze(0)
        _, winding_abs_fft, winding_sum_fft = winding_density(spin_fft_tensor)
        _, winding_abs_un, winding_sum_un = winding_density(spin_un_tensor)

        full_fft['iters'].append(itern1)
        full_fft['vortices'].append(vortex_count_fft)
        full_fft['mz'].append(hist_fft['mz'][-1])
        full_fft['time'].append(time_elapsed_fft)
        full_un['iters'].append(itern2)
        full_un['vortices'].append(vortex_count_un)
        full_un['mz'].append(hist_un['mz'][-1])
        full_un['time'].append(time_elapsed_un)

        # full_fft['demag'].append(hist_fft['e_demag'][-1])
        # full_fft['anis'].append(hist_fft['e_anis'][-1])
        # full_fft['excha'].append(hist_fft['e_excha'][-1])
        # full_fft['exter'].append(hist_fft['e_exter'][-1])
        # full_fft['total'].append(hist_fft['e_total'][-1])
        # full_un['demag'].append(hist_un['e_demag'][-1])
        # full_un['anis'].append(hist_un['e_anis'][-1])
        # full_un['excha'].append(hist_un['e_excha'][-1])
        # full_un['exter'].append(hist_un['e_exter'][-1])
        # full_un['total'].append(hist_un['e_total'][-1])

        full_fft['tau_hd'].append(hist_fft['tau_hd'][-1])
        full_fft['tau_he'].append(hist_fft['tau_he'][-1])
        full_fft['tau_ha'].append(hist_fft['tau_ha'][-1])
        full_fft['tau_heff'].append(hist_fft['tau_heff'][-1])
        full_un['tau_hd'].append(hist_un['tau_hd'][-1])
        full_un['tau_he'].append(hist_un['tau_he'][-1])
        full_un['tau_ha'].append(hist_un['tau_ha'][-1])
        full_un['tau_heff'].append(hist_un['tau_heff'][-1])

        # Calculate the spatial average magnitude across the grid sample
        hex_mm_plot.append(np.mean(np.linalg.norm(film1.He.detach().cpu().numpy(), axis=-1)))
        hanis_mm_plot.append(np.mean(np.linalg.norm(film1.Ha.detach().cpu().numpy(), axis=-1)))
        hd_mm_plot.append(np.mean(np.linalg.norm(film1.Hd.detach().cpu().numpy(), axis=-1))) 
        heff_mm_plot.append(np.mean(np.linalg.norm(film1.Heff.detach().cpu().numpy(), axis=-1))) 
        hex_un_plot.append(np.mean(np.linalg.norm(film2.He.detach().cpu().numpy(), axis=-1)))
        hanis_un_plot.append(np.mean(np.linalg.norm(film2.Ha.detach().cpu().numpy(), axis=-1)))             
        hd_un_plot.append(np.mean(np.linalg.norm(film2.Hd.detach().cpu().numpy(), axis=-1)))       
        heff_un_plot.append(np.mean(np.linalg.norm(film2.Heff.detach().cpu().numpy(), axis=-1)))

        # Calculate and record the Mean Absolute Error (MAE) between UNet and FFT fields
        hex_error_mae.append(np.mean(np.abs(film2.He.detach().cpu().numpy() - film1.He.detach().cpu().numpy())))
        hanis_error_mae.append(np.mean(np.abs(film2.Ha.detach().cpu().numpy() - film1.Ha.detach().cpu().numpy())))
        heff_error_mae.append(np.mean(np.abs(film2.Heff.detach().cpu().numpy() - film1.Heff.detach().cpu().numpy())))
        hd_error_mae.append(np.mean(np.abs(film2.Hd.detach().cpu().numpy() - film1.Hd.detach().cpu().numpy())))
        trajectory_shift_mae.append(np.mean(np.abs(film2.Spin.detach().cpu().numpy() - film1.Spin.detach().cpu().numpy())))

        # Record one converged M-H-step row. FFT quantities are predictors;
        # UNet quantities and FFT-vs-UNet errors are diagnostics/targets.
        physics_recorder.capture(mh_step=nloop, hext_scalar=Hext_val, hext_vector=Hext, film_fft=film1,
            film_unet=film2,
            fft_winding_abs=winding_abs_fft,
            fft_winding_sum=winding_sum_fft,
            unet_winding_abs=winding_abs_un,
            unet_winding_sum=winding_sum_un,
            fft_iterations=itern1,
            unet_iterations=itern2,
            fft_final_convergence_error=final_err_fft,
            unet_final_convergence_error=final_err_un,
            fft_runtime_seconds=time_elapsed_fft,
            unet_runtime_seconds=time_elapsed_un,
            cell_count=cell_count,
        )

        spin_mm = film1.Spin.detach().cpu().numpy()
        spin_un = film2.Spin.detach().cpu().numpy()
        Hd_mm = film1.Hd.detach().cpu().numpy()
        Hd_un = film2.Hd.detach().cpu().numpy()
        
        #MH loop data
        x_plot.append(Hext_val)
        y1_plot.append( np.dot(spin_mm.sum(axis=(0,1,2)), Hext_vec)/ cell_count )
        y2_plot.append( np.dot(spin_un.sum(axis=(0,1,2)), Hext_vec)/ cell_count )

        general_title_iteration = (f"Film Layers: {args.layers} | Grid Size: {args.w}x{args.w} | Split: {spin_split} | Seed: {rand_seed} | Mask: {args.mask}\n"
                    f"Material Properties ── $M_s$: {args.Ms} emu/cc | $A_x$: {args.Ax} pJ/m | $K_u$: {args.Ku} $J/m^3$\n"
                f"Loop: {nloop} | $H_{{ext}}$ = {Hext_val:.1f} Oe | Iterations: mm [{itern1}] | un [{itern2}] | Error: mm\n ")

        # Plot results
        plot_results(nloop=nloop, spin_mm=spin_mm, spin_un=spin_un, itern1=itern1, itern2=itern2, Hd_mm=Hd_mm, Hd_un=Hd_un, 
                     x_plot=x_plot, y1_plot=y1_plot, y2_plot=y2_plot, Hext_range=Hext_range, error1_rcd=error1_rcd, error2_rcd=error2_rcd, save_path=save_path_iteration, general_title_iteration=general_title_iteration)

        plot_iteration_panel(general_title_iteration=general_title_iteration, hist_fft=hist_fft, hist_un=hist_un, save_path=save_path_iteration, plot_type="iteration_torques", specific_title="Torque Evolution During Relaxation",
            panel_specs=[{'key': 'tau_hd',   'title': 'Demagnetizing Torque', 'ylabel': r'$|m \times H|$'},
                         {'key': 'tau_he',   'title': 'Exchange Torque',      'ylabel': r'$|m \times H|$'},
                         {'key': 'tau_ha',   'title': 'Anisotropy Torque',    'ylabel': r'$|m \times H|$'},
                         {'key': 'tau_heff', 'title': 'Effective Torque',     'ylabel': r'$|m \times H|$'}])
# add y label to go with units of torque, which is in A/m^2, but we can just use the magnitude for now?
        plot_iteration_panel(general_title_iteration=general_title_iteration, hist_fft=hist_fft, hist_un=hist_un, save_path=save_path_iteration, plot_type="iteration_alignment", specific_title="Field Alignment During Relaxation",
            panel_specs=[{'key': 'align_hd',   'title': 'Demagnetizing Field', 'ylabel': r'$\langle \cos(\theta)\rangle$', 'ylim': (-1.05, 1.05)},
                         {'key': 'align_he',   'title': 'Exchange Field',      'ylabel': r'$\langle \cos(\theta)\rangle$', 'ylim': (-1.05, 1.05)},
                         {'key': 'align_ha',   'title': 'Anisotropy Field',    'ylabel': r'$\langle \cos(\theta)\rangle$', 'ylim': (-1.05, 1.05)},
                         {'key': 'align_heff', 'title': 'Effective Field',     'ylabel': r'$\langle \cos(\theta)\rangle$', 'ylim': (-1.05, 1.05)}])

        plot_iteration_domain_walls(general_title_iteration, save_path_iteration, film1, film2, nloop)       
        plot_iteration_fields(general_title_iteration, hist_fft, hist_un, save_path_iteration, nloop)
        plot_iteration_winding_density(general_title_iteration, film1, film2, save_path_iteration, nloop)
        # plot_iteration_energy(general_title_iteration, hist_fft, hist_un, save_path_iteration, nloop)
        plot_iteration_torque(general_title_iteration, save_path_iteration, hist_fft, hist_un, nloop)
        plot_iteration_alignment(general_title_iteration, hist_fft, hist_un, save_path_iteration, nloop)
        coloc = plot_hd_error_vs_vortex_cores(general_title_iteration, save_path_iteration,film1, film2, nloop, core_threshold=0.5)
        coloc_rcd.append(coloc)  # accumulate for a summary plot at the end 
        candidate_history.append(extract_candidate_predictors(film1))
        
        # Save MH data
        np.save(filename + "Hext_array", x_plot)
        np.save(filename + "Mext_array_mm", y1_plot)
        np.save(filename + "Mext_array_un", y2_plot)
        np.save(filename + "instantaneous_hd_mae", hd_error_mae)
        np.save(filename + "trajectory_shift_mae", trajectory_shift_mae)

        # Save Mr
        if Hext_val == 0:
            np.save(filename + "Mr_spin_mm", spin_mm)
            np.save(filename + "Mr_spin_un", spin_un)

        # Save Hc
        Mi = spin0_mm[:,:,:,0].sum()
        Mj = spin_mm[:,:,:,0].sum()
        if Mi > 0 and Mj <= 0:
            np.save(filename + "Hc{}_spin_mm".format(nloop-1), spin0_mm)
            np.save(filename + "Hc{}_spin_mm".format(nloop), spin_mm)

        Mi = spin0_un[:,:,:,0].sum()
        Mj = spin_un[:,:,:,0].sum()
        if Mi > 0 and Mj <= 0:
            np.save(filename + "Hc{}_spin_un".format(nloop-1), spin0_un)
            np.save(filename + "Hc{}_spin_un".format(nloop), spin_un)

    physics_csv = os.path.join(filename, "physics_snapshots.csv")
    physics_df = physics_recorder.save_csv(physics_csv)
    print(f"Saved {len(physics_df)} converged physics snapshots to {physics_csv}")

    plot_full_energy_summary(general_title_summary, save_path_summary, full_fft, full_un, Hext_range)
    plot_performance_summary(general_title_summary, save_path_summary, full_fft, full_un, Hext_range)
    plot_error_summary(general_title_summary, save_path_summary, Hext_range, hd_error_mae, trajectory_shift_mae, 
                       hex_error_mae, hanis_error_mae)
    plot_fields_summary(general_title_summary, save_path_summary, Hext_range, hex_mm_plot, hex_un_plot, hanis_mm_plot, hanis_un_plot, 
                        hd_mm_plot, hd_un_plot, heff_mm_plot, heff_un_plot)
    plot_error_correlations(general_title_summary, save_path_summary, hd_error_mae, hex_error_mae, hanis_error_mae, 
                            trajectory_shift_mae, Hext_range=Hext_range)

    # Part 1 output for the later leading-indicator analyzer.
    # Do not include UNet errors or topology labels in predictor_dict.
    predictor_dict = physics_recorder.predictor_dict()
    trajectory_error = physics_df["spin_mae"].to_numpy(dtype=float)
    transition_signal = physics_df["fft_winding_abs"].to_numpy(dtype=float)

    events, kinds = detect_transition_events(transition_signal, event_type="both")
    for event, kind in zip(events, kinds):
        print(
            f"Transition event at step {event}, "
            f"Hext={physics_df.loc[event, 'hext_scalar']:.1f} Oe, kind={kind}"
        )
    
    plot_error_vs_transition_proximity(general_title_summary, save_path_summary, trajectory_shift_mae, full_fft["vortices"], event_type='both')
    plot_error_vs_transition_proximity(general_title_summary, save_path_summary, trajectory_shift_mae, full_fft["vortices"], event_type='nucleation')
    plot_error_vs_transition_proximity(general_title_summary, save_path_summary, trajectory_shift_mae, full_fft["vortices"], event_type='annihilation')
    plot_colocalization_summary(general_title_summary, save_path_summary, Hext_range, coloc_rcd)
    # plot_temporal_variance_vs_error(general_title_summary, save_path_summary, Hext_range, temporal_var_rcd, trajectory_error)