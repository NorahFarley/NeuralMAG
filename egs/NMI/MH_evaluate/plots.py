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

from libs.misc import Culist, MaskTp, spin_prepare, winding_density
import libs.MAG2305 as MAG2305
from libs.Unet import UNet

def plot_iteration_domain_walls(film1, film2, base_path, nloop, Hext_val, args, spin_split, rand_seed, itern1, itern2, err_fft, err_un):
    """
    Generates a 2x2 multi-panel spatial and quantitative analysis sheet comparing 
    domain wall layouts with locked color scales, 1D line cuts, and metadata.
    """
    # Automatically generates a separate, dedicated folder for this plot type
    dw_folder = os.path.join(base_path, "domain_walls_spatial")
    os.makedirs(dw_folder, exist_ok=True)
    
    # Isolate domain wall profiles by computing local exchange field vector magnitudes (Layer 0)
    dw_mm = np.linalg.norm(film1.He.detach().cpu().numpy()[:, :, 0, :], axis=-1)
    dw_un = np.linalg.norm(film2.He.detach().cpu().numpy()[:, :, 0, :], axis=-1)
    dw_diff = np.abs(dw_mm - dw_un)
    
    # Extract a 1D cross-section cut across the center row of the film
    mid_row = dw_mm.shape[0] // 2
    line_mm = dw_mm[mid_row, :]
    line_un = dw_un[mid_row, :]
    
    fig, axs = plt.subplots(2, 2, figsize=(15, 13))
    
    title_text = (f"Film Layers: {args.layers} | Grid Size: {args.w}x{args.w} | Split: {spin_split} | Seed: {rand_seed} | Mask: {args.mask}\n"
                  f"Material Properties ── $M_s$: {args.Ms} emu/cc | $A_x$: {args.Ax} pJ/m | $K_u$: {args.Ku} $J/m^3$\n")
    
    fig.suptitle("Domain Wall Position Graphs\n\n" + title_text, fontsize=13, fontweight="bold")
    

    # Find the global maximum exchange intensity between both models
    global_vmax = max(np.max(dw_mm), np.max(dw_un))
    global_vmin = 0.0
    
    # Panel 1: FFT Ground Truth Heatmap
    im0 = axs[0, 0].imshow(dw_mm, cmap='viridis', origin='lower', vmin=global_vmin, vmax=global_vmax)
    axs[0, 0].set_title('FFT Solver (mm) Domain Wall Spatial Map', fontsize=11, fontweight='bold')
    axs[0, 0].set_xlabel('x [nm]', fontsize=9)
    axs[0, 0].set_ylabel('y [nm]', fontsize=9)
    fig.colorbar(im0, ax=axs[0, 0], label='Exchange Intensity [Oe]')
    
    # Panel 2: UNet Prediction Heatmap (Locked to identical color scales as FFT)
    im1 = axs[0, 1].imshow(dw_un, cmap='viridis', origin='lower', vmin=global_vmin, vmax=global_vmax)
    axs[0, 1].set_title('UNet Model (un) Domain Wall Spatial Map', fontsize=11, fontweight='bold')
    axs[0, 1].set_xlabel('x [nm]', fontsize=9)
    axs[0, 1].set_ylabel('y [nm]', fontsize=9)
    fig.colorbar(im1, ax=axs[0, 1], label='Exchange Intensity [Oe]')
    
    # Panel 3: Spatial Difference Heatmap
    im2 = axs[1, 0].imshow(dw_diff, cmap='hot', origin='lower', vmin=0.0)
    axs[1, 0].set_title('Domain Tracking Spatial Difference', fontsize=11, fontweight='bold')
    axs[1, 0].set_xlabel('x [nm]', fontsize=9)
    axs[1, 0].set_ylabel('y [nm]', fontsize=9)
    fig.colorbar(im2, ax=axs[1, 0], label='Absolute Deviation [Oe]')
    
    # Panel 4: 1D Line Cut Cross-Section Graph with Tailored Padding Limits
    axs[1, 1].plot(line_mm, color='blue', lw=2, linestyle='-', label='FFT Solver Cut')
    axs[1, 1].plot(line_un, color='red', lw=2, linestyle='-', label='UNet Model Cut')
    
    # padding for the line graph axis limits
    line_max = max(np.max(line_mm), np.max(line_un))
    line_min = min(np.min(line_mm), np.min(line_un))
    line_range = line_max - line_min if line_max != line_min else 1.0
    
    line_ymax_padded = line_max + (line_range * 0.05)
    line_ymin_padded = -0.05 * line_max if line_min == 0.0 else line_min - (line_range * 0.05)
    line_xmax_padded = len(line_mm) * 1.05
    
    axs[1, 1].set_title(f'Domain Wall Profile Cut (Row Y = {mid_row})', fontsize=11, fontweight='bold')
    axs[1, 1].set_xlabel('Spatial Coordinate X [Cell Index]', fontsize=9)
    axs[1, 1].set_ylabel('Local Exchange Intensity [Oe]', fontsize=9)
    axs[1, 1].set_xlim(0, line_xmax_padded)
    axs[1, 1].set_ylim(line_ymin_padded, line_ymax_padded)
    axs[1, 1].grid(True, linestyle='--', alpha=0.4)
    axs[1, 1].legend(loc='upper right', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(dw_folder, f'spatial_dw_loop_{nloop}.png'), dpi=150)
    plt.close()

def plot_iteration_fields(hist_fft, hist_un, base_path, nloop, Hext_val, args, 
                         spin_split, rand_seed, itern1, itern2, err_fft, err_un):
    """
    Generates a 2x2 multi-panel line graph mapping every field variable 
    trajectory iteration-by-iteration for explicit path tracking.
    """

    # Create the dedicated subfolder
    iter_folder = os.path.join(base_path, "iteration_error_plots")
    os.makedirs(iter_folder, exist_ok=True)
    fig, axs = plt.subplots(3, 2, figsize=(15, 15))
    
    title_text = (f"Film Layers: {args.layers} | Grid Size: {args.w}x{args.w} | Split: {spin_split} | Seed: {rand_seed} | Mask: {args.mask}\n"
                  f"Material Properties ── $M_s$: {args.Ms} emu/cc | $A_x$: {args.Ax} pJ/m | $K_u$: {args.Ku} $J/m^3$\n"
                  f"Target Threshold ($\Delta m_{{min}}$): {args.error_min:.2e}\n"
                  f"FFT Steps: {itern1} (Final Err: {err_fft:.2e}) | UNet Steps: {itern2} (Final Err: {err_un:.2e})\n")
    # fig.suptitle(title_text, fontsize=13, fontweight='bold')
    # fig.suptitle(f'Field Component Convergence Trajectories | Hext = {Hext_val:.1f} Oe (Loop {nloop})', fontsize=14, fontweight='bold')
    fig.suptitle("Field Component Convergence Trajectories\n\n"+ title_text, fontsize=13, fontweight="bold")

    # field axis scaling (Applies to Panels 1-4)
    all_field_values = (hist_fft['hd'] + hist_fft['ha'] + hist_fft['he'] + hist_fft['heff'] +
                        hist_un['hd'] + hist_un['ha'] + hist_un['he'] + hist_un['heff'])
    
    min_field = min(all_field_values)
    max_field = max(all_field_values)
    field_range = max_field - min_field if max_field != min_field else 1.0
    
    global_ymax = max_field + (field_range * 0.05)
    global_ymin = -0.05 * max_field if min_field == 0.0 and max_field != 0.0 else min_field - (field_range * 0.05)
    
    global_xmax = max(len(hist_fft['hd']), len(hist_un['hd']))
    global_xmax_padded = global_xmax * 1.05

    m_title = f'Magnetization State ($M_{{ext}}/M_s$)\nFinal ── FFT: {hist_fft["m"][-1]:.3f} | UNet: {hist_un["m"][-1]:.3f}'
    mz_title = f'Out-of-Plane Component ($|M_z|$)\nFinal ── FFT: {hist_fft["mz"][-1]:.3f} | UNet: {hist_un["mz"][-1]:.3f}'
    
    # Map tracking parameters to grid slots
    plot_map = [('hd', 'Demagnetizing Field ($H_{demag}$)', 'Mean Demagnetizing Field [Oe]', axs[0, 0], 'field'),
                ('ha', 'Anisotropy Field ($H_{anis}$)', 'Mean Anisotropy Field [Oe]', axs[0, 1], 'field'),
                ('he', 'Exchange Field ($H_{ex}$)', 'Mean Exchange Field [Oe]', axs[1, 0], 'field'),
                ('heff', 'Total Effective Field ($H_{eff}$)', 'Mean Effective Field [Oe]', axs[1, 1], 'field'),
                ('m', m_title, 'Projected Magnetization $M_{ext}/M_s$', axs[2, 0], 'm_axis'),
                ('mz', mz_title, 'Mean Out-of-Plane Magnetization Magnitude $|M_z|$', axs[2, 1], 'mz_axis')]
    
    for key, panel_title, y_label, ax, scale_type in plot_map:
        ax.plot(hist_fft[key], color='blue', lw=2, linestyle='-', label='FFT Solver Path')
        ax.plot(hist_un[key], color='red', lw=2, linestyle='-', label='UNet Model Path')
        
        ax.set_title(panel_title, fontsize=11, fontweight='bold')
        ax.set_xlabel('Internal Solver Step (Iteration)', fontsize=9)
        ax.set_ylabel(y_label, fontsize=9)
        ax.set_xlim(0, global_xmax_padded)
        
        if scale_type == 'field':
            ax.set_ylim(global_ymin, global_ymax)
        elif scale_type == 'm_axis':
            ax.set_ylim(-1.1, 1.1)
        elif scale_type == 'mz_axis':
            ax.set_ylim(-0.05, 1.1)
        
    plt.tight_layout()
    plt.savefig(os.path.join(iter_folder, f'iteration_trajectory_loop_{nloop}.png'), dpi=150)
    plt.close()

def plot_iteration_winding_density(film1, film2, base_path, nloop, Hext_val, args, spin_split, rand_seed, itern1, itern2, err_fft, err_un):
    """
    Generates a 2x2 multi-panel spatial and quantitative topological chart 
    comparing vortex core winding densities with symmetric scales.
    """
    # Create a dedicated separate subfolder for this plot classification
    topo_folder = os.path.join(base_path, "winding_density_spatial")
    os.makedirs(topo_folder, exist_ok=True)

    # Reshape spin arrays to channel-first format with batch dimension [1, 3, W, W] for layer 0
    spin_fft_tensor = film1.Spin.permute(3, 0, 1, 2)[:, :, :, 0].unsqueeze(0)
    spin_un_tensor  = film2.Spin.permute(3, 0, 1, 2)[:, :, :, 0].unsqueeze(0)
    
    # Run the built-in micromagnetic winding density analyzer
    topo_fft_raw, winding_abs_fft, _ = winding_density(spin_fft_tensor)
    topo_un_raw,  winding_abs_un,  _ = winding_density(spin_un_tensor)
    
    # Squeeze down to standard 2D numpy matrices for plotting
    topo_fft = topo_fft_raw.squeeze().detach().cpu().numpy()
    topo_un  = topo_un_raw.squeeze().detach().cpu().numpy()
    topo_diff = np.abs(topo_fft - topo_un)
    
    # Extract 1D cross-section data through the horizontal center row
    mid_row = topo_fft.shape[0] // 2
    line_fft = topo_fft[mid_row, :]
    line_un  = topo_un[mid_row, :]
    
    fig, axs = plt.subplots(2, 2, figsize=(15, 13))
    
    title_text = (f"Film Layers: {args.layers} | Grid Size: {args.w}x{args.w} | Split: {spin_split} | Seed: {rand_seed} | Mask: {args.mask}\n"
                  f"Material Properties ── $M_s$: {args.Ms} emu/cc | $A_x$: {args.Ax} pJ/m | $K_u$: {args.Ku} $J/m^3$\n"
                  f"Loop: {nloop} | $H_{{ext}}$ = {Hext_val:.1f} Oe | Target Threshold ($\Delta m_{{min}}$): {args.error_min:.2e}\n")
    fig.suptitle(title_text, fontsize=13, fontweight='bold')
    
    # Dynamically scales to the highest peak but keeps the bounds symmetrical around zero
    max_charge = max(np.max(np.abs(topo_fft)), np.max(np.abs(topo_un)))
    global_vmax = max_charge if max_charge > 1e-8 else 1.0
    global_vmin = -global_vmax

    # linthresh controls the linear region around 0. Anything smaller than this stays linear.
    norm = colors.SymLogNorm(linthresh=max_charge*0.02, vmin=global_vmin, vmax=global_vmax, base=10)

    # Panel 1: FFT Ground Truth Topological Heatmap
    im0 = axs[0, 0].imshow(topo_fft, cmap='PuOr', origin='lower', norm=norm)
    axs[0, 0].set_title(f'FFT Solver Topological Charge Map\nTotal Absolute Vortices: {winding_abs_fft:.1f}', fontsize=11, fontweight='bold')
    axs[0, 0].set_xlabel('x [nm]', fontsize=9)
    axs[0, 0].set_ylabel('y [nm]', fontsize=9)
    fig.colorbar(im0, ax=axs[0, 0], label='Local Topological Charge Density')
    
    # Panel 2: UNet Framework Topological Heatmap (Locked to identical bounds)
    im1 = axs[0, 1].imshow(topo_un, cmap='PuOr', origin='lower', norm=norm)
    axs[0, 1].set_title(f'UNet Model Topological Charge Map\nTotal Absolute Vortices: {winding_abs_un:.1f}', fontsize=11, fontweight='bold')
    axs[0, 1].set_xlabel('x [nm]', fontsize=9)
    axs[0, 1].set_ylabel('y [nm]', fontsize=9)
    fig.colorbar(im1, ax=axs[0, 1], label='Local Topological Charge Density')
    
    # Panel 3: Spatial Tracking Difference Map
    im2 = axs[1, 0].imshow(topo_diff, cmap='hot', origin='lower', vmin=0.0)
    axs[1, 0].set_title('Absolute Tracking Topological Difference', fontsize=11, fontweight='bold')
    axs[1, 0].set_xlabel('x [nm]', fontsize=9)
    axs[1, 0].set_ylabel('y [nm]', fontsize=9)
    fig.colorbar(im2, ax=axs[1, 0], label='Absolute Deviation')
    
    #  Panel 4: 1D Line Cut Cross-Section Graph with Proportional 5% Margin Padding
    axs[1, 1].plot(line_fft, color='blue', lw=2, linestyle='-', label='FFT Core Cut')
    axs[1, 1].plot(line_un, color='red', lw=2, linestyle='-', label='UNet Core Cut')
    
    # Calculate tailored 5% padding configuration limits
    line_max = max(np.max(line_fft), np.max(line_un))
    line_min = min(np.min(line_fft), np.min(line_un))
    line_range = line_max - line_min if line_max != line_min else 1.0
    
    line_ymax_padded = line_max + (line_range * 0.05)
    line_ymin_padded = line_min - (line_range * 0.05)
    line_xmax_padded = len(line_fft) * 1.05
    
    axs[1, 1].set_title(f'Topological Core Profile Cut (Row Y = {mid_row})', fontsize=11, fontweight='bold')
    axs[1, 1].set_xlabel('Spatial Coordinate X [Cell Index]', fontsize=9)
    axs[1, 1].set_ylabel('Topological Winding Value', fontsize=9)
    axs[1, 1].set_xlim(0, line_xmax_padded)
    axs[1, 1].set_ylim(line_ymin_padded, line_ymax_padded)
    axs[1, 1].grid(True, linestyle='--', alpha=0.4)
    axs[1, 1].legend(loc='upper right', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(topo_folder, f'spatial_topology_loop_{nloop}.png'), dpi=150)
    plt.close()

def plot_iteration_energy(hist_fft, hist_un, base_path, nloop, Hext_val, args, spin_split, rand_seed, itern1, itern2, err_fft, err_un):
    """
    Generates a 2x2 multi-panel line graph mapping individual energy component 
    relaxation curves iteration-by-iteration with tailored 5% boundary padding.
    """
    iter_energy_folder = os.path.join(base_path, "iteration_energy")
    os.makedirs(iter_energy_folder, exist_ok=True)
    
    fig, axs = plt.subplots(2, 2, figsize=(15, 11))
    
    title_text = (f"Film Layers: {args.layers} | Grid Size: {args.w}x{args.w} | Split: {spin_split} | Seed: {rand_seed} | Mask: {args.mask}\n"
                  f"Material Properties ── $M_s$: {args.Ms} emu/cc | $A_x$: {args.Ax} pJ/m | $K_u$: {args.Ku} $J/m^3$\n"
                  f"Loop: {nloop} | $H_{{ext}}$ = {Hext_val:.1f} Oe | Target Threshold ($\Delta m_{{min}}$): {args.error_min:.2e}\n")
    fig.suptitle(title_text, fontsize=13, fontweight='bold')
    
    global_xmax_padded = max(len(hist_fft['e_demag']), len(hist_un['e_demag'])) * 1.05
    
    plot_map = [('e_demag', 'Demagnetizing Energy ($E_{demag}$)', 'Energy [Joules]', axs[0, 0]),
                ('e_anis', 'Anisotropy Energy ($E_{anis}$)', 'Energy [Joules]', axs[0, 1]),
                ('e_excha', 'Exchange Energy ($E_{excha}$)', 'Energy [Joules]', axs[1, 0]),
                ('e_exter', 'Exter Energy ($E_{exter}$)', 'Energy [Joules]', axs[1, 1])]
    
    for key, panel_title, y_label, ax in plot_map:
        ax.plot(hist_fft[key], color='blue', lw=2, linestyle='-', label='FFT Solver Path')
        ax.plot(hist_un[key], color='red', lw=2, linestyle='-', label='UNet Model Path')
        
        ax.set_title(panel_title, fontsize=11, fontweight='bold')
        ax.set_xlabel('Internal Solver Step (Iteration)', fontsize=10)
        ax.set_ylabel(y_label, fontsize=10)
        
        # Calculate Proportional 5% padding dynamically per panel to handle scale differences
        combined_vals = hist_fft[key] + hist_un[key]
        if len(combined_vals) > 0:
            max_v, min_v = max(combined_vals), min(combined_vals)
            v_range = max_v - min_v if max_v != min_v else 1.0
            ymax = max_v + (v_range * 0.05)
            ymin = -0.05 * max_v if min_v == 0.0 and max_v != 0.0 else min_v - (v_range * 0.05)
            ax.set_ylim(ymin, ymax)
            
        ax.set_xlim(0, global_xmax_padded)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.legend(loc='upper right', fontsize=9)
        
    plt.tight_layout()
    plt.savefig(os.path.join(iter_energy_folder, f'iteration_energy_loop_{nloop}.png'), dpi=150)
    plt.close()

    fig_tot, ax_tot = plt.subplots(figsize=(9, 6))
    fig_tot.suptitle(title_text, fontsize=11, fontweight='bold')
    
    ax_tot.plot(hist_fft['e_total'], color='blue', lw=2.5, linestyle='-', label='FFT Solver Path')
    ax_tot.plot(hist_un['e_total'], color='red', lw=2.5, linestyle='-', label='UNet Model Path')
    ax_tot.set_title('Total Effective Field Energy ($E_{total}$)', fontsize=12, fontweight='bold')
    ax_tot.set_xlabel('Internal Solver Step (Iteration)', fontsize=11)
    ax_tot.set_ylabel('Total Energy [Joules]', fontsize=11) 
    
    combined_tot = hist_fft['e_total'] + hist_un['e_total']
    if len(combined_tot) > 0:
        max_v, min_v = max(combined_tot), min(combined_tot)
        v_range = max_v - min_v if max_v != min_v else 1.0
        ax_tot.set_ylim(min_v - (v_range * 0.05), max_v + (v_range * 0.05))
        
    ax_tot.set_xlim(0, global_xmax_padded)
    ax_tot.grid(True, linestyle='--', alpha=0.4)
    ax_tot.legend(loc='upper right', fontsize=10)
    
    plt.tight_layout()
    fig_tot.subplots_adjust(top=0.85)
    plt.savefig(os.path.join(iter_energy_folder, f'iteration_total_energy_loop_{nloop}.png'), dpi=150)
    plt.close()

def plot_full_energy_summary(full_data_fft, full_data_un, Hext_range, base_path, args, spin_split, rand_seed):
    """
    Generates a final 2x2 multi-panel graph charting equilibrium energy components 
    across the entire completed external field sweep loop range.
    """
    full_energy_folder = os.path.join(base_path, "summary_plots")
    os.makedirs(full_energy_folder, exist_ok=True)
    
    fig, axs = plt.subplots(2, 2, figsize=(15, 11))
    
    title_text = (f"Film Layers: {args.layers} | Grid Size: {args.w}x{args.w} | Split: {spin_split} | Seed: {rand_seed} | Mask: {args.mask}\n"
                  f"Material Properties ── $M_s$: {args.Ms} emu/cc | $A_x$: {args.Ax} pJ/m | $K_u$: {args.Ku} $J/m^3$\n")
    fig.suptitle(title_text, fontsize=13, fontweight='bold')
    
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
    fig_tot.suptitle(f"Total System Energy Profile Across M-H Sweep\n{title_text}", fontsize=11, fontweight='bold')
    
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

def plot_performance_summary(performance_fft, performance_un, Hext_range, base_path, args, spin_split, rand_seed):
    """
    Generates a final 2x2 multi-panel chart compiling global optimization metrics,
    topological structures, and execution times across the full Hext range.
    """
    performance_folder = os.path.join(base_path, "summary_plots")
    os.makedirs(performance_folder, exist_ok=True)
    
    fig, axs = plt.subplots(2, 2, figsize=(15, 11))
    
    title_text = (f"Film Layers: {args.layers} | Grid Size: {args.w}x{args.w} | Split: {spin_split} | Seed: {rand_seed} | Mask: {args.mask}\n"
                  f"Material Properties ── $M_s$: {args.Ms} emu/cc | $A_x$: {args.Ax} pJ/m | $K_u$: {args.Ku} $J/m^3$\n")
    fig.suptitle(title_text, fontsize=13, fontweight='bold')
    
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

def plot_error_summary(Hext_range, inst_hd_mae, traj_shift_mae, hex_err_mae, hanis_err_mae, base_path, args, spin_split, rand_seed):
    """
    Generates a final 2x2 multi-panel master report compiling all local field approximations,
    historical path tracking drift, and intrinsic field deviations across the Hext sweep.
    """
    error_summary_folder = os.path.join(base_path, "summary_plots")
    os.makedirs(error_summary_folder, exist_ok=True)
    
    print("Generating comprehensive 4-panel error tracking analysis...")
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    
    title_text = (f"Film Layers: {args.layers} | Grid Size: {args.w}x{args.w} | Split: {spin_split} | Seed: {rand_seed} | Mask: {args.mask}\n"
                  f"Material Properties ── $M_s$: {args.Ms} emu/cc | $A_x$: {args.Ax} pJ/m | $K_u$: {args.Ku} $J/m^3$\n")
    fig.suptitle(title_text, fontsize=13, fontweight='bold')
    
    # Calculate uniform X-axis limits with standard 5% padding while maintaining the reversed sweep
    max_h, min_h = max(Hext_range), min(Hext_range)
    h_range = max_h - min_h if max_h != min_h else 1.0
    xmax_padded = max_h + (h_range * 0.05)
    xmin_padded = min_h - (h_range * 0.05)
    
    # Structural Mapping Matrix to cycle configurations cleanly
    plot_map = [(inst_hd_mae, 'darkorange', 'Total Unet Model $H_{demag}$ Approximation Error', '$H_{demag}$ Field Prediction Error', 'Instantaneous $H_{demag}$ MAE [Oe]', axs[0, 0]),
                (traj_shift_mae, 'crimson', 'Magnetization Trajectory Drift (Accumulated Error)', 'Predicted Magnetization Error', 'Cumulative Spin $\\vec{m}$ MAE', axs[0, 1]),
                (hex_err_mae, 'purple', 'Total Exchange Field ($H_{ex}$) Error Accumulation', '$H_{ex}$ Prediction Error', 'Exchange Field MAE [Oe]', axs[1, 0]),
                (hanis_err_mae, 'teal', 'Total Anisotropy Field ($H_{anis}$) Error Accumulation', '$H_{anis}$ Prediction Error', 'Anisotropy Field MAE [Oe]', axs[1, 1])]
    
    for data, color, subtitle, label, y_label, ax in plot_map:
        ax.plot(Hext_range, data, color=color, lw=2, linestyle='-', label=label)
        ax.set_title(subtitle, fontsize=11, fontweight='bold')
        ax.set_xlabel('External Magnetic Field $H_{ext}$ [Oe]', fontsize=10)
        ax.set_ylabel(y_label, fontsize=10)
        
        max_v, min_v = max(data), min(data)
        v_range = max_v - min_v if max_v != min_v else 1.0
        ymax = max_v + (v_range * 0.05)
        ymin = -0.05 * max_v if min_v == 0.0 and max_v != 0.0 else min_v - (v_range * 0.05)
        
        # Apply bounds and format canvas grids
        ax.set_xlim(xmax_padded, xmin_padded) 
        ax.set_ylim(ymin, ymax)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.legend(loc='upper right', fontsize=9)
        
    plt.tight_layout()
    plt.savefig(os.path.join(error_summary_folder, 'comprehensive_error_analysis.png'), dpi=300)
    plt.close()

def plot_fields_summary(Hext_range, hex_mm, hex_un, hanis_mm, hanis_un, hd_mm, hd_un, heff_mm, heff_un, base_path, args, spin_split, rand_seed):
    """
    Generates a 2x2 panel graph chart recording equilibrium 
    magnitudes of all internal fields across the completed Hext sweep range.
    """
    fields_summary_folder = os.path.join(base_path, "summary_plots")
    os.makedirs(fields_summary_folder, exist_ok=True)
    
    print("Generating final 4-panel physical field summary plot...")
    fig, axs = plt.subplots(2, 2, figsize=(15, 11))
    
    title_text = (f"Film Layers: {args.layers} | Grid Size: {args.w}x{args.w} | Split: {spin_split} | Seed: {rand_seed} | Mask: {args.mask}\n"
                  f"Material Properties ── $M_s$: {args.Ms} emu/cc | $A_x$: {args.Ax} pJ/m | $K_u$: {args.Ku} $J/m^3$\n")
    fig.suptitle(title_text, fontsize=13, fontweight='bold')
    
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

def plot_error_correlations(hd_error, hex_error, hanis_error, traj_error, base_path, args, spin_split, rand_seed, Hext_range):
    """
    Generates scatter plots comparing each internal field error to the
    trajectory error over the entire hysteresis sweep.
    """
    fields_summary_folder = os.path.join(base_path, "summary_plots")
    os.makedirs(fields_summary_folder, exist_ok=True)
        
    title_text = (f"Layers: {args.layers} | Grid Size: {args.w}x{args.w} | Split: {spin_split} | Seed: {rand_seed} | Mask: {args.mask}\n"
                  f"Material Properties ── $M_s$: {args.Ms} emu/cc | $A_x$: {args.Ax} pJ/m | $K_u$: {args.Ku} $J/m^3$\n")

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
    fig.suptitle("Correlation Between Internal Field Errors and Trajectory Error\n\n" + title_text, 
                 fontsize=13, fontweight='bold')

    plt.savefig(os.path.join(fields_summary_folder, "error_correlations.png"), dpi=300, bbox_inches="tight")
    plt.close()

def plot_iteration_torque(hist_fft, hist_un, base_path, nloop, Hext_val, args, spin_split, rand_seed, 
                          itern1, itern2, err_fft, err_un):
    """
    Plot evolution of the four torque magnitudes during relaxation.

    Torque = |m x H|

    These plots are often much more physically meaningful than the
    field magnitudes because the LLG equation evolves according to
    torque rather than field alone.
    """

    folder = os.path.join(base_path, "iteration_torques")
    os.makedirs(folder, exist_ok=True)

    fig, axs = plt.subplots(2,2, figsize=(14,10))

    title = (f"Torque Evolution During Relaxation\n\n"
             f"Grid: {args.w}x{args.w} | Layers: {args.layers} | Split={spin_split} | Seed={rand_seed}\n"
             f"Mask={args.mask} | Ms: {args.Ms} | Ax: {args.Ax} | Ku: {args.Ku}\n"
             f"Hext={Hext_val:.1f} Oe")

    fig.suptitle(title, fontsize=13, fontweight='bold')

    plots = [("tau_hd", "Demagnetizing Torque", axs[0,0]), 
             ("tau_he", "Exchange Torque", axs[0,1]), 
             ("tau_ha", "Anisotropy Torque", axs[1,0]), 
             ("tau_heff", "Effective Torque", axs[1,1])]

    for key, title, ax in plots:
        ax.plot(hist_fft[key], color="blue", linewidth=2, label="FFT")
        ax.plot(hist_un[key], color="red", linewidth=2, label="UNet")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Iteration")
        ax.set_ylabel(r"$|m \times H|$")
        ax.grid(alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.savefig(
        os.path.join(folder, f"torque_iteration_loop_{nloop}.png"), dpi=200)
    plt.close()

def plot_iteration_alignment(hist_fft, hist_un, base_path, nloop, Hext_val, args, spin_split, 
                             rand_seed, itern1, itern2, err_fft, err_un):
    """
    Mean alignment between magnetization and each internal field.

    +1 = parallel
     0 = perpendicular
    -1 = antiparallel
    """

    folder = os.path.join(base_path, "iteration_alignment")
    os.makedirs(folder, exist_ok=True)

    fig, axs = plt.subplots(2,2, figsize=(14,10))

    fig.suptitle(f"Field Alignment During Relaxation\n\n"
                 f"Hext={Hext_val:.1f} Oe", fontsize=13, fontweight="bold")

    plot_map = [("align_hd", "Demagnetizing Field", axs[0,0]),
                ("align_he", "Exchange Field", axs[0,1]),
                ("align_ha", "Anisotropy Field", axs[1,0]),
                ("align_heff", "Effective Field", axs[1,1])]

    for key, title, ax in plot_map:
        ax.plot(hist_fft[key], color="blue", lw=2, label="FFT")
        ax.plot(hist_un[key], color="red", lw=2, label="UNet")
        ax.set_title(title)
        ax.set_xlabel("Iteration")
        ax.set_ylabel(r"$\langle \cos(\theta)\rangle$")
        ax.set_ylim(-1.05, 1.05)
        ax.grid(alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.savefig(
        os.path.join(folder, f"alignment_iteration_loop_{nloop}.png"), dpi=200)
    plt.close()


def plot_error_vs_transition_proximity(trajectory_error, vortex_count, base_path, args, spin_split, rand_seed,
                                        max_window=15, event_type='both'):
    """
    Bin trajectory error by "frames since nearest topological event"
    (vortex nucleation or annihilation, detected as a change in vortex
    count between consecutive Hext steps) and plot the resulting decay/
    rise curve.
    """
    folder = os.path.join(base_path, "summary_plots")
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
 
    title_text = (f"Layers: {args.layers} | Grid: {args.w}x{args.w} | Split: {spin_split} | Seed: {rand_seed} | Mask: {args.mask}\n"
                  f"Events aligned: {len(event_indices)} ({event_type})\n")
    fig.suptitle("Trajectory Error Aligned to Topological Events\n\n" + title_text, fontsize=12, fontweight='bold')
 
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
 
 
def plot_ablation_comparison_table(ablation_results, base_path, args, spin_split, rand_seed):
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
 
    title_text = (f"Layers: {args.layers} | Grid: {args.w}x{args.w} | Split: {spin_split} | Seed: {rand_seed} | Mask: {args.mask}")
    ax.set_title("Model Variant Ablation Comparison\n" + title_text, fontsize=12, fontweight='bold', pad=20)
 
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

def plot_hd_error_vs_vortex_cores(film1, film2, base_path, nloop, Hext_val, args, spin_split, rand_seed,
                                   itern1, itern2, err_fft, err_un, core_threshold=0.5):
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
    folder = os.path.join(base_path, "hd_error_vs_cores")
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
 
    title_text = (f"Film Layers: {args.layers} | Grid Size: {args.w}x{args.w} | Split: {spin_split} | Seed: {rand_seed} | Mask: {args.mask}\n"
                  f"Loop: {nloop} | $H_{{ext}}$ = {Hext_val:.1f} Oe | Vortex cells (|winding|>{core_threshold}): {core_mask.sum()}\n")
    fig.suptitle("Demag Field Error vs. Vortex Core Locations\n\n" + title_text, fontsize=12, fontweight='bold')
 
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

