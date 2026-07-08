# -*- coding: utf-8 -*-
"""
Created on Tue Apr 04 10:00:00 2023
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import argparse
import torch
import seaborn as sns
import time


from libs.misc import Culist, MaskTp, spin_prepare, winding_density
import libs.MAG2305 as MAG2305
from libs.Unet import UNet



def load_unet_model(args):
    # load Unet Model
    model = UNet(kc=args.krn, inc=args.layers*3, ouc=args.layers*3).eval().to(device)
    ckpt = '../ckpt/k{}/{}'.format(args.krn, args.model_name)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    MAG2305.load_model(model)
    print('Unet model loaded from {}'.format(ckpt))

def initialize_models(args):
    #Initialize MAG2305 models.
    film1 = MAG2305.mmModel(types='bulk', size=(args.w, args.w, args.layers), cell=(3,3,3), 
                            Ms=args.Ms, Ax=args.Ax, Ku=args.Ku, Kvec=args.Kvec, 
                            device="cuda:" + str(args.gpu))
    
    film2 = MAG2305.mmModel(types='bulk', size=(args.w, args.w, args.layers), cell=(3,3,3), 
                            Ms=args.Ms, Ax=args.Ax, Ku=args.Ku, Kvec=args.Kvec, 
                            device="cuda:" + str(args.gpu))

    print('Creating {} layer models \n'.format(args.layers))

    # Initialize demag matrix
    film1.DemagInit()
    print('initializing demag matrix \n')

    # load Unet Model
    load_unet_model(args)

    return film1, film2

def prepare_spin_state(film1, film2, args):
    """
    Prepare the initial spin state.
    """
    # spin_split = np.random.randint(low=2, high=32)
    # rand_seed  = np.random.randint(low=1000, high=100000)
    spin_split = 8
    rand_seed  = 1234
    spin = spin_prepare(spin_split, film1, rand_seed, mask=args.mask)
    film1.SpinInit(spin)
    film2.SpinInit(spin)
    cell_count = (np.linalg.norm(spin, axis=-1) > 0).sum()
    return spin_split, rand_seed, cell_count

def update_spin_fft(model, Hext, Hext_vec, cell_count, args):
    """
    Update the spin state of the model.
    """
    error = 1.0
    itern = 0
    error_rcd = np.array([])
    history = {
        'hd': [], 'ha': [], 'he': [], 'heff': [], 'm': [], 'mz': [],
        'e_demag': [], 'e_excha': [], 'e_anis': [], 'e_zeeman': []
    }
    h_vec_gpu = torch.tensor(Hext_vec, dtype=torch.float32, device=model.device)


    while itern < args.max_iter and error > args.error_min:
        # FFT_Hd spin update
        error = model.SpinLLG_RK4(Hext=Hext, dtime=args.dtime, damping=0.1)
        error_rcd = np.append(error_rcd, error)
        
        # track the field magnitude at this iteration
        # detached and sent to CPU as a single number, not an array
        he_mag = torch.mean(torch.linalg.norm(model.He, dim=-1)).item()
        ha_mag = torch.mean(torch.linalg.norm(model.Ha, dim=-1)).item()
        hd_mag = torch.mean(torch.linalg.norm(model.Hd, dim=-1)).item()
        heff_mag = torch.mean(torch.linalg.norm(model.Heff, dim=-1)).item()

        spin_sum = torch.sum(model.Spin, dim=(0, 1, 2))
        m_proj = torch.dot(spin_sum, h_vec_gpu).item() / cell_count
        mz_abs_avg = torch.mean(torch.abs(model.Spin[..., 2])).item()

        model.GetEnergy_detailed(Hext=Hext)

        history['hd'].append(hd_mag)
        history['ha'].append(ha_mag)
        history['he'].append(he_mag)
        history['heff'].append(heff_mag)
        history['m'].append(m_proj)
        history['mz'].append(mz_abs_avg)
        history['e_demag'].append(model.Energy_demag.item())
        history['e_excha'].append(model.Energy_excha.item())
        history['e_anis'].append(model.Energy_aniso.item() if hasattr(model, 'Energy_aniso') else 0.0)
        history['e_zeeman'].append(model.Energy_exter.item())
        history['e_total'].append(model.Energy.item())

        # Print iteration info
        if error <= args.error_min or itern % 1000 == 0:  # Adjust the frequency of printing as needed
            print(f'Iteration: {itern} \n'
                  f'Error_converge FFT: {error:.2e}')
        itern += 1

    return error_rcd, itern, hist_fft

def update_spin_unet(model, Hext, Hext_vec, cell_count, args):
    """
    Update the spin state of the model.
    """
    error = 1.0
    itern = 0
    error_fluc = 1.0
    error_rcd = np.array([])
    history = {
        'hd': [], 'ha': [], 'he': [], 'heff': [], 'm': [], 'mz': [],
        'e_demag': [], 'e_excha': [], 'e_anis': [], 'e_zeeman': []
    }
    h_vec_gpu = torch.tensor(Hext_vec, dtype=torch.float32, device=model.device)

    while itern < args.max_iter and error > args.error_min:
        # Unet_Hd spin update
        error = model.SpinLLG_RK4_unetHd(Hext=Hext, dtime=args.dtime, damping=0.1)
        error_rcd = np.append(error_rcd, error)

        # track the field magnitude at this iteration
        # detached and sent to CPU as a single number, not an array
        he_mag = torch.mean(torch.linalg.norm(model.He, dim=-1)).item()
        ha_mag = torch.mean(torch.linalg.norm(model.Ha, dim=-1)).item()
        hd_mag = torch.mean(torch.linalg.norm(model.Hd, dim=-1)).item()
        heff_mag = torch.mean(torch.linalg.norm(model.Heff, dim=-1)).item()

        spin_sum = torch.sum(model.Spin, dim=(0, 1, 2))
        m_proj = torch.dot(spin_sum, h_vec_gpu).item() / cell_count
        mz_abs_avg = torch.mean(torch.abs(model.Spin[..., 2])).item()

        model.GetEnergy_detailed(Hext=Hext)

        history['hd'].append(hd_mag)
        history['ha'].append(ha_mag)
        history['he'].append(he_mag)
        history['heff'].append(heff_mag)
        history['m'].append(m_proj)
        history['mz'].append(mz_abs_avg)
        history['e_demag'].append(model.Energy_demag.item())
        history['e_excha'].append(model.Energy_excha.item())
        history['e_anis'].append(model.Energy_aniso.item() if hasattr(model, 'Energy_aniso') else 0.0)
        history['e_zeeman'].append(model.Energy_zeeman.item())
        history['e_total'].append(model.Energy.item())
        
        # fluctation error break condition
        if itern > 20000:
            error_fluc = np.abs(error_rcd[-2000:].mean() - error_rcd[-500:].mean()) / error_rcd[-2000:].mean()
            if error_fluc < 0.02 and error < 1.0e-4:
                print('Unet error not decreasing! Break.')
                break
        # Print iteration info
        if error <= args.error_min or itern % 1000 == 0:  # Adjust the frequency of printing as needed
            print(f'Iteration: {itern} \n'
                  f'Error_converge UNet: {error:.2e}')
        itern += 1

    return error_rcd, itern, hist_un

def plot_results():
    """
    Plot and save the results.
    """
    fig, axs = plt.subplots(2, 4, figsize=(20, 10))
    title_text = (
        f"Film Layers: {args.layers} | Grid Size: {args.w}x{args.w} | Split: {spin_split} | Seed: {rand_seed}\n"
        f"Material Properties ── $M_s$: {args.Ms} emu/cc | $A_x$: {args.Ax} pJ/m | $K_u$: {args.Ku} $J/m^3$\n"
        f"Loop: {nloop} | $H_{{ext}}$ = {Hext_val:.1f} Oe | Target Threshold ($\Delta m_{{min}}$): {args.error_min:.2e}\n"
    )

    fig.suptitle(title_text, fontsize=13, fontweight='bold')
    fig.suptitle('{} layers film size:{}_split{}_seed{}_Ms{}_Ax{}_Ku{}\n \nloop:{} , Hext={}'.format(args.layers, args.w, spin_split, 
    rand_seed, args.Ms, args.Ax, args.Ku, nloop, Hext), fontsize=18 )
        
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
    plt.savefig(filename+'loop_{}.png'.format(nloop))
    plt.close()

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
    
    
    title_text = (
        f"Film Layers: {args.layers} | Grid Size: {args.w}x{args.w} | Split: {spin_split} | Seed: {rand_seed}\n"
        f"Material Properties ── $M_s$: {args.Ms} emu/cc | $A_x$: {args.Ax} pJ/m | $K_u$: {args.Ku} $J/m^3$\n"
    )
    
    fig.suptitle(title_text, fontsize=13, fontweight='bold')
    fig.suptitle("Domain Wall Position Graphs", fontsize=13, fontweight='bold')
    

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
    
    title_text = (
        f"Film Layers: {args.layers} | Grid Size: {args.w}x{args.w} | Split: {spin_split} | Seed: {rand_seed}\n"
        f"Material Properties ── $M_s$: {args.Ms} emu/cc | $A_x$: {args.Ax} pJ/m | $K_u$: {args.Ku} $J/m^3$\n"
        f"Target Threshold ($\Delta m_{{min}}$): {args.error_min:.2e}\n"
        f"FFT Steps: {itern1} (Final Err: {err_fft:.2e}) | UNet Steps: {itern2} (Final Err: {err_un:.2e})\n"
    )
    fig.suptitle(title_text, fontsize=13, fontweight='bold')
    fig.suptitle(f'Field Component Convergence Trajectories | Hext = {Hext_val:.1f} Oe (Loop {nloop})', fontsize=14, fontweight='bold')


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
    plot_map = [
        ('hd', 'Demagnetizing Field ($H_{demag}$)', 'Mean Demagnetizing Field [Oe]', axs[0, 0], 'field'),
        ('ha', 'Anisotropy Field ($H_{anis}$)', 'Mean Anisotropy Field [Oe]', axs[0, 1], 'field'),
        ('he', 'Exchange Field ($H_{ex}$)', 'Mean Exchange Field [Oe]', axs[1, 0], 'field'),
        ('heff', 'Total Effective Field ($H_{eff}$)', 'Mean Effective Field [Oe]', axs[1, 1], 'field'),
        ('m', m_title, 'Projected Magnetization $M_{ext}/M_s$', axs[2, 0], 'm_axis'),
        ('mz', mz_title, 'Mean Out-of-Plane Magnetization Magnitude $|M_z|$', axs[2, 1], 'mz_axis')
    ]
    
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
    
    title_text = (
        f"Film Layers: {args.layers} | Grid Size: {args.w}x{args.w} | Split: {spin_split} | Seed: {rand_seed}\n"
        f"Material Properties ── $M_s$: {args.Ms} emu/cc | $A_x$: {args.Ax} pJ/m | $K_u$: {args.Ku} $J/m^3$\n"
        f"Loop: {nloop} | $H_{{ext}}$ = {Hext_val:.1f} Oe | Target Threshold ($\Delta m_{{min}}$): {args.error_min:.2e}\n"
    )
    fig.suptitle(title_text, fontsize=13, fontweight='bold')
    
    # Dynamically scales to the highest peak but keeps the bounds symmetrical around zero
    max_charge = max(np.max(np.abs(topo_fft)), np.max(np.abs(topo_un)))
    global_vmax = max_charge if max_charge > 0.1 else 1.0
    global_vmin = -global_vmax
    
    # Panel 1: FFT Ground Truth Topological Heatmap
    im0 = axs[0, 0].imshow(topo_fft, cmap='bwr', origin='lower', vmin=global_vmin, vmax=global_vmax)
    axs[0, 0].set_title(f'FFT Solver Topological Charge Map\nTotal Absolute Vortices: {winding_abs_fft.item():.1f}', fontsize=11, fontweight='bold')
    axs[0, 0].set_xlabel('x [nm]', fontsize=9)
    axs[0, 0].set_ylabel('y [nm]', fontsize=9)
    fig.colorbar(im0, ax=axs[0, 0], label='Local Topological Charge Density')
    
    # Panel 2: UNet Framework Topological Heatmap (Locked to identical bounds)
    im1 = axs[0, 1].imshow(topo_un, cmap='bwr', origin='lower', vmin=global_vmin, vmax=global_vmax)
    axs[0, 1].set_title(f'UNet Model Topological Charge Map\nTotal Absolute Vortices: {winding_abs_un.item():.1f}', fontsize=11, fontweight='bold')
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
    
    title_text = (
        f"Film Layers: {args.layers} | Grid Size: {args.w}x{args.w} | Split: {spin_split} | Seed: {rand_seed}\n"
        f"Material Properties ── $M_s$: {args.Ms} emu/cc | $A_x$: {args.Ax} pJ/m | $K_u$: {args.Ku} $J/m^3$\n"
        f"Loop: {nloop} | $H_{{ext}}$ = {Hext_val:.1f} Oe | Target Threshold ($\Delta m_{{min}}$): {args.error_min:.2e}\n"
    )
    fig.suptitle(title_text, fontsize=13, fontweight='bold')
    
    global_xmax_padded = max(len(hist_fft['e_demag']), len(hist_un['e_demag'])) * 1.05
    
    plot_map = [
        ('e_demag', 'Demagnetizing Energy ($E_{demag}$)', 'Energy [Joules]', axs[0, 0]),
        ('e_anis', 'Anisotropy Energy ($E_{anis}$)', 'Energy [Joules]', axs[0, 1]),
        ('e_excha', 'Exchange Energy ($E_{excha}$)', 'Energy [Joules]', axs[1, 0]),
        ('e_zeeman', 'Zeeman Energy ($E_{zeeman}$)', 'Energy [Joules]', axs[1, 1])
    ]
    
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
        ax.set_grid(True, linestyle='--', alpha=0.4)
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
    full_energy_folder = os.path.join(base_path, "summary_energy")
    os.makedirs(full_energy_folder, exist_ok=True)
    
    fig, axs = plt.subplots(2, 2, figsize=(15, 11))
    
    title_text = (
        f"Film Layers: {args.layers} | Grid Size: {args.w}x{args.w} | Split: {spin_split} | Seed: {rand_seed}\n"
        f"Material Properties ── $M_s$: {args.Ms} emu/cc | $A_x$: {args.Ax} pJ/m | $K_u$: {args.Ku} $J/m^3$\n"
    )
    fig.suptitle(title_text, fontsize=13, fontweight='bold')
    
    # Calculate uniform X-axis padding based on external field bounds
    max_h, min_h = max(Hext_range), min(Hext_range)
    h_range = max_h - min_h
    xmax_padded = max_h + (h_range * 0.05)
    xmin_padded = min_h - (h_range * 0.05)
    
    plot_map = [
        ('demag', 'Equilibrium Demagnetizing Energy ($E_{demag}$)', axs[0, 0]),
        ('anis', 'Equilibrium Anisotropy Energy ($E_{anis}$)', axs[0, 1]),
        ('excha', 'Equilibrium Exchange Energy ($E_{excha}$)', axs[1, 0]),
        ('zeeman', 'Equilibrium Zeeman Energy ($E_{zeeman}$)', axs[1, 1])
    ]
    
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
        ax.set_grid(True, linestyle='--', alpha=0.4)
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
    ax_tot.set_grid(True, linestyle='--', alpha=0.4)
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
    performance_folder = os.path.join(base_path, "summary_performance")
    os.makedirs(performance_folder, exist_ok=True)
    
    fig, axs = plt.subplots(2, 2, figsize=(15, 11))
    
    title_text = (
        f"Film Layers: {args.layers} | Grid Size: {args.w}x{args.w} | Split: {spin_split} | Seed: {rand_seed}\n"
        f"Material Properties ── $M_s$: {args.Ms} emu/cc | $A_x$: {args.Ax} pJ/m | $K_u$: {args.Ku} $J/m^3$\n"
    )
    fig.suptitle(title_text, fontsize=13, fontweight='bold')
    
    # Calculate uniform X-axis bounds with  5% padding
    max_h, min_h = max(Hext_range), min(Hext_range)
    h_range = max_h - min_h
    xmax_padded = max_h + (h_range * 0.05)
    xmin_padded = min_h - (h_range * 0.05)
    
    plot_map = [
        ('iters', 'Solver Iterations Per Loop', 'Total Iteration Count/Hext Step', axs[0, 0]),
        ('vortices', 'Topological Vortex Count', 'Absolute Vortex Population Count', axs[0, 1]),
        ('mz', 'Mean Out-of-Plane Magnetization ($|M_z|$)', 'Average Absolute Magnitude $|M_z|$', axs[1, 0]),
        ('time', 'Real-World Total Execution Time', 'Compute Duration [Seconds]', axs[1, 1]) 
    ]
    
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
        ax.set_grid(True, linestyle='--', alpha=0.4)
        ax.legend(loc='upper right', fontsize=9)
        
    plt.tight_layout()
    plt.savefig(os.path.join(performance_folder, 'performance_summary.png'), dpi=200)
    plt.close()

def plot_error_summary(Hext_range, inst_hd_mae, traj_shift_mae, hex_err_mae, hanis_err_mae, base_path, args, spin_split, rand_seed):
    """
    Generates a final 2x2 multi-panel master report compiling all local field approximations,
    historical path tracking drift, and intrinsic field deviations across the Hext sweep.
    """
    error_summary_folder = os.path.join(base_path, "summary_errors")
    os.makedirs(error_summary_folder, exist_ok=True)
    
    print("Generating comprehensive 4-panel error tracking analysis...")
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    
    title_text = (
        f"Film Layers: {args.layers} | Grid Size: {args.w}x{args.w} | Split: {spin_split} | Seed: {rand_seed}\n"
        f"Material Properties ── $M_s$: {args.Ms} emu/cc | $A_x$: {args.Ax} pJ/m | $K_u$: {args.Ku} $J/m^3$\n"
    )
    fig.suptitle(title_text, fontsize=13, fontweight='bold')
    
    # Calculate uniform X-axis limits with standard 5% padding while maintaining the reversed sweep
    max_h, min_h = max(Hext_range), min(Hext_range)
    h_range = max_h - min_h if max_h != min_h else 1.0
    xmax_padded = max_h + (h_range * 0.05)
    xmin_padded = min_h - (h_range * 0.05)
    
    # Structural Mapping Matrix to cycle configurations cleanly
    plot_map = [
        (inst_hd_mae, 'darkorange', 'Total Unet Model $H_{demag}$ Approximation Error', '$H_{demag}$ Field Prediction Error', 'Instantaneous $H_{demag}$ MAE [Oe]', axs[0, 0]),
        (traj_shift_mae, 'crimson', 'Magnetization Trajectory Drift (Accumulated Error)', 'Predicted Magnetization Error', 'Cumulative Spin $\\vec{m}$ MAE', axs[0, 1]),
        (hex_err_mae, 'purple', 'Total Exchange Field ($H_{ex}$) Error Accumulation', '$H_{ex}$ Prediction Error', 'Exchange Field MAE [Oe]', axs[1, 0]),
        (hanis_err_mae, 'teal', 'Total Anisotropy Field ($H_{anis}$) Error Accumulation', '$H_{anis}$ Prediction Error', 'Anisotropy Field MAE [Oe]', axs[1, 1])
    ]
    
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
    Generates a final 2x2 multi-panel master physical report charting the equilibrium 
    magnitudes of all internal fields across the completed Hext sweep range.
    """
    fields_summary_folder = os.path.join(base_path, "summary_fields")
    os.makedirs(fields_summary_folder, exist_ok=True)
    
    print("Generating final 4-panel physical field summary plot...")
    fig, axs = plt.subplots(2, 2, figsize=(15, 11))
    
    title_text = (
        f"Film Layers: {args.layers} | Grid Size: {args.w}x{args.w} | Split: {spin_split} | Seed: {rand_seed}\n"
        f"Material Properties ── $M_s$: {args.Ms} emu/cc | $A_x$: {args.Ax} pJ/m | $K_u$: {args.Ku} $J/m^3$\n"
    )
    fig.suptitle(title_text, fontsize=13, fontweight='bold')
    
    max_h, min_h = max(Hext_range), min(Hext_range)
    h_range = max_h - min_h if max_h != min_h else 1.0
    xmax_padded = max_h + (h_range * 0.05)
    xmin_padded = min_h - (h_range * 0.05)
    
    plot_map = [
        (hex_mm, hex_un, 'Exchange Field ($H_{ex}$)', 'Mean $H_{ex}$ Magnitude [Oe]', axs[0, 0]),
        (hanis_mm, hanis_un, 'Anisotropy Field ($H_{anis}$)', 'Mean $H_{anis}$ Magnitude [Oe]', axs[0, 1]),
        (hd_mm, hd_un, 'Demagnetizing Field ($H_{demag}$)', 'Mean $H_{demag}$ Magnitude [Oe]', axs[1, 0]),
        (heff_mm, heff_un, 'Total Effective Field ($H_{eff}$)', 'Mean $H_{eff}$ Magnitude [Oe]', axs[1, 1]) 
    ]
    
    for data_mm, data_un, panel_title, y_label, ax in plot_map:
        ax.plot(Hext_range, data_mm, color='blue', lw=2.5, linestyle='-', label='FFT Simulator (mm)')
        ax.plot(Hext_range, data_un, color='red', lw=2.5, linestyle='-', label='UNet Model (un)')
        
        ax.set_title(panel_title, fontsize=11, fontweight='bold')
        ax.set_xlabel('External Field $H_{ext}$ [Oe]', fontsize=10)
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

    # initialize spin state
    spin_split, rand_seed, cell_count = prepare_spin_state(film1, film2, args)
    
    # create folder
    filename='./figs_k{}/model_{}/shape_{}/size{}_Ms{}_Ax{}_Ku{}_dtime{}_split{}_seed{}_Layers{}/'.format(
                    args.krn, args.loss_type, args.mask, args.w, 
                    args.Ms, args.Ax, args.Ku, 
                    args.dtime, spin_split, rand_seed, args.layers
                    )
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    
    # get MH data
    x_plot,y1_plot,y2_plot = [],[],[]

    # Hext range
    Hext_range = np.linspace(1000,-1000,201)
    Hext_vec = np.array([np.cos(0.01), np.sin(0.01), 0.0])

    spin_mm = np.array([[[[1]]]])
    spin_un = np.array([[[[1]]]])

    # Error Tracking Lists Hdemag
    instantaneous_hd_mae = []   # Local network prediction discrepancy at equilibrium
    trajectory_shift_mae = []   # Accumulated configuration divergence over historical path

    # Initialize lists to track Hex and Hanis data across the loop
    hex_mm_plot, hex_un_plot = [], []
    hanis_mm_plot, hanis_un_plot = [], []
    hd_mm_plot, hd_un_plot = [], []
    heff_mm_plot, heff_un_plot = [], []
    
    full_energy_fft = {'demag': [], 'anis': [], 'excha': [], 'zeeman': [], 'total': []}
    full_energy_un  = {'demag': [], 'anis': [], 'excha': [], 'zeeman': [], 'total': []}
    performance_fft = {'iters': [], 'vortices': [], 'mz': [], 'time': []}
    performance_un  = {'iters': [], 'vortices': [], 'mz': [], 'time': []}

    # Error accumulation tracking lists for intrinsic fields
    hex_error_mae = []
    hanis_error_mae = []

    # Main loop
    for nloop, Hext_val in enumerate(Hext_range):
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

        # Extract topological counts using winding density 
        spin_fft_tensor = film1.Spin.permute(3, 0, 1, 2)[:, :, :, 0].unsqueeze(0)
        spin_un_tensor  = film2.Spin.permute(3, 0, 1, 2)[:, :, :, 0].unsqueeze(0)
        _, vortex_count_fft, _ = winding_density(spin_fft_tensor)
        _, vortex_count_un,  _ = winding_density(spin_un_tensor)

        # Append information for 
        performance_fft['iters'].append(itern1)
        performance_fft['vortices'].append(vortex_count_fft.item())
        performance_fft['mz'].append(hist_fft['mz'][-1])
        performance_fft['time'].append(time_elapsed_fft)
        performance_fft['total'].append(hist_fft['e_total'][-1])
        
        performance_un['iters'].append(itern2)
        performance_un['vortices'].append(vortex_count_un.item())
        performance_un['mz'].append(hist_un['mz'][-1])
        performance_un['time'].append(time_elapsed_un)
        performance_un['total'].append(hist_un['e_total'][-1])
    
        # Extract final convergence values from the error logs
        final_err_fft = error1_rcd[-1] if len(error1_rcd) > 0 else 0.0
        final_err_un  = error2_rcd[-1] if len(error2_rcd) > 0 else 0.0

        full_energy_fft['demag'].append(hist_fft['e_demag'][-1])
        full_energy_fft['anis'].append(hist_fft['e_anis'][-1])
        full_energy_fft['excha'].append(hist_fft['e_excha'][-1])
        full_energy_fft['zeeman'].append(hist_fft['e_zeeman'][-1])
        full_energy_fft['total'].append(hist_fft['e_total'][-1])

        
        full_energy_un['demag'].append(hist_un['e_demag'][-1])
        full_energy_un['anis'].append(hist_un['e_anis'][-1])
        full_energy_un['excha'].append(hist_un['e_excha'][-1])
        full_energy_un['zeeman'].append(hist_un['e_zeeman'][-1])
        full_energy_un['total'].append(hist_un['e_total'][-1])


        # get spin and Hd
        spin_mm = film1.Spin.detach().cpu().numpy()
        spin_un = film2.Spin.detach().cpu().numpy()
        Hd_mm = film1.Hd.detach().cpu().numpy()
        Hd_un = film2.Hd.detach().cpu().numpy()

        # Extract Hex and Hanis arrays from both models
        Hex_mm = film1.Ha.detach().cpu().numpy()
        Hex_un = film2.Ha.detach().cpu().numpy()
        Hanis_mm = film1.He.detach().cpu().numpy()
        Hanis_un = film2.He.detach().cpu().numpy()

        # Calculate the spatial average magnitude across the grid sample
        hex_mm_plot.append(np.mean(np.linalg.norm(Hex_mm, axis=-1)))
        hex_un_plot.append(np.mean(np.linalg.norm(Hex_un, axis=-1)))
        hanis_mm_plot.append(np.mean(np.linalg.norm(Hanis_mm, axis=-1)))
        hanis_un_plot.append(np.mean(np.linalg.norm(Hanis_un, axis=-1)))
        hd_mm_plot.append(np.mean(np.linalg.norm(Hd_mm, axis=-1)))      
        hd_un_plot.append(np.mean(np.linalg.norm(Hd_un, axis=-1)))
        heff_mm_plot.append(hist_fft['heff'][-1]) 
        heff_un_plot.append(hist_un['heff'][-1])

        # Calculate and record the Mean Absolute Error (MAE) between UNet and FFT fields
        hex_error_mae.append(np.mean(np.abs(Hex_un - Hex_mm)))
        hanis_error_mae.append(np.mean(np.abs(Hanis_un - Hanis_mm)))

        # Calculate and append tracking errors
        hd_error = np.mean(np.abs(Hd_un - Hd_mm))
        spin_error = np.mean(np.abs(spin_un - spin_mm))

        instantaneous_hd_mae.append(hd_error)
        trajectory_shift_mae.append(spin_error)
        
        #MH loop data
        x_plot.append(Hext_val)
        y1_plot.append( np.dot(spin_mm.sum(axis=(0,1,2)), Hext_vec)/ cell_count )
        y2_plot.append( np.dot(spin_un.sum(axis=(0,1,2)), Hext_vec)/ cell_count )

        # Plot results
        plot_results()
        plot_iteration_domain_walls(
            film1, film2, filename, nloop, Hext_val, 
            args, spin_split, rand_seed, itern1, itern2, 
            final_err_fft, final_err_un
        )        
        plot_iteration_fields(
            hist_fft, hist_un, filename, nloop, Hext_val, 
            args, spin_split, rand_seed, itern1, itern2, 
            final_err_fft, final_err_un
        )
        plot_iteration_winding_density(
            film1, film2, filename, nloop, Hext_val, 
            args, spin_split, rand_seed, itern1, itern2, 
            final_err_fft, final_err_un
        )
        plot_iteration_energy(
            hist_fft, hist_un, filename, nloop, Hext_val, 
            args, spin_split, rand_seed, itern1, itern2, 
            final_err_fft, final_err_un
        )

        # Save MH data
        np.save(filename + "Hext_array", x_plot)
        np.save(filename + "Mext_array_mm", y1_plot)
        np.save(filename + "Mext_array_un", y2_plot)

        # Save tracking errors dynamically
        np.save(filename + "instantaneous_hd_mae", instantaneous_hd_mae)
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

    plot_full_energy_summary(
        full_energy_fft, full_energy_un, Hext_range, 
        filename, args, spin_split, rand_seed
    )

            
    plot_performance_summary(
        performance_fft, performance_un, Hext_range, 
        filename, args, spin_split, rand_seed
    )

    plot_error_summary(
        Hext_range, instantaneous_hd_mae, trajectory_shift_mae, 
        hex_error_mae, hanis_error_mae, filename, 
        args, spin_split, rand_seed
    )

    plot_fields_summary(
        Hext_range, hex_mm_plot, hex_un_plot, 
        hanis_mm_plot, hanis_un_plot, hd_mm_plot, hd_un_plot, 
        heff_mm_plot, heff_un_plot, filename, 
        args, spin_split, rand_seed
    )



