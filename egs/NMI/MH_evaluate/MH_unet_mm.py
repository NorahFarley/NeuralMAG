# -*- coding: utf-8 -*-
"""
Created on Tue Apr 04 10:00:00 2023
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
import csv
from scipy.stats import pearsonr

from egs.NMI.MH_evaluate.searcher import analyze_transition_peaks, compare_transition_predictors, extract_candidate_predictors, rank_transition_predictors
from libs.misc import Culist, MaskTp, spin_prepare, winding_density
import libs.MAG2305 as MAG2305
from libs.Unet import UNet
from plots import (plot_iteration_domain_walls, plot_iteration_fields, plot_iteration_winding_density, plot_iteration_energy, 
                   plot_full_energy_summary, plot_performance_summary, plot_error_summary, plot_fields_summary, 
                   plot_error_correlations, plot_iteration_torque, plot_iteration_alignment)


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
    hist_fft = {'hd': [], 'ha': [], 'he': [], 'heff': [], 'm': [], 'mz': [],
                'tau_hd': [], 'tau_he': [], 'tau_ha': [], 'tau_heff': [],
                'e_demag': [], 'e_excha': [], 'e_anis': [], 'e_exter': [], 'e_total': [], 
                'align_hd': [], 'align_he': [], 'align_ha': [], 'align_heff': []}
    
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

        tau_hd = torch.mean(torch.linalg.norm(torch.cross(model.Spin, model.Hd, dim=-1), dim=-1)).item()
        tau_he = torch.mean(torch.linalg.norm(torch.cross(model.Spin, model.He, dim=-1), dim=-1)).item()
        tau_ha = torch.mean(torch.linalg.norm(torch.cross(model.Spin, model.Ha, dim=-1), dim=-1)).item()
        tau_heff = torch.mean(torch.linalg.norm(torch.cross(model.Spin, model.Heff, dim=-1), dim=-1)).item()

        eps = 1e-12

        align_hd = torch.mean(torch.sum(model.Spin * model.Hd, dim=-1) / (torch.linalg.norm(model.Hd, dim=-1) + eps)).item()
        align_he = torch.mean(torch.sum(model.Spin * model.He, dim=-1) / (torch.linalg.norm(model.He, dim=-1) + eps)).item()
        align_ha = torch.mean(torch.sum(model.Spin * model.Ha, dim=-1) / (torch.linalg.norm(model.Ha, dim=-1) + eps)).item()
        align_heff = torch.mean(torch.sum(model.Spin * model.Heff, dim=-1) / (torch.linalg.norm(model.Heff, dim=-1) + eps)).item()

        #model.GetEnergy_detailed(Hext=Hext)
        hist_fft['align_hd'].append(align_hd)
        hist_fft['align_he'].append(align_he)
        hist_fft['align_ha'].append(align_ha)
        hist_fft['align_heff'].append(align_heff)
        hist_fft['tau_hd'].append(tau_hd)
        hist_fft['tau_he'].append(tau_he)
        hist_fft['tau_ha'].append(tau_ha)
        hist_fft['tau_heff'].append(tau_heff)
        hist_fft['hd'].append(hd_mag)
        hist_fft['ha'].append(ha_mag)
        hist_fft['he'].append(he_mag)
        hist_fft['heff'].append(heff_mag)
        hist_fft['m'].append(m_proj)
        hist_fft['mz'].append(mz_abs_avg)
        # hist_fft['e_demag'].append(model.Energy_demag.item())
        # hist_fft['e_excha'].append(model.Energy_excha.item())
        # hist_fft['e_anis'].append(model.Energy_aniso.item() if hasattr(model, 'Energy_aniso') else 0.0)
        # hist_fft['e_exter'].append(model.Energy_exter.item())
        # hist_fft['e_total'].append(model.Energy.item())

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
    hist_un = {'hd': [], 'ha': [], 'he': [], 'heff': [], 'm': [], 'mz': [],
               'tau_hd': [], 'tau_he': [], 'tau_ha': [], 'tau_heff': [],
               'e_demag': [], 'e_excha': [], 'e_anis': [], 'e_exter': [], 'e_total': [], 
               'align_hd': [],'align_he': [],'align_ha': [],'align_heff': []}
    
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

        tau_hd = torch.mean(torch.linalg.norm(torch.cross(model.Spin, model.Hd, dim=-1), dim=-1)).item()
        tau_he = torch.mean(torch.linalg.norm(torch.cross(model.Spin, model.He, dim=-1), dim=-1)).item()
        tau_ha = torch.mean(torch.linalg.norm(torch.cross(model.Spin, model.Ha, dim=-1), dim=-1)).item()
        tau_heff = torch.mean(torch.linalg.norm(torch.cross(model.Spin, model.Heff, dim=-1), dim=-1)).item()

        eps = 1e-12

        align_hd = torch.mean(torch.sum(model.Spin * model.Hd, dim=-1) / (torch.linalg.norm(model.Hd, dim=-1) + eps)).item()
        align_he = torch.mean(torch.sum(model.Spin * model.He, dim=-1) / (torch.linalg.norm(model.He, dim=-1) + eps)).item()
        align_ha = torch.mean(torch.sum(model.Spin * model.Ha, dim=-1) / (torch.linalg.norm(model.Ha, dim=-1) + eps)).item()
        align_heff = torch.mean(torch.sum(model.Spin * model.Heff, dim=-1) / (torch.linalg.norm(model.Heff, dim=-1) + eps)).item()

        # model.GetEnergy_detailed(Hext=Hext)
        hist_un['align_hd'].append(align_hd)
        hist_un['align_he'].append(align_he)
        hist_un['align_ha'].append(align_ha)
        hist_un['align_heff'].append(align_heff)
        hist_un['tau_hd'].append(tau_hd)
        hist_un['tau_he'].append(tau_he)
        hist_un['tau_ha'].append(tau_ha)
        hist_un['tau_heff'].append(tau_heff)
        hist_un['hd'].append(hd_mag)
        hist_un['ha'].append(ha_mag)
        hist_un['he'].append(he_mag)
        hist_un['heff'].append(heff_mag)
        hist_un['m'].append(m_proj)
        hist_un['mz'].append(mz_abs_avg)
        # hist_un['e_demag'].append(model.Energy_demag.item())
        # hist_un['e_excha'].append(model.Energy_excha.item())
        # hist_un['e_anis'].append(model.Energy_aniso.item() if hasattr(model, 'Energy_aniso') else 0.0)
        # hist_un['e_exter'].append(model.Energy_exter.item())
        # hist_un['e_total'].append(model.Energy.item())
        
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


def plot_results(spin_split, rand_seed, args, nloop, Hext_val, spin_mm, spin_un, itern1, itern2,
                  Hd_mm, Hd_un, x_plot, y1_plot, y2_plot, Hext_range, error1_rcd, error2_rcd, filename):
    """
    Plot and save the results.
    """
    fig, axs = plt.subplots(2, 4, figsize=(20, 10))
    title_text = (f"Film Layers: {args.layers} | Grid Size: {args.w}x{args.w} | Split: {spin_split} | Seed: {rand_seed} | Mask: {args.mask}\n"
                  f"Material Properties ── $M_s$: {args.Ms} emu/cc | $A_x$: {args.Ax} pJ/m | $K_u$: {args.Ku} $J/m^3$\n"
                  f"Loop: {nloop} | $H_{{ext}}$ = {Hext_val:.1f} Oe\n")

    fig.suptitle(title_text, fontsize=13, fontweight='bold')
    # fig.suptitle('{} layers film size:{}_split{}_seed{}_Ms{}_Ax{}_Ku{}\n \nloop:{} , Hext={}'.format(args.layers, args.w, spin_split, 
    # rand_seed, args.Ms, args.Ax, args.Ku, nloop, Hext), fontsize=18 )
    
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


def plot_transition_predictors(Hext_range, trajectory_error, save_path, predictor_dict, peak_count=2, max_lag=15):
    """
    Compare transition predictor candidates against trajectory error.

    Parameters
    ----------
    Hext_range : array, External field values.
    trajectory_error : array, Spin trajectory MAE.
    save_path : str, Base output directory.
    predictor_dict : dict, Dictionary of form:

        {"Hd Error": instantaneous_hd_mae,
        "Hex Error": hex_error_mae, 
        "Hanis Error": hanis_error_mae,
        "Vortex Count": full_fft["vortices"],
        ...}

    peak_count : int, Number of largest trajectory-error peaks to mark.
    max_lag : int, Maximum lag (±max_lag) used in lag correlation search.
    """

    out_dir = os.path.join(save_path, "transition_analysis")
    os.makedirs(out_dir, exist_ok=True)

    error = np.asarray(trajectory_error)

    # Normalize helper
    def normalize(x):
        x = np.asarray(x)
        if np.max(x) == np.min(x):
            return np.zeros_like(x)
        return (x - np.min(x)) / (np.max(x) - np.min(x))

    peak_indices = np.argpartition(error, -peak_count)[-peak_count:]
    peak_indices = peak_indices[np.argsort(error[peak_indices])[::-1]]

    plt.figure(figsize=(14,8))
    plt.plot(Hext_range, normalize(error), linewidth=3, color="black", label="Trajectory Error")

    colors = plt.cm.tab10(np.linspace(0,1,len(predictor_dict)))
    results = []

    for color,(name,data) in zip(colors,predictor_dict.items()):
        data=np.asarray(data)
        plt.plot(Hext_range, normalize(data), linewidth=2, alpha=0.8, label=name, color=color)

        # Pearson correlation
        r,p = pearsonr(error,data)

        best_r = r
        best_lag = 0

        for lag in range(-max_lag,max_lag+1):
            if lag<0:
                rlag,_ = pearsonr(error[-lag:], data[:lag])
            elif lag>0:
                rlag,_ = pearsonr(error[:-lag], data[lag:])
            else:
                rlag=r

            if abs(rlag)>abs(best_r):
                best_r=rlag
                best_lag=lag

        peak_values=[]

        for idx in peak_indices:
            peak_values.append(data[idx])

        results.append([name, r, p, best_r, best_lag, *peak_values])

    for i,idx in enumerate(peak_indices):
        plt.axvline(Hext_range[idx], color="red", linestyle="--", alpha=0.6)
        plt.text(Hext_range[idx], 1.02, f"Peak {i+1}", rotation=90, fontsize=10)

    plt.grid(alpha=0.3)
    plt.xlabel("External Field (Oe)")
    plt.ylabel("Normalized Quantity")
    plt.title("Transition Predictor Comparison")
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir,"transition_predictors.png"), dpi=300)
    plt.close()

    csv_file=os.path.join(out_dir, "transition_predictor_statistics.csv")
    header=["Predictor", "Pearson r", "p value", "Best lag correlation", "Best lag",]

    for i in range(peak_count):
        header.append(f"Value at Peak {i+1}")

    with open(csv_file,"w",newline="") as f:
        writer=csv.writer(f)
        writer.writerow(header)
        for row in results:
            writer.writerow(row)

    print()
    print("Transition predictor analysis saved.")
    print(csv_file)

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
    
    full_fft = {'demag': [], 'anis': [], 'excha': [], 'exter': [], 'total': [], 
                'tau_hd': [], 'tau_he': [], 'tau_ha': [], 'tau_heff': [], 
                'iters': [], 'vortices': [], 'mz': [], 'time': []}
    full_un = {'demag': [], 'anis': [], 'excha': [], 'exter': [], 'total': [], 
               'tau_hd': [], 'tau_he': [], 'tau_ha': [], 'tau_heff': [], 
               'iters': [], 'vortices': [], 'mz': [], 'time': []}

    hex_error_mae = []
    hanis_error_mae = []
    heff_error_mae = []
    hd_error_mae = []   
    trajectory_shift_mae = [] 

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

        # Extract final convergence values from the error logs
        final_err_fft = error1_rcd[-1] if len(error1_rcd) > 0 else 0.0
        final_err_un  = error2_rcd[-1] if len(error2_rcd) > 0 else 0.0

        film1.GetEnergy_detailed(Hext=Hext)
        film2.GetEnergy_detailed(Hext=Hext)

        # Extract topological counts using winding density 
        spin_fft_tensor = film1.Spin.permute(3, 0, 1, 2)[:, :, :, 0].unsqueeze(0)
        spin_un_tensor  = film2.Spin.permute(3, 0, 1, 2)[:, :, :, 0].unsqueeze(0)
        _, vortex_count_fft, _ = winding_density(spin_fft_tensor)
        _, vortex_count_un,  _ = winding_density(spin_un_tensor)

        full_fft['iters'].append(itern1)
        full_fft['vortices'].append(vortex_count_fft)
        full_fft['mz'].append(hist_fft['mz'][-1])
        full_fft['time'].append(time_elapsed_fft)
        full_un['iters'].append(itern2)
        full_un['vortices'].append(vortex_count_un)
        full_un['mz'].append(hist_un['mz'][-1])
        full_un['time'].append(time_elapsed_un)

        full_fft['demag'].append(hist_fft['e_demag'][-1])
        full_fft['anis'].append(hist_fft['e_anis'][-1])
        full_fft['excha'].append(hist_fft['e_excha'][-1])
        full_fft['exter'].append(hist_fft['e_exter'][-1])
        full_fft['total'].append(hist_fft['e_total'][-1])
        full_un['demag'].append(hist_un['e_demag'][-1])
        full_un['anis'].append(hist_un['e_anis'][-1])
        full_un['excha'].append(hist_un['e_excha'][-1])
        full_un['exter'].append(hist_un['e_exter'][-1])
        full_un['total'].append(hist_un['e_total'][-1])

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
        
        #MH loop data
        x_plot.append(Hext_val)
        y1_plot.append( np.dot(spin_mm.sum(axis=(0,1,2)), Hext_vec)/ cell_count )
        y2_plot.append( np.dot(spin_un.sum(axis=(0,1,2)), Hext_vec)/ cell_count )

        title_text = (f"Film Layers: {args.layers} | Grid Size: {args.w}x{args.w} | Split: {spin_split} | Seed: {rand_seed} | Mask: {args.mask}\n"
                  f"Material Properties ── $M_s$: {args.Ms} emu/cc | $A_x$: {args.Ax} pJ/m | $K_u$: {args.Ku} $J/m^3$\n"
                  f"Loop: {nloop} | $H_{{ext}}$ = {Hext_val:.1f} Oe | Iterations: mm [{itern1}] | un [{itern2}] | Error: mm [{final_err_fft:.2e}] | un [{final_err_un:.2e}]\n ")

        # Plot results
        plot_results(spin_split, rand_seed, args, nloop, Hext_val, spin_mm, spin_un, itern1, itern2,
                     full_fft['demag'], full_un['demag'], x_plot, y1_plot, y2_plot, Hext_range, error1_rcd, error2_rcd, filename)
        plot_iteration_domain_walls(film1, film2, filename, nloop, Hext_val, args, spin_split, rand_seed, 
                                    itern1, itern2, final_err_fft, final_err_un)        
        plot_iteration_fields(hist_fft, hist_un, filename, nloop, Hext_val, args, spin_split, rand_seed, 
                              itern1, itern2, final_err_fft, final_err_un)
        plot_iteration_winding_density(film1, film2, filename, nloop, Hext_val, args, spin_split, rand_seed, 
                                       itern1, itern2, final_err_fft, final_err_un)
        # plot_iteration_energy(hist_fft, hist_un, filename, nloop, Hext_val, args, spin_split, rand_seed, 
        #                       itern1, itern2, final_err_fft, final_err_un)
        plot_iteration_torque(hist_fft, hist_un, filename, nloop, Hext_val, args, spin_split, rand_seed,
                               itern1, itern2, final_err_fft, final_err_un)
        plot_iteration_alignment(hist_fft, hist_un, filename, nloop, Hext_val, args, spin_split, rand_seed,
                                  itern1, itern2, final_err_fft, final_err_un)
        
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


    plot_full_energy_summary(full_fft, full_un, Hext_range, filename, args, spin_split, rand_seed)
    plot_performance_summary(full_fft, full_un, Hext_range, filename, args, spin_split, rand_seed)
    plot_error_summary(Hext_range, hd_error_mae, trajectory_shift_mae, hex_error_mae, hanis_error_mae, 
                       filename, args, spin_split, rand_seed)
    plot_fields_summary(Hext_range, hex_mm_plot, hex_un_plot, hanis_mm_plot, hanis_un_plot, hd_mm_plot, hd_un_plot, 
                        heff_mm_plot, heff_un_plot, filename, args, spin_split, rand_seed)
    plot_error_correlations(hd_error_mae, hex_error_mae, hanis_error_mae, trajectory_shift_mae,
                             filename, args, spin_split, rand_seed, Hext_range=Hext_range)

    # predictors2 = {"Vortex Count": vortex_count,
    #                "Winding Density": winding_density,
    #                "Gradient Magnitude": grad_mag,
    #                "Exchange Energy": exchange_energy,
    #                "Demag Torque": demag_torque,
    #                "Precessional Torque": precessional_torque,
    #                "Hexch Error": hexch_error,
    #                "Hanis Error": hanis_error,
    #                "Hdemag Error": hdemag_error,}

#TODO: add mask to plot titles

    predictors = {"Hd Error": hd_error_mae,
                  "Hex Error": hex_error_mae,
                  "Hanis Error": hanis_error_mae,
                  "Heff Error": heff_error_mae,
                  "Hd Magnitude": hd_mm_plot,
                  "Hex Magnitude": hex_mm_plot,
                  "Hanis Magnitude": hanis_mm_plot,
                  "Heff Magnitude": heff_mm_plot,
                  "Mz": full_fft["mz"],
                  "Vortex Count": full_fft["vortices"],
                  "Demag Energy": full_fft["demag"],
                  "Exchange Energy": full_fft["excha"],
                  "Anisotropy Energy": full_fft["anis"]}

    candidate_predictors = extract_candidate_predictors(film1)

    rank_transition_predictors(trajectory_shift_mae, predictors, Hext_range, filename)
    plot_transition_predictors(Hext_range, trajectory_shift_mae, filename, predictors)

    analyze_transition_peaks(Hext_range, trajectory_shift_mae, full_fft["vortices"], parameter_name="Vortex Count")
    analyze_transition_peaks(Hext_range, trajectory_shift_mae, full_fft["excha"], parameter_name="Exchange Energy")
    analyze_transition_peaks(Hext_range, trajectory_shift_mae, hd_error_mae, parameter_name="Hdemag Error")

    compare_transition_predictors(Hext_range, trajectory_shift_mae, predictors, csv_path=filename, parameter_names=list(predictors.keys()))