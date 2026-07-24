import os
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import os
import psutil

import torch
import torchvision.utils as vutils
import torchvision.transforms.functional as Func


class AverageMeter(object):
    """Computes and stores the average and current value.
       Code imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count





def tensor2rgb(tensor1, tensor2, tensor3, save_path):
    tensor1 = tensor1.to(torch.float)
    tensor2 = tensor2.to(torch.float)
    tensor3 = tensor3.to(torch.float)
    nrow = tensor1.shape[0]

    tensor1 = (tensor1 - tensor1.min()) / (tensor1.max() - tensor1.min())
    tensor2 = (tensor2 - tensor2.min()) / (tensor2.max() - tensor2.min())
    tensor3 = (tensor3 - tensor3.min()) / (tensor3.max() - tensor3.min())

    # 将张量转换为 RGB 图像，并将其拼接在一起
    combined_tensor = torch.cat((tensor1, tensor2, tensor3), dim=0)
    combined_image = vutils.make_grid(combined_tensor, nrow=nrow, padding=0)

    # 保存为 png 文件
    vutils.save_image(combined_image, save_path, normalize=True)



def vectorgraph(t1, t2, t3, save_path):
    mask = torch.where(t1 != 0, torch.ones_like(t1), torch.tensor(0.0, device=t1.device)).cpu().numpy()
    a1, a2, a3 = t1.cpu().numpy(), t2.cpu().numpy(), t3.cpu().numpy()
    fig, axs = plt.subplots(4, 10, figsize=(15, 7), dpi=1000)
    for i in range(len(a1)):
        arr = a1[i].transpose(1, 2, 0)
        axs[0, i].quiver(np.arange(arr.shape[0]), np.arange(arr.shape[1]), arr[:,:,0].T, arr[:,:,1].T, arr[:,:,2].T, clim=[-0.5, 0.5])
        axs[0, i].set_title('Input {}'.format(i))
        
        arr = a2[i].transpose(1, 2, 0)
        axs[1, i].quiver(np.arange(arr.shape[0]), np.arange(arr.shape[1]), arr[:,:,0].T, arr[:,:,1].T, arr[:,:,2].T, clim=[-0.5, 0.5])
        axs[1, i].set_title('Output {}'.format(i))
        
        arr = a3[i].transpose(1, 2, 0)
        axs[2, i].quiver(np.arange(arr.shape[0]), np.arange(arr.shape[1]), arr[:,:,0].T, arr[:,:,1].T, arr[:,:,2].T, clim=[-0.5, 0.5])
        axs[2, i].set_title('Label {}'.format(i))

        # Calculate MSE between a2 and a3 for this index
        mse = np.square(a2[i] - a3[i]).sum(axis=0)*(mask[i][0])
        mse = np.transpose(mse)
        axs[3, i].imshow(mse, cmap='hot', origin="lower")
        axs[3, i].set_title('MSE {:.1e} \n ({:.1e}, {:.1e})'.format(mse.mean(), mse.min(), mse.max()), fontsize=5)
        
    plt.tight_layout()
    fig.savefig(os.path.join(save_path))
    plt.close()






def create_mask(tensor):
    with torch.no_grad():
        device = tensor.device  # 获取张量所在的设备
        mask = torch.where(tensor != 0, torch.ones_like(tensor), torch.tensor(0.1, device=device))
    return mask

def mse(x, y):
    mse_tensor = torch.square(x-y)
    return mse_tensor

def SLA(x):
    return torch.where(x >= 0, torch.log(x+1), -torch.log(-x+1))

def ISLA(x):
    return torch.where(x >= 0, torch.exp(x)-1, -torch.exp(-x)+1)


def winding_density(spin_batch):
    """
    用于计算batch数据的winding density
    Args:
    spin_batch: torch.tensor
                形状为(batch_size, 3, 32, 32)的tensor，表示包含batch_size个样本的spin数据
    Returns:
    winding_density_batch: torch.tensor
                形状为(batch_size, 32, 32)的tensor，表示batch数据的winding density
    """
    # 调整spin的维度顺序为[batch_size, 32, 32, 1, 3]
    spin = torch.tensor(spin_batch).permute(0, 2, 3, 1).unsqueeze(-2)
    spin_xp = torch.roll(spin, shifts=-1, dims=1)
    spin_xm = torch.roll(spin, shifts=1, dims=1)
    spin_yp = torch.roll(spin, shifts=-1, dims=2)
    spin_ym = torch.roll(spin, shifts=1, dims=2)
    spin_xp[:, -1, :, :, :] = spin[:, -1, :, :, :]
    spin_xm[:, 0, :, :, :]  = spin[:, 0, :, :, :]
    spin_yp[:, :, -1, :, :] = spin[:, :, -1, :, :]
    spin_ym[:, :, 0, :, :]  = spin[:, :, 0, :, :]
    winding_density = (spin_xp[:,:,:, 0, 0] - spin_xm[:,:,:, 0, 0]) / 2 * (spin_yp[:,:,:, 0, 1] - spin_ym[:,:,:, 0, 1]) / 2 \
                    - (spin_xp[:,:,:, 0, 1] - spin_xm[:,:,:, 0, 1]) / 2 * (spin_yp[:,:,:, 0, 0] - spin_ym[:,:,:, 0, 0]) / 2
    
    winding_density = winding_density / np.pi
    winding_abs = torch.abs(winding_density).sum(dim=(1,2))

    return winding_density, torch.round(winding_abs).cpu().numpy()

def magnetic_divergence(spin_batch):
    """
    Computes magnetic charge density for a batch of magnetization fields

    Args:
        spin_batch: Tensor of shape (batch, 3, H, W)

    Returns:
        div_mag: Tensor of shape (batch, H, W)
    """

    spin = spin_batch

    Mx = spin[:,0,:,:]
    My = spin[:,1,:,:]

    Mx_xp = torch.roll(Mx,-1,dims=1)
    Mx_xm = torch.roll(Mx, 1,dims=1)
    My_yp = torch.roll(My,-1,dims=2)
    My_ym = torch.roll(My, 1,dims=2)

    dMxdx = (Mx_xp - Mx_xm)/2
    dMydy = (My_yp - My_ym)/2

    charge = -(dMxdx + dMydy)

    return charge

def gradient_magnitude(spin_batch):
    """
    Computes magnitude of spatial magnetization gradient

    Args:
        spin_batch: Tensor of shape (batch, 3, H, W)

    Returns:
        grad_mag: Tensor of shape (batch, H, W)
    """

    grad_sq = 0.0

    for c in range(3):

        M = spin_batch[:, c]

        M_xp = torch.roll(M, shifts=-1, dims=1)
        M_xm = torch.roll(M, shifts=1, dims=1)

        M_yp = torch.roll(M, shifts=-1, dims=2)
        M_ym = torch.roll(M, shifts=1, dims=2)

        # replicate edge values
        M_xp[:, -1, :] = M[:, -1, :]
        M_xm[:,  0, :] = M[:,  0, :]

        M_yp[:, :, -1] = M[:, :, -1]
        M_ym[:, :,  0] = M[:, :,  0]

        dMx = (M_xp - M_xm) / 2
        dMy = (M_yp - M_ym) / 2

        grad_sq += dMx**2 + dMy**2

    grad_mag = torch.sqrt(grad_sq)

    return grad_mag

import torch
import torch


def gradient_magnitude2(spin_batch, dx=1.0,dy=1.0, dz=1.0, eps=0.0,):
    """
    Calculate the magnitude of the spatial gradient of normalized
    magnetization m = (mx, my, mz).

    Supported input shapes
    ----------------------
    2D:
        (batch, 3, Nx, Ny)

        Calculates:
            sqrt(sum_c[(dm_c/dx)^2 + (dm_c/dy)^2])

    3D:
        (batch, 3, Nx, Ny, Nz)

        Calculates:
            sqrt(sum_c[
                (dm_c/dx)^2
                + (dm_c/dy)^2
                + (dm_c/dz)^2
            ])

    Args:
        spin_batch:
            Normalized magnetization tensor.

        dx, dy, dz:
            Spatial cell sizes. Use 1.0 to measure gradients per cell.

        eps:
            Optional small value added inside the square root.

    Returns:
        For 2D input:
            Tensor of shape (batch, Nx, Ny)

        For 3D input:
            Tensor of shape (batch, Nx, Ny, Nz)
    """

    if spin_batch.ndim not in (4, 5):
        raise ValueError(
            "spin_batch must have shape "
            "(batch, 3, Nx, Ny) or "
            "(batch, 3, Nx, Ny, Nz), "
            f"but received {tuple(spin_batch.shape)}."
        )

    if spin_batch.shape[1] != 3:
        raise ValueError(
            "spin_batch must contain exactly three magnetization "
            "channels: mx, my, and mz."
        )

    if dx <= 0 or dy <= 0 or dz <= 0:
        raise ValueError("dx, dy, and dz must all be positive.")

    is_3d = spin_batch.ndim == 5

    nx = spin_batch.shape[2]
    ny = spin_batch.shape[3]

    if nx < 2 or ny < 2:
        raise ValueError(
            "Nx and Ny must each contain at least two cells."
        )

    if is_3d:
        nz = spin_batch.shape[4]

        if nz < 2:
            raise ValueError(
                "A spatial z derivative requires at least two z layers. "
                "For Nz=1, dm/dz cannot be inferred from the data."
            )

    # Same spatial shape as one magnetization component.
    grad_sq = torch.zeros_like(spin_batch[:, 0])

    # Loop over mx, my, and mz.
    for component in range(3):
        m = spin_batch[:, component]

        dm_dx = torch.empty_like(m)
        dm_dy = torch.empty_like(m)

        # ----------------------------------------------------------
        # x derivative
        # ----------------------------------------------------------

        # Centered differences for interior cells.
        dm_dx[:, 1:-1, ...] = (
            m[:, 2:, ...] - m[:, :-2, ...]
        ) / (2.0 * dx)

        # One-sided differences at the x boundaries.
        dm_dx[:, 0, ...] = (
            m[:, 1, ...] - m[:, 0, ...]
        ) / dx

        dm_dx[:, -1, ...] = (
            m[:, -1, ...] - m[:, -2, ...]
        ) / dx

        # ----------------------------------------------------------
        # y derivative
        # ----------------------------------------------------------

        if is_3d:
            dm_dy[:, :, 1:-1, :] = (
                m[:, :, 2:, :] - m[:, :, :-2, :]
            ) / (2.0 * dy)

            dm_dy[:, :, 0, :] = (
                m[:, :, 1, :] - m[:, :, 0, :]
            ) / dy

            dm_dy[:, :, -1, :] = (
                m[:, :, -1, :] - m[:, :, -2, :]
            ) / dy

        else:
            dm_dy[:, :, 1:-1] = (
                m[:, :, 2:] - m[:, :, :-2]
            ) / (2.0 * dy)

            dm_dy[:, :, 0] = (
                m[:, :, 1] - m[:, :, 0]
            ) / dy

            dm_dy[:, :, -1] = (
                m[:, :, -1] - m[:, :, -2]
            ) / dy

        grad_sq = grad_sq + dm_dx.square() + dm_dy.square()

        # ----------------------------------------------------------
        # z derivative for genuinely 3D data
        # ----------------------------------------------------------

        if is_3d:
            dm_dz = torch.empty_like(m)

            dm_dz[:, :, :, 1:-1] = (
                m[:, :, :, 2:] - m[:, :, :, :-2]
            ) / (2.0 * dz)

            dm_dz[:, :, :, 0] = (
                m[:, :, :, 1] - m[:, :, :, 0]
            ) / dz

            dm_dz[:, :, :, -1] = (
                m[:, :, :, -1] - m[:, :, :, -2]
            ) / dz

            grad_sq = grad_sq + dm_dz.square()

    return torch.sqrt(grad_sq + eps)

def print_memory(msg=""):
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024**3)
    print(f"{msg} | RAM: {mem:.2f} GB", flush=True)

def tensor_rotate(tensor, symtype=None):
    #spins(bsz,w,h,channel)
    tensor=tensor.permute(0,2,3,1)

    if symtype=='RX': #x mirror 
        X_mirrored = torch.flip(tensor, [1])
        X_mirrored[:, :, :, 0] = -X_mirrored[:, :, :, 0]
        return X_mirrored.permute(0,3,1,2)
    
    elif symtype=='RY': #y mirror
        Y_mirrored = torch.flip(tensor, [2])
        Y_mirrored[:, :, :, 1] = -Y_mirrored[:, :, :, 1]
        return Y_mirrored.permute(0,3,1,2)
    
    elif symtype == 'R90': # 逆时针旋转90度
        spinrt90 = torch.rot90(tensor, k=1, dims=(1, 2))
        spinrt90[:, :, :, [0, 1]] = spinrt90[:, :, :, [1, 0]]
        spinrt90[:, :, :, 0] = -spinrt90[:, :, :, 0]
        return spinrt90.permute(0,3,1,2)
    
    elif symtype == 'R180': # 逆时针旋转180度
        spinrt180 = tensor.flip(1).flip(2)
        spinrt180[:, :, :, [0, 1]] = -spinrt180[:, :, :, [0, 1]]
        return spinrt180.permute(0,3,1,2)
    
    elif symtype == 'R270': # 逆时针旋转270度
        spinrt270 = torch.rot90(tensor, k=3, dims=(1, 2))
        spinrt270[:, :, :, [0, 1]] = spinrt270[:, :, :, [1, 0]]
        spinrt270[:, :, :, 1] = -spinrt270[:, :, :, 1]
        return spinrt270.permute(0,3,1,2)
     


def dataug(x,y):
    symtype=['R90', 'R180', 'R270', 'RX', 'RY']    
    selected_symmetry = random.choice(symtype)
    n=x.shape[0]//10
    xr_L1 = tensor_rotate(x[:n, :3, :, :], symtype=selected_symmetry)
    yr_L1 = tensor_rotate(y[:n, :3, :, :], symtype=selected_symmetry)
    xr_L2 = tensor_rotate(x[:n, 3:, :, :], symtype=selected_symmetry)
    yr_L2 = tensor_rotate(y[:n, 3:, :, :], symtype=selected_symmetry)
    xr = torch.cat((xr_L1, xr_L2), dim=1)
    yr = torch.cat((yr_L1, yr_L2), dim=1)
    x_combine = torch.cat((x, xr), dim=0)
    y_combine = torch.cat((y, yr), dim=0)
    return x_combine, y_combine



def visualize(mode, epoch, ex_path, x, y, ISLA_y, size):
    # Validate mode
    if mode not in ['eval', 'train']:
        raise ValueError("Mode must be 'eval' or 'train'")

    # Create necessary directories
    directories = [f'{size}rgb_{mode}', f'{size}vectgraph_{mode}']
    for dir_name in directories:
        os.makedirs(os.path.join(ex_path, dir_name), exist_ok=True)

    # Function to handle visualization for each layer
    def visualize_layer(ch_index, layer_name):
        tensor1 = x[:10, ch_index:ch_index+3, :, :].detach()
        tensor2 = ISLA_y[:10, ch_index:ch_index+3, :, :].detach()
        tensor3 = y[:10, ch_index:ch_index+3, :, :].detach()

        tensor2rgb(tensor1, tensor2, tensor3, f'{ex_path}/{size}rgb_{mode}/epoch{epoch}_{layer_name}.png')
        vectorgraph(tensor1, tensor2, tensor3, f'{ex_path}/{size}vectgraph_{mode}/epoch{epoch}_{layer_name}.png')

    # Visualize for each layer
    visualize_layer(0, 'L1')
    visualize_layer(3, 'L2')

# ------------------------------------------------------------------------
# RATE OF CHANGE FUNCTIONS
# ------------------------------------------------------------------------
import torch


def _temporal_spin_to_5d(spin_batch):
    """
    Convert NeuralMAG channel-first bilayer data to explicit layers.

    Input:
        (B, 3*L, Nx, Ny)
    Output:
        (B, 3, Nx, Ny, L)

    Channel order is assumed to be:
        [L0_mx, L0_my, L0_mz, L1_mx, L1_my, L1_mz, ...]
    which matches gen_data_new.py's reshape from (Nx, Ny, layers, 3).
    """
    if spin_batch.ndim != 4:
        raise ValueError(
            "spin_batch must have shape (batch, 3*layers, Nx, Ny), "
            f"but received {tuple(spin_batch.shape)}."
        )

    batch, channels, nx, ny = spin_batch.shape

    if channels % 3 != 0:
        raise ValueError(
            "The channel count must be divisible by 3 "
            "(mx, my, mz for each layer)."
        )

    layers = channels // 3

    return (
        spin_batch.reshape(batch, layers, 3, nx, ny)
        .permute(0, 2, 3, 4, 1)
        .contiguous()
    )


def _active_geometry(spin_5d, threshold=1.0e-12):
    """
    Return magnetic-cell mask with shape (B, Nx, Ny, L).
    """
    return torch.linalg.vector_norm(spin_5d, dim=1) > threshold


def _masked_first_derivative(values, active, dim, spacing):
    """
    First spatial derivative without differentiating through nonmagnetic cells.

    values:
        (B, 3, Nx, Ny, L)
    active:
        (B, Nx, Ny, L)
    dim:
        Spatial dimension in values: 2=x, 3=y, 4=z.

    Interior cells with two magnetic neighbors use a centered difference.
    Cells with only one magnetic neighbor use a one-sided difference.
    Cells with no magnetic neighbor, and nonmagnetic cells, get derivative 0.
    """
    if spacing <= 0:
        raise ValueError("Spatial spacing must be positive.")

    active5 = active.unsqueeze(1)

    plus = torch.roll(values, shifts=-1, dims=dim)
    minus = torch.roll(values, shifts=1, dims=dim)

    plus_active = torch.roll(active5, shifts=-1, dims=dim)
    minus_active = torch.roll(active5, shifts=1, dims=dim)

    # Invalidate wrapped neighbors at physical array boundaries.
    plus_slice = [slice(None)] * values.ndim
    plus_slice[dim] = -1
    plus_active[tuple(plus_slice)] = False

    minus_slice = [slice(None)] * values.ndim
    minus_slice[dim] = 0
    minus_active[tuple(minus_slice)] = False

    both = active5 & plus_active & minus_active
    only_plus = active5 & plus_active & ~minus_active
    only_minus = active5 & minus_active & ~plus_active

    derivative = torch.zeros_like(values)

    centered = (plus - minus) / (2.0 * spacing)
    forward = (plus - values) / spacing
    backward = (values - minus) / spacing

    derivative = torch.where(both, centered, derivative)
    derivative = torch.where(only_plus, forward, derivative)
    derivative = torch.where(only_minus, backward, derivative)

    return derivative


def magnetization_gradient_tensor(spin_batch, dx=1.0, dy=1.0, dz=1.0, active_mask=None, active_threshold=1.0e-12,):
    """
    Full spatial gradient tensor of normalized magnetization m.

    Input:
        spin_batch: (B, 3*L, Nx, Ny)

    Output:
        grad: (B, 3, Nx, Ny, L, 3)

    The last dimension is:
        [d/dx, d/dy, d/dz]

    For a bilayer (L=2), d/dz is necessarily one-sided because there
    are only two z samples. For L=1, d/dz is returned as zero.
    """
    spin_5d = _temporal_spin_to_5d(spin_batch)

    if active_mask is None:
        active = _active_geometry(spin_5d, active_threshold)
    else:
        active = active_mask.bool()
        expected = (spin_5d.shape[0],
                    spin_5d.shape[2],
                    spin_5d.shape[3],
                    spin_5d.shape[4],)
        if tuple(active.shape) != expected:
            raise ValueError(f"active_mask must have shape {expected}, "
                             f"but received {tuple(active.shape)}.")

    dm_dx = _masked_first_derivative(spin_5d, active, dim=2, spacing=dx)
    dm_dy = _masked_first_derivative(spin_5d, active, dim=3, spacing=dy)

    if spin_5d.shape[4] > 1:
        dm_dz = _masked_first_derivative(spin_5d, active, dim=4, spacing=dz)
    else:
        dm_dz = torch.zeros_like(spin_5d)

    return torch.stack((dm_dx, dm_dy, dm_dz), dim=-1)


def magnetization_gradient_rate(current_spin, previous_spin, dt=1.0, dx=1.0, dy=1.0, dz=1.0, active_threshold=1.0e-12,):
    r"""
    Backward finite-difference rate of the FULL magnetization-gradient tensor.

        || [grad(m_t) - grad(m_{t-1})] / dt ||_F

    The Frobenius norm is taken over:
        magnetization component (mx,my,mz),
        layer,
        spatial derivative direction (x,y,z).

    Returns:
        (B, Nx, Ny), suitable for the existing
        weight = 1 + alpha * abs(wd)
        training pattern.
    """
    if dt <= 0:
        raise ValueError("dt must be positive.")
    if current_spin.shape != previous_spin.shape:
        raise ValueError("current_spin and previous_spin must have identical shapes.")

    current_5d = _temporal_spin_to_5d(current_spin)
    previous_5d = _temporal_spin_to_5d(previous_spin)

    active_current = _active_geometry(current_5d, active_threshold)
    active_previous = _active_geometry(previous_5d, active_threshold)

    if not torch.equal(active_current, active_previous):
        raise ValueError("Current and previous states do not have the same magnetic "
                         "geometry. Their samples may be misaligned.")

    grad_current = magnetization_gradient_tensor(current_spin, dx, dy, dz, active_current, active_threshold)
    grad_previous = magnetization_gradient_tensor(previous_spin, dx, dy, dz, active_current, active_threshold)

    grad_rate = (grad_current - grad_previous) / dt

    # grad_rate shape: (B, 3, Nx, Ny, L, 3)
    return torch.sqrt(torch.sum(grad_rate.square(), dim=(1, 4, 5)))


def magnetization_difference_gradient_rate(current_spin, previous_spin, dt=1.0, dx=1.0, dy=1.0, dz=1.0, active_threshold=1.0e-12,):
    r"""
    Gradient AFTER temporal subtraction:

        || grad[(m_t - m_{t-1}) / dt] ||_F

    With the same linear finite-difference operator and geometry mask, this is
    mathematically the same quantity as magnetization_gradient_rate().
    This function is provided explicitly so the two formulations can be checked.
    """
    if dt <= 0:
        raise ValueError("dt must be positive.")
    if current_spin.shape != previous_spin.shape:
        raise ValueError("current_spin and previous_spin must have identical shapes.")

    current_5d = _temporal_spin_to_5d(current_spin)
    previous_5d = _temporal_spin_to_5d(previous_spin)

    active_current = _active_geometry(current_5d, active_threshold)
    active_previous = _active_geometry(previous_5d, active_threshold)

    if not torch.equal(active_current, active_previous):
        raise ValueError(
            "Current and previous states do not have the same magnetic "
            "geometry. Their samples may be misaligned."
        )

    dm_dt = (current_spin - previous_spin) / dt

    grad_dm_dt = magnetization_gradient_tensor(
        dm_dt,
        dx=dx,
        dy=dy,
        dz=dz,
        active_mask=active_current,
        active_threshold=active_threshold,
    )

    return torch.sqrt(
        torch.sum(grad_dm_dt.square(), dim=(1, 4, 5))
    )


def magnetization_gradient_magnitude_rate(current_spin, previous_spin, dt=1.0, dx=1.0, dy=1.0, dz=1.0, active_threshold=1.0e-12,):
    r"""
    Backward rate of change of the scalar gradient magnitude:

        | ||grad(m_t)||_F - ||grad(m_{t-1})||_F | / dt

    This is genuinely different from magnetization_gradient_rate().
    It only tracks change in gradient MAGNITUDE and discards changes in the
    orientation/sign/structure of the gradient tensor.

    Returns:
        (B, Nx, Ny)
    """
    if dt <= 0:
        raise ValueError("dt must be positive.")
    if current_spin.shape != previous_spin.shape:
        raise ValueError("current_spin and previous_spin must have identical shapes.")

    current_5d = _temporal_spin_to_5d(current_spin)
    previous_5d = _temporal_spin_to_5d(previous_spin)

    active_current = _active_geometry(current_5d, active_threshold)
    active_previous = _active_geometry(previous_5d, active_threshold)

    if not torch.equal(active_current, active_previous):
        raise ValueError("Current and previous states do not have the same magnetic "
                         "geometry. Their samples may be misaligned.")

    grad_current = magnetization_gradient_tensor(current_spin, dx, dy, dz, active_current, active_threshold)
    grad_previous = magnetization_gradient_tensor(previous_spin, dx, dy, dz, active_current, active_threshold)

    grad_mag_current = torch.sqrt(torch.sum(grad_current.square(), dim=(1, 4, 5)))
    grad_mag_previous = torch.sqrt(torch.sum(grad_previous.square(), dim=(1, 4, 5)))

    return torch.abs(grad_mag_current - grad_mag_previous) / dt


def dataug_temporal(x, y, x_prev, x_next=None):
    """
    Temporal-safe version of the repository's data augmentation.

    The SAME randomly chosen symmetry is applied to current m, target Hd,
    previous m, and (optionally) next m. This is essential: applying different
    transforms to neighboring states would corrupt the temporal finite difference.

    Assumes tensor_rotate() and random are already available in utils.py.
    """
    symtype = ['R90', 'R180', 'R270', 'RX', 'RY']
    selected_symmetry = random.choice(symtype)
    n = x.shape[0] // 10

    if n == 0:
        if x_next is None:
            return x, y, x_prev
        return x, y, x_prev, x_next

    def augment_first_n(tensor):
        if tensor.shape[1] % 3 != 0:
            raise ValueError("Channel count must be divisible by 3.")

        transformed_layers = []
        for start in range(0, tensor.shape[1], 3):
            transformed_layers.append(
                tensor_rotate(
                    tensor[:n, start:start + 3, :, :],
                    symtype=selected_symmetry,
                )
            )
        transformed = torch.cat(transformed_layers, dim=1)
        return torch.cat((tensor, transformed), dim=0)

    x_aug = augment_first_n(x)
    y_aug = augment_first_n(y)
    prev_aug = augment_first_n(x_prev)

    if x_next is None:
        return x_aug, y_aug, prev_aug

    next_aug = augment_first_n(x_next)
    return x_aug, y_aug, prev_aug, next_aug

