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


def get_memory_stats():
    """
    Return current process RAM, DataLoader-worker RAM, and CUDA memory.

    The old print_memory() reported only the parent Python process RSS, which can
    badly under-report a Slurm job that has many DataLoader worker processes.
    """
    process = psutil.Process(os.getpid())
    main_rss = process.memory_info().rss

    workers_rss = 0
    worker_count = 0
    for child in process.children(recursive=True):
        try:
            workers_rss += child.memory_info().rss
            worker_count += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    stats = {
        "ram_main_gb": main_rss / (1024**3),
        "ram_workers_gb": workers_rss / (1024**3),
        "ram_total_with_workers_gb": (main_rss + workers_rss) / (1024**3),
        "worker_processes": worker_count,
    }
    if torch.cuda.is_available():
        stats.update({
            "gpu_allocated_gb": torch.cuda.memory_allocated() / (1024**3),
            "gpu_reserved_gb": torch.cuda.memory_reserved() / (1024**3),
            "gpu_max_allocated_gb": torch.cuda.max_memory_allocated() / (1024**3),
            "gpu_max_reserved_gb": torch.cuda.max_memory_reserved() / (1024**3),
        })
    else:
        stats.update({
            "gpu_allocated_gb": 0.0,
            "gpu_reserved_gb": 0.0,
            "gpu_max_allocated_gb": 0.0,
            "gpu_max_reserved_gb": 0.0,
        })

    return stats


def print_memory(msg=""):
    stats = get_memory_stats()
    text = (
        f"{msg} | RAM main: {stats['ram_main_gb']:.2f} GB | "
        f"RAM workers: {stats['ram_workers_gb']:.2f} GB "
        f"({stats['worker_processes']} workers) | "
        f"RAM total observed: {stats['ram_total_with_workers_gb']:.2f} GB"
    )
    if torch.cuda.is_available():
        text += (
            f" | GPU allocated/reserved: "
            f"{stats['gpu_allocated_gb']:.2f}/{stats['gpu_reserved_gb']:.2f} GB | "
            f"GPU peak allocated/reserved: "
            f"{stats['gpu_max_allocated_gb']:.2f}/{stats['gpu_max_reserved_gb']:.2f} GB"
        )
    print(text, flush=True)
    return stats


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


def demag_torque_mismatch_loss(spin_batch, predicted_hd, true_hd, active_threshold=1.0e-12):
    """
    Mean squared demagnetizing-torque mismatch over magnetic cells:

        || m x (Hdemag_pred - Hdemag_true) ||^2

    predicted_hd and true_hd must both be in physical Hdemag units.
    Therefore call this using ISLA(pred_y), NOT pred_y.
    """
    _same_shape(spin_batch, predicted_hd, "spin/predicted Hd")
    _same_shape(spin_batch, true_hd, "spin/true Hd")

    m = _nm_channels_to_grid(spin_batch)

    hd_error = (_nm_channels_to_grid(predicted_hd) - _nm_channels_to_grid(true_hd))
    torque_error = torch.linalg.cross(m, hd_error, dim=-1)
    torque_error_sq = torch.sum(torque_error.square(), dim=-1)

    active = (torch.linalg.vector_norm(m, dim=-1) > active_threshold)
    active_count = active.sum().clamp_min(1)

    return (torque_error_sq * active.to(torque_error_sq.dtype)).sum() / active_count

# ------------------------------------------------------------------------
# RATE OF CHANGE FUNCTIONS
# ------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Tensor layout helpers
# ---------------------------------------------------------------------------
def _nm_channels_to_grid(spin_batch):
    """
    (B, 3*L, Nx, Ny) -> (B, Nx, Ny, L, 3)
    """
    if spin_batch.ndim != 4:
        raise ValueError("Expected spin tensor shape (B, 3*layers, Nx, Ny), "
                         f"got {tuple(spin_batch.shape)}.")

    batch, channels, nx, ny = spin_batch.shape
    if channels % 3 != 0:
        raise ValueError(f"Channel count {channels} is not divisible by 3.")

    layers = channels // 3

    return (spin_batch.reshape(batch, layers, 3, nx, ny).permute(0, 3, 4, 1, 2).contiguous())


def _nm_grid_to_channels(grid):
    """
    (B, Nx, Ny, L, 3) -> (B, 3*L, Nx, Ny)
    """
    if grid.ndim != 5 or grid.shape[-1] != 3:
        raise ValueError("Expected grid tensor shape (B, Nx, Ny, layers, 3).")

    b, nx, ny, layers, _ = grid.shape

    return (grid.permute(0, 3, 4, 1, 2).contiguous().reshape(b, 3 * layers, nx, ny))


def _same_shape(current, previous, name="tensor"):
    if current.shape != previous.shape:
        raise ValueError(
            f"Current and previous {name} tensors must have identical shapes; "
            f"got {tuple(current.shape)} and {tuple(previous.shape)}.")


def _collapse_vector_layers(vector_grid):
    """
    Collapse (B, Nx, Ny, L, 3) into one scalar map (B, Nx, Ny)
    using the Euclidean norm over layer and vector-component axes.
    """
    return torch.sqrt(torch.sum(vector_grid.square(), dim=(-1, -2)))


def _collapse_scalar_layers_rms(scalar_grid):
    """
    Collapse (B, Nx, Ny, L) into (B, Nx, Ny) using RMS over layers.
    """
    return torch.sqrt(torch.mean(scalar_grid.square(), dim=-1))


# ---------------------------------------------------------------------------
# Non-periodic spatial finite differences
# ---------------------------------------------------------------------------

def _shift_nonperiodic(values, shift, dim, boundary="replicate"):
    """
    torch.roll with wrapped elements replaced at the physical array boundary.

    boundary='replicate' matches the non-PBC style used by MAG2305 torch_roll
    at the outer rectangular box.
    """
    shifted = torch.roll(values, shifts=shift, dims=dim)

    index = [slice(None)] * values.ndim

    if shift == 1:
        index[dim] = 0
        if boundary == "replicate":
            shifted[tuple(index)] = values[tuple(index)]
        elif boundary == "zero":
            shifted[tuple(index)] = 0
        else:
            raise ValueError("boundary must be 'replicate' or 'zero'.")

    elif shift == -1:
        index[dim] = -1
        if boundary == "replicate":
            shifted[tuple(index)] = values[tuple(index)]
        elif boundary == "zero":
            shifted[tuple(index)] = 0
        else:
            raise ValueError("boundary must be 'replicate' or 'zero'.")

    else:
        raise ValueError("_shift_nonperiodic only supports shift +/-1.")

    return shifted


def _first_derivative_all_cells(values, dim, spacing):
    """
    First spatial derivative using centered differences in the interior
    and true one-sided differences at the outer grid boundaries.
    """
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


def magnetization_gradient_tensor_3d(spin_batch, dx=1.0, dy=1.0, dz=1.0,):
    """
    Full spatial gradient tensor grad(m) for both magnetic layers.

    Input:
        spin_batch : (B, 3*L, Nx, Ny)

    Internal representation:
        m : (B, Nx, Ny, L, 3)

    Output:
        grad : (B, Nx, Ny, L, 3, 3)

    grad[..., component, derivative_direction]
    derivative_direction order = (x, y, z)

    For L=2, d/dz is necessarily a two-layer finite difference. There is no
    centered three-point z stencil because only two z cell centers exist.
    """
    m = _nm_channels_to_grid(spin_batch)

    # m dimensions: B, x, y, z(layer), vector-component
    dm_dx = _first_derivative_all_cells(m, dim=1, spacing=dx)
    dm_dy = _first_derivative_all_cells(m, dim=2, spacing=dy)

    if m.shape[3] == 1:
        dm_dz = torch.zeros_like(m)
    elif m.shape[3] == 2:
        # With two layers, the physically available first-order derivative
        # is the difference between the two layer-center values.
        dz_pair = (m[:, :, :, 1, :] - m[:, :, :, 0, :]) / dz
        dm_dz = torch.stack((dz_pair, dz_pair), dim=3)
    else:
        dm_dz = _first_derivative_all_cells(m, dim=3, spacing=dz)

    return torch.stack((dm_dx, dm_dy, dm_dz), dim=-1)


def magnetization_gradient_magnitude_3d(spin_batch, dx=1.0, dy=1.0, dz=1.0):
    r"""
    Full bilayer Frobenius magnitude ||grad(m)||_F at each x-y location.

    Returns:
        (B, Nx, Ny)
    """
    grad = magnetization_gradient_tensor_3d(spin_batch, dx=dx, dy=dy, dz=dz)

    # sum over layer, m component, derivative direction
    return torch.sqrt(torch.sum(grad.square(), dim=(3, 4, 5)))


# ---------------------------------------------------------------------------
# The TWO genuinely different magnetization-gradient rate definitions
# ---------------------------------------------------------------------------

def gradient_magnitude_rate(
    current_spin,
    previous_spin,
    dt=1.0,
    dx=1.0,
    dy=1.0,
    dz=1.0,
):
    r"""
    Rate of change of the SCALAR gradient magnitude:

        | ||grad(m_t)||_F - ||grad(m_(t-1))||_F | / dt

    This detects changes in how strong the spatial nonuniformity is.

    Returns:
        (B, Nx, Ny), nonnegative.
    """
    _same_shape(current_spin, previous_spin, "spin")

    if dt <= 0:
        raise ValueError("dt must be positive.")

    g_now = magnetization_gradient_magnitude_3d(
        current_spin, dx=dx, dy=dy, dz=dz
    )
    g_prev = magnetization_gradient_magnitude_3d(
        previous_spin, dx=dx, dy=dy, dz=dz
    )

    return torch.abs(g_now - g_prev) / dt


def gradient_tensor_rate(
    current_spin,
    previous_spin,
    dt=1.0,
    dx=1.0,
    dy=1.0,
    dz=1.0,
):
    r"""
    Rate of change of the FULL gradient tensor:

        || grad(m_t) - grad(m_(t-1)) ||_F / dt

    This detects changes in gradient direction/tensor structure as well as
    changes in gradient magnitude.

    Returns:
        (B, Nx, Ny), nonnegative.
    """
    _same_shape(current_spin, previous_spin, "spin")

    if dt <= 0:
        raise ValueError("dt must be positive.")

    grad_now = magnetization_gradient_tensor_3d(
        current_spin, dx=dx, dy=dy, dz=dz
    )
    grad_prev = magnetization_gradient_tensor_3d(
        previous_spin, dx=dx, dy=dy, dz=dz
    )

    delta_grad = (grad_now - grad_prev) / dt

    return torch.sqrt(
        torch.sum(delta_grad.square(), dim=(3, 4, 5))
    )


def delta_m_gradient_rate(
    current_spin,
    previous_spin,
    dt=1.0,
    dx=1.0,
    dy=1.0,
    dz=1.0,
):
    r"""
    Explicitly calculate:

        || grad[(m_t - m_(t-1))/dt] ||_F

    With the same linear finite-difference operator, this is algebraically
    equivalent to gradient_tensor_rate(). It is included as a numerical
    cross-check, not as a separate physics loss candidate.
    """
    _same_shape(current_spin, previous_spin, "spin")

    if dt <= 0:
        raise ValueError("dt must be positive.")

    dm_dt = (current_spin - previous_spin) / dt
    grad_dm_dt = magnetization_gradient_tensor_3d(
        dm_dt, dx=dx, dy=dy, dz=dz
    )

    return torch.sqrt(
        torch.sum(grad_dm_dt.square(), dim=(3, 4, 5))
    )


# ---------------------------------------------------------------------------
# Exact repository-style exchange field
# ---------------------------------------------------------------------------

def exchange_field(
    spin_batch,
    Ms=1000.0,
    Ax=0.5e-6,
    cell_nm=(3.0, 3.0, 3.0),
):
    r"""
    Reproduce MAG2305's uniform-material Heisenberg exchange field:

        H_ex = sum_neighbors Hx0_neighbor * (m_neighbor - m_center)

    for a finite ('bulk') rectangular box.

    For uniform Ms and Ax, MAG2305's neighbor coefficient reduces to:

        Hx0 = 2 * 1e14 * Ax / (Ms * D^2)   [Oe]

    separately for x, y, z cell sizes.

    The zero-spin masked cells are intentionally NOT removed from this
    calculation. That matches the current data generator / MAG2305 behavior:
    the geometry was made by multiplying spin by a zero mask rather than by
    changing film.model.

    Returns:
        (B, Nx, Ny, L, 3), exchange field in Oe.
    """
    if Ms <= 0:
        raise ValueError("Ms must be positive.")
    if Ax < 0:
        raise ValueError("Ax must be nonnegative.")

    if len(cell_nm) != 3 or any(float(d) <= 0 for d in cell_nm):
        raise ValueError("cell_nm must contain three positive cell sizes.")

    m = _nm_channels_to_grid(spin_batch)

    coeff_x = 2.0e14 * Ax / (Ms * float(cell_nm[0]) ** 2)
    coeff_y = 2.0e14 * Ax / (Ms * float(cell_nm[1]) ** 2)
    coeff_z = 2.0e14 * Ax / (Ms * float(cell_nm[2]) ** 2)

    hx = coeff_x * (
        (_shift_nonperiodic(m, 1, 1) - m)
        + (_shift_nonperiodic(m, -1, 1) - m)
    )

    hy = coeff_y * (
        (_shift_nonperiodic(m, 1, 2) - m)
        + (_shift_nonperiodic(m, -1, 2) - m)
    )

    hz = coeff_z * (
        (_shift_nonperiodic(m, 1, 3) - m)
        + (_shift_nonperiodic(m, -1, 3) - m)
    )

    return hx + hy + hz


def exchange_field_rate(
    current_spin,
    previous_spin,
    dt=1.0,
    Ms=1000.0,
    Ax=0.5e-6,
    cell_nm=(3.0, 3.0, 3.0),
):
    r"""
    || H_ex(t) - H_ex(t-1) || / dt

    Norm is over vector components and both layers.

    Returns:
        (B, Nx, Ny), nonnegative.
    """
    _same_shape(current_spin, previous_spin, "spin")
    if dt <= 0:
        raise ValueError("dt must be positive.")

    h_now = exchange_field(current_spin, Ms, Ax, cell_nm)
    h_prev = exchange_field(previous_spin, Ms, Ax, cell_nm)

    return _collapse_vector_layers((h_now - h_prev) / dt)


# ---------------------------------------------------------------------------
# Exchange energy density
# ---------------------------------------------------------------------------

def exchange_energy_density(
    spin_batch,
    Ms=1000.0,
    Ax=0.5e-6,
    cell_nm=(3.0, 3.0, 3.0),
):
    r"""
    Local exchange energy density matching MAG2305's detailed-energy form:

        e_ex = -0.5 * Ms * m dot H_ex

    Returns:
        (B, Nx, Ny, L)

    In cgs units this has the energy-density scale erg/cc.
    """
    m = _nm_channels_to_grid(spin_batch)
    h_ex = exchange_field(spin_batch, Ms, Ax, cell_nm)

    return -0.5 * Ms * torch.sum(m * h_ex, dim=-1)


def exchange_energy_density_rate(
    current_spin,
    previous_spin,
    dt=1.0,
    Ms=1000.0,
    Ax=0.5e-6,
    cell_nm=(3.0, 3.0, 3.0),
):
    r"""
    RMS-over-layer rate of change of exchange energy density:

        RMS_layers( [e_ex(t) - e_ex(t-1)] / dt )

    Returns:
        (B, Nx, Ny), nonnegative.
    """
    _same_shape(current_spin, previous_spin, "spin")
    if dt <= 0:
        raise ValueError("dt must be positive.")

    e_now = exchange_energy_density(
        current_spin, Ms=Ms, Ax=Ax, cell_nm=cell_nm
    )
    e_prev = exchange_energy_density(
        previous_spin, Ms=Ms, Ax=Ax, cell_nm=cell_nm
    )

    return _collapse_scalar_layers_rms((e_now - e_prev) / dt)


# ---------------------------------------------------------------------------
# Demagnetizing-field and torque rates
# ---------------------------------------------------------------------------

def demag_field_rate(
    current_hd,
    previous_hd,
    dt=1.0,
):
    r"""
    Direct target-field rate:

        || H_d(t) - H_d(t-1) || / dt

    Returns:
        (B, Nx, Ny), nonnegative.
    """
    _same_shape(current_hd, previous_hd, "demag-field")
    if dt <= 0:
        raise ValueError("dt must be positive.")

    hd_now = _nm_channels_to_grid(current_hd)
    hd_prev = _nm_channels_to_grid(previous_hd)

    return _collapse_vector_layers((hd_now - hd_prev) / dt)


def demag_torque(
    spin_batch,
    hd_batch,
):
    r"""
    Demagnetizing precessional torque proxy:

        tau_d = m x H_d

    Returns:
        (B, Nx, Ny, L, 3)
    """
    _same_shape(spin_batch, hd_batch, "spin/Hd")

    m = _nm_channels_to_grid(spin_batch)
    hd = _nm_channels_to_grid(hd_batch)

    return torch.linalg.cross(m, hd, dim=-1)


def demag_torque_rate(
    current_spin,
    previous_spin,
    current_hd,
    previous_hd,
    dt=1.0,
):
    r"""
    || [m_t x H_d,t] - [m_(t-1) x H_d,t-1] || / dt

    Returns:
        (B, Nx, Ny), nonnegative.
    """
    _same_shape(current_spin, previous_spin, "spin")
    _same_shape(current_hd, previous_hd, "demag-field")
    _same_shape(current_spin, current_hd, "spin/Hd")

    if dt <= 0:
        raise ValueError("dt must be positive.")

    tau_now = demag_torque(current_spin, current_hd)
    tau_prev = demag_torque(previous_spin, previous_hd)

    return _collapse_vector_layers((tau_now - tau_prev) / dt)


# ---------------------------------------------------------------------------
# Exchange torque rate
# ---------------------------------------------------------------------------

def exchange_torque(
    spin_batch,
    Ms=1000.0,
    Ax=0.5e-6,
    cell_nm=(3.0, 3.0, 3.0),
):
    r"""
    Exchange precessional torque proxy:

        tau_ex = m x H_ex

    Returns:
        (B, Nx, Ny, L, 3)
    """
    m = _nm_channels_to_grid(spin_batch)
    h_ex = exchange_field(spin_batch, Ms, Ax, cell_nm)

    return torch.linalg.cross(m, h_ex, dim=-1)


def exchange_torque_rate(
    current_spin,
    previous_spin,
    dt=1.0,
    Ms=1000.0,
    Ax=0.5e-6,
    cell_nm=(3.0, 3.0, 3.0),
):
    r"""
    || tau_ex(t) - tau_ex(t-1) || / dt

    Returns:
        (B, Nx, Ny), nonnegative.
    """
    _same_shape(current_spin, previous_spin, "spin")
    if dt <= 0:
        raise ValueError("dt must be positive.")

    tau_now = exchange_torque(current_spin, Ms, Ax, cell_nm)
    tau_prev = exchange_torque(previous_spin, Ms, Ax, cell_nm)

    return _collapse_vector_layers((tau_now - tau_prev) / dt)


# ---------------------------------------------------------------------------
# Winding-density rate (lower-priority ablation)
# ---------------------------------------------------------------------------

def winding_density_per_layer(spin_batch):
    r"""
    Layer-resolved version of the repository's current winding_density formula.

    It intentionally follows the SAME in-plane expression already used by the
    NeuralMAG training utilities:

        [(d mx/dx)(d my/dy) - (d my/dx)(d mx/dy)] / pi

    This is kept for direct comparability with the user's earlier winding loss.
    It is not being relabeled as skyrmion/topological-charge density.

    Returns:
        (B, Nx, Ny, L)
    """
    m = _nm_channels_to_grid(spin_batch)

    mx = m[..., 0]
    my = m[..., 1]

    mx_xp = _shift_nonperiodic(mx, -1, 1)
    mx_xm = _shift_nonperiodic(mx, 1, 1)
    my_xp = _shift_nonperiodic(my, -1, 1)
    my_xm = _shift_nonperiodic(my, 1, 1)

    mx_yp = _shift_nonperiodic(mx, -1, 2)
    mx_ym = _shift_nonperiodic(mx, 1, 2)
    my_yp = _shift_nonperiodic(my, -1, 2)
    my_ym = _shift_nonperiodic(my, 1, 2)

    dmx_dx = (mx_xp - mx_xm) / 2.0
    dmy_dx = (my_xp - my_xm) / 2.0
    dmx_dy = (mx_yp - mx_ym) / 2.0
    dmy_dy = (my_yp - my_ym) / 2.0

    return (
        dmx_dx * dmy_dy - dmy_dx * dmx_dy
    ) / torch.pi


def winding_density_rate(
    current_spin,
    previous_spin,
    dt=1.0,
):
    r"""
    RMS-over-layer local rate of change of the existing winding-density map:

        RMS_layers( [w_t - w_(t-1)] / dt )

    Returns:
        (B, Nx, Ny), nonnegative.
    """
    _same_shape(current_spin, previous_spin, "spin")
    if dt <= 0:
        raise ValueError("dt must be positive.")

    w_now = winding_density_per_layer(current_spin)
    w_prev = winding_density_per_layer(previous_spin)

    return _collapse_scalar_layers_rms((w_now - w_prev) / dt)


# ---------------------------------------------------------------------------
# Safe temporal data augmentation
# ---------------------------------------------------------------------------

def dataug_temporal_physics(
    x,
    y,
    x_prev,
    y_prev,
    x_next=None,
    y_next=None,
):
    """
    Apply ONE identical physical symmetry to all temporally aligned arrays.

    This replaces separate dataug() calls, which would corrupt temporal rates if
    current and previous states receive different randomly selected symmetries.

    Requires the existing tensor_rotate() function in utils.py.
    """
    tensors = [x, y, x_prev, y_prev]

    if x_next is not None:
        tensors.append(x_next)
    if y_next is not None:
        tensors.append(y_next)

    n = x.shape[0] // 10
    if n == 0:
        return tuple(tensors)

    selected_symmetry = random.choice(
        ['R90', 'R180', 'R270', 'RX', 'RY']
    )

    def _augment_one(tensor):
        if tensor.ndim != 4 or tensor.shape[1] % 3 != 0:
            raise ValueError(
                "Each augmented tensor must have shape "
                "(B, 3*layers, Nx, Ny)."
            )

        transformed_layers = []

        for start in range(0, tensor.shape[1], 3):
            transformed_layers.append(
                tensor_rotate(
                    tensor[:n, start:start + 3],
                    symtype=selected_symmetry,
                )
            )

        transformed = torch.cat(transformed_layers, dim=1)
        return torch.cat((tensor, transformed), dim=0)

    return tuple(_augment_one(t) for t in tensors)


def gradient_magnitude(spin_batch, dx=1.0, dy=1.0, dz=1.0):
    """
    Full bilayer magnetization-gradient magnitude.

    Input:
        spin_batch: (B, 3*layers, Nx, Ny)

    Returns:
        (B, Nx, Ny)
    """

    grad = magnetization_gradient_tensor_3d(
        spin_batch,
        dx=dx,
        dy=dy,
        dz=dz,
    )

    # grad:
    # (B, Nx, Ny, layers, m_component, spatial_direction)

    return torch.sqrt(
        torch.sum(
            grad ** 2,
            dim=(3, 4, 5)
        )
    )