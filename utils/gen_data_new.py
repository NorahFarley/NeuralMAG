# -*- coding: utf-8 -*-
"""
Temporal training-data generation for NeuralMAG.

Each saved training example is centered on one exact FFT/LLG state t and contains
three consecutive state/field pairs:
    (m_{t-1}, Hd_{t-1}), (m_t, Hd_t), (m_{t+1}, Hd_{t+1}).

Spins.npy and Hds.npy contain the center state so the existing single-state loader
can remain compatible. The four additional arrays provide the neighboring states
needed for centered, forward, or backward temporal differences.
"""

import argparse
import json
import os
import shutil
import time
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

from libs.misc import Culist, initial_spin_prepare, create_random_mask, error_plot
import libs.MAG2305 as MAG2305


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate consecutive NeuralMAG FFT/LLG training triplets"
    )
    parser.add_argument("--gpu",        type=int,    default=0,         help="GPU ID (default: 0)")
    parser.add_argument('--w',          type=int,    default=32,        help='MAG model grid size (default: 32)')
    parser.add_argument('--layers',     type=int,    default=2,         help='MAG model layers (default: 2)')
    parser.add_argument('--Ms',         type=float,  default=1000,      help='MAG model Ms (default: 1000)')
    parser.add_argument('--Ax',         type=float,  default=0.5e-6,    help='MAG model Ax (default: 0.5e-6)')
    parser.add_argument('--Ku',         type=float,  default=0.0,       help='MAG model Ku (default: 0.0)')
    parser.add_argument('--Kvec',       type=Culist, default=(0,0,1),   help='MAG model Kvec (default: (0,0,1))')

    parser.add_argument('--damping',    type=float,  default=0.1,       help='MAG model damping (default: 0.1)')
    parser.add_argument('--dtime',      type=float,  default=1.0e-13,   help='real time step (default: 1.0e-13)')
    parser.add_argument('--error_min',  type=float,  default=1.0e-5,    help='min error (default: 1.0e-5)')
    parser.add_argument("--field-mode", choices=("random", "zero"), default="random", help="random gives one constant in-plane field of 100-1000 Oe per simulation",)
    parser.add_argument('--max_iter',   type=int,    default=50000,     help='max iteration number (default: 50000)')
    parser.add_argument('--sav_samples', type=int,   default=500,       help='saved center-state triplets per simulation (default: 500)')

    parser.add_argument('--masked',     type=int,    default=0,         help='number of masked simulations to generate (default: 0)')
    parser.add_argument('--unmasked',   type=int,    default=0,         help='number of unmasked simulations to generate (default: 0)')
    parser.add_argument('--seed-start', type=int,    default=0,         help='first seed ID used within each requested group (default: 0)')
    parser.add_argument('--dataset-seed', type=int,  default=2345,      help='base seed used to derive reproducible independent RNG streams')

    parser.add_argument('--mask-min-points', type=int, default=3)
    parser.add_argument('--mask-max-points', type=int, default=0, help='0 means w-1, matching the broad point-count range of the repository generator',)
    parser.add_argument('--output-root', type=str, default='./Dataset/rate_change')
    parser.add_argument('--allow-unconverged', action='store_true', help='save a trajectory that reaches max_iter before the convergence threshold',)
    parser.add_argument('--overwrite', action='store_true', help='replace an existing output directory for the same masked/unmasked seed',)
    return parser


def validate_generation_request(args: argparse.Namespace) -> None:
    if args.masked < 0 or args.unmasked < 0:
        raise ValueError('--masked and --unmasked must both be nonnegative integers')
    if args.masked == 0 and args.unmasked == 0:
        raise ValueError('Nothing was requested. Set --masked, --unmasked, or both above zero.')
    if args.seed_start < 0:
        raise ValueError('--seed-start must be nonnegative')
    if args.sav_samples <= 0:
        raise ValueError('--sav_samples must be positive')


def prepare_model(args: argparse.Namespace):
    film = MAG2305.mmModel(types='bulk', size=(args.w, args.w, args.layers), cell=(3,3,3), Ms=args.Ms, Ax=args.Ax, 
                           Ku=args.Ku, Kvec=args.Kvec, device=f'cuda:{args.gpu}',)

    print(f'Creating {args.layers}-layer model')
    print(f'Convergence threshold: {args.error_min:.3e}')

    start = time.time()
    film.DemagInit()
    print(f'Demag initialization time: {time.time() - start:.3f} s\n')

    return film


def requested_simulations(args: argparse.Namespace) -> List[Tuple[int, bool]]:
    """
    Return (simulation_id, masked) jobs for exactly the requested counts.

    Masked and unmasked simulations are stored in separate directories, so both
    groups may use the same seed IDs without colliding. The masked flag is also
    included in the RNG seed sequence, so masked seed 0 and unmasked seed 0 do
    not receive identical initial spins or external fields.
    """
    jobs: List[Tuple[int, bool]] = []

    for offset in range(args.masked):
        jobs.append((args.seed_start + offset, True))

    for offset in range(args.unmasked):
        jobs.append((args.seed_start + offset, False))

    return jobs


def random_in_plane_field(rng: np.random.Generator,) -> Tuple[np.ndarray, float, float]:
    """
    Draw one field with magnitude in [100, 1000] Oe and angle in [0, 2pi).
    """
    angle = float(rng.uniform(0.0, 2.0 * np.pi))
    magnitude = float(rng.uniform(100.0, 1000.0))
    field = np.asarray([magnitude * np.cos(angle), magnitude * np.sin(angle), 0.0], dtype=np.float64,)
    
    return field, magnitude, angle


def random_mask_with_isolated_seed(args: argparse.Namespace, mask_seed: int,) -> Tuple[np.ndarray, int]:
    """
    Generate a random convex polygon without disturbing the other RNG streams.

    create_random_mask uses NumPy's legacy global RNG. fixshape=False is required
    here so the function does not reset itself to its default fixed seed.
    """
    max_points = args.mask_max_points if args.mask_max_points > 0 else args.w - 1
    max_points = min(max_points, args.w - 1)

    if args.mask_min_points < 3:
        raise ValueError('mask-min-points must be at least 3 for a polygon')
    if max_points < args.mask_min_points:
        raise ValueError('mask-max-points is smaller than mask-min-points')

    old_state = np.random.get_state()
    try:
        np.random.seed(mask_seed)
        num_points = int(
            np.random.randint(args.mask_min_points, max_points + 1))
        mask = create_random_mask(shape=(args.w, args.w), num_points=num_points, fixshape=False, inverse=False,)
    finally:
        np.random.set_state(old_state)

    return np.asarray(mask, dtype=np.float32), num_points


def spin_numpy(film) -> np.ndarray:
    return film.Spin.detach().cpu().numpy().astype(np.float32, copy=True)


def reservoir_add(reservoir: List[Tuple[int, np.ndarray, np.ndarray, np.ndarray]], item: Tuple[int, np.ndarray, np.ndarray, np.ndarray], 
                  seen: int, capacity: int, rng: np.random.Generator,) -> None:
    """
    Uniform reservoir sampling over all valid interior trajectory states.
    """
    if len(reservoir) < capacity:
        reservoir.append(item)
        return

    replacement = int(rng.integers(0, seen))
    if replacement < capacity:
        reservoir[replacement] = item


def simulate_and_sample_spin_triplets(film, Hext: np.ndarray, args: argparse.Namespace, 
                                      sample_rng: np.random.Generator,) -> Tuple[np.ndarray, np.ndarray, List[float], bool, int]:
    """
    Run one FFT/LLG trajectory and uniformly sample consecutive spin triplets.

    The three states in a saved item are exact consecutive RK4 states:
    [m_(t-1), m_t, m_(t+1)]. The first and last trajectory states are not valid
    centers because a centered temporal difference needs both neighbors.
    """
    reservoir: List[Tuple[int, np.ndarray, np.ndarray, np.ndarray]] = []
    error_list: List[float] = []
    valid_centers_seen = 0

    previous = spin_numpy(film)  # m_0
    center = None
    error = np.inf
    step = 0

    while error > args.error_min and step < args.max_iter:
        error = float(film.SpinLLG_RK4(Hext=Hext, dtime=args.dtime, damping=args.damping,))
        step += 1
        error_list.append(error)
        new_state = spin_numpy(film)  # m_step

        if center is None:
            center = new_state  # m_1; no centered triplet exists yet
            continue

        # At this point the window is (m_(step-2), m_(step-1), m_step).
        center_step = step - 1
        valid_centers_seen += 1
        reservoir_add(reservoir=reservoir, item=(center_step, previous, center, new_state), seen=valid_centers_seen, 
                      capacity=args.sav_samples, rng=sample_rng,)
        previous, center = center, new_state

    converged = bool(error <= args.error_min)

    if not converged and not args.allow_unconverged:
        raise RuntimeError(
            f'Simulation reached max_iter={args.max_iter} with error={error:.3e}, '
            f'above error_min={args.error_min:.3e}.')

    if valid_centers_seen < args.sav_samples:
        raise RuntimeError(
            f'Trajectory produced only {valid_centers_seen} valid interior center '
            f'states, but {args.sav_samples} unique temporal samples were requested.')

    reservoir.sort(key=lambda entry: entry[0])
    center_steps = np.asarray([entry[0] for entry in reservoir], dtype=np.int32,)
    spin_triplets = np.stack(
        [np.stack(entry[1:], axis=0) for entry in reservoir],
        axis=0,
    ).astype(np.float32, copy=False)

    return spin_triplets, center_steps, error_list, converged, step


def exact_demag_for_triplets(film, spin_triplets: np.ndarray) -> np.ndarray:
    """
    Compute Hd from each exact saved spin state.

    SpinLLG_RK4 leaves film.Hd associated with its final RK4 substage, not with the
    final normalized state returned by the step. Recomputing Demag here guarantees
    that every saved target is Hd(m) for the exact saved m.
    """
    hds = np.empty_like(spin_triplets, dtype=np.float32)

    saved_spin = film.Spin.detach().clone()
    saved_hd = film.Hd.detach().clone()
    device = film.device
    dtype = saved_spin.dtype

    try:
        with torch.no_grad():
            for sample_index in range(spin_triplets.shape[0]):
                for temporal_index in range(3):
                    film.Spin = torch.as_tensor(
                        spin_triplets[sample_index, temporal_index],
                        dtype=dtype,
                        device=device,
                    ).clone()
                    film.Demag()
                    hds[sample_index, temporal_index] = (
                        film.Hd.detach().cpu().numpy().astype(np.float32, copy=False)
                    )
    finally:
        film.Spin = saved_spin
        film.Hd = saved_hd

    return hds


def flatten_layers(array: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    """(N, w, w, layers, 3) -> (N, w, w, layers*3)."""
    return array.reshape(
        array.shape[0],
        args.w,
        args.w,
        args.layers * 3,
    )


def claim_output_directory(save_path: str, overwrite: bool) -> None:
    """
    Atomically claim one seed directory so parallel jobs cannot silently collide.
    """
    parent = os.path.dirname(save_path)
    os.makedirs(parent, exist_ok=True)

    if overwrite and os.path.exists(save_path):
        shutil.rmtree(save_path)

    try:
        os.mkdir(save_path)
    except FileExistsError as exc:
        raise FileExistsError(
            f'Output already exists: {save_path}. Use a different --seed-start, '
            'remove the incomplete directory, or pass --overwrite intentionally.'
        ) from exc


def save_temporal_data(
    spin_triplets: np.ndarray,
    hd_triplets: np.ndarray,
    center_steps: np.ndarray,
    error_list: Sequence[float],
    Hext: np.ndarray,
    metadata: Dict,
    save_path: str,
    args: argparse.Namespace,
) -> None:
    # Preserve the center-state filenames used by the original dataset loader.
    np.save(
        os.path.join(save_path, 'Spins.npy'),
        flatten_layers(spin_triplets[:, 1], args),
    )
    np.save(
        os.path.join(save_path, 'Hds.npy'),
        flatten_layers(hd_triplets[:, 1], args),
    )

    # Neighboring exact states for temporal finite differences.
    np.save(
        os.path.join(save_path, 'Spins_prev.npy'),
        flatten_layers(spin_triplets[:, 0], args),
    )
    np.save(
        os.path.join(save_path, 'Spins_next.npy'),
        flatten_layers(spin_triplets[:, 2], args),
    )
    np.save(
        os.path.join(save_path, 'Hds_prev.npy'),
        flatten_layers(hd_triplets[:, 0], args),
    )
    np.save(
        os.path.join(save_path, 'Hds_next.npy'),
        flatten_layers(hd_triplets[:, 2], args),
    )

    np.save(os.path.join(save_path, 'StepIndices.npy'), center_steps)
    np.save(
        os.path.join(save_path, 'Errors.npy'),
        np.asarray(error_list, dtype=np.float32),
    )
    np.save(
        os.path.join(save_path, 'Hext.npy'),
        np.asarray(Hext, dtype=np.float64),
    )

    # The mask is constant for the complete trajectory. Masked cells stay exactly
    # zero under the repository's LLG update, while magnetic cells remain unit norm.
    active_mask_3d = (
        np.linalg.norm(spin_triplets[0, 1], axis=-1) > 1.0e-12
    ).astype(np.uint8)
    np.save(os.path.join(save_path, 'Mask3D.npy'), active_mask_3d)
    np.save(
        os.path.join(save_path, 'Mask2D.npy'),
        np.any(active_mask_3d, axis=2).astype(np.uint8),
    )

    with open(
        os.path.join(save_path, 'metadata.json'),
        'w',
        encoding='utf-8',
    ) as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)

    final_error = float(error_list[-1]) if error_list else float('nan')
    error_plot(
        error_list,
        os.path.join(
            save_path,
            f'iterns{len(error_list):.1e}_errors_{final_error:.1e}',
        ),
        f'[{Hext[0]:.2f}, {Hext[1]:.2f}, {Hext[2]:.2f}]',
    )


def generate_one_simulation(
    args: argparse.Namespace,
    film,
    simulation_id: int,
    masked: bool,
) -> str:
    # Including masked/unmasked in the SeedSequence makes the two categories
    # independent even when they use the same displayed seed ID.
    root_sequence = np.random.SeedSequence(
        [args.dataset_seed, args.w, simulation_id, int(masked)]
    )
    field_ss, mask_ss, sample_ss, spin_ss = root_sequence.spawn(4)

    field_rng = np.random.default_rng(field_ss)
    mask_rng = np.random.default_rng(mask_ss)
    sample_rng = np.random.default_rng(sample_ss)
    spin_rng = np.random.default_rng(spin_ss)

    if args.field_mode == 'random':
        Hext, magnitude, angle = random_in_plane_field(field_rng)
    else:
        Hext = np.zeros(3, dtype=np.float64)
        magnitude = 0.0
        angle = 0.0

    spin_seed = int(
        spin_rng.integers(
            0,
            np.iinfo(np.uint32).max,
            dtype=np.uint32,
        )
    )
    spin = initial_spin_prepare(args.w, args.layers, spin_seed)

    mask_seed = None
    mask_points = None
    if masked:
        mask_seed = int(
            mask_rng.integers(
                0,
                np.iinfo(np.uint32).max,
                dtype=np.uint32,
            )
        )
        mask, mask_points = random_mask_with_isolated_seed(args, mask_seed)
        spin = spin * mask

    film.SpinInit(spin)

    spin_triplets, center_steps, errors, converged, total_steps = (
        simulate_and_sample_spin_triplets(
            film=film,
            Hext=Hext,
            args=args,
            sample_rng=sample_rng,
        )
    )

    # Exact FFT targets for the exact previous, center, and next spin states.
    hd_triplets = exact_demag_for_triplets(film, spin_triplets)

    category = 'masked' if masked else 'unmasked'
    save_path = os.path.join(
        args.output_root,
        f'w{args.w}',
        category,
        f'seed{simulation_id:06d}',
    )
    claim_output_directory(save_path, overwrite=args.overwrite)

    metadata = {
        'simulation_id': simulation_id,
        'simulation_type': category,
        'width': args.w,
        'layers': args.layers,
        'cell_nm': film.cell.tolist(),
        'boundary_type': film.types,
        'Ms_emu_per_cc': args.Ms,
        'Ax_erg_per_cm': args.Ax,
        'Ku_erg_per_cc': args.Ku,
        'Kvec': list(args.Kvec),
        'damping': args.damping,
        'dtime_s': args.dtime,
        'error_min': args.error_min,
        'max_iter': args.max_iter,
        'total_steps': total_steps,
        'converged': converged,
        'final_error': float(errors[-1]),
        'samples_saved': int(spin_triplets.shape[0]),
        'sampling_method': 'uniform reservoir sampling over interior trajectory states',
        'temporal_offsets': [-1, 0, 1],
        'centered_difference_denominator_s': 2.0 * args.dtime,
        'Hext_Oe': Hext.tolist(),
        'Hext_magnitude_Oe': magnitude,
        'Hext_angle_rad': angle,
        'field_mode': args.field_mode,
        'masked': masked,
        'mask_seed': mask_seed,
        'mask_num_points': mask_points,
        'spin_seed': spin_seed,
        'dataset_seed': args.dataset_seed,
        'exact_same_state_Hd': True,
    }

    save_temporal_data(
        spin_triplets=spin_triplets,
        hd_triplets=hd_triplets,
        center_steps=center_steps,
        error_list=errors,
        Hext=Hext,
        metadata=metadata,
        save_path=save_path,
        args=args,
    )

    return save_path


def generate_data(args: argparse.Namespace, film) -> None:
    jobs = requested_simulations(args)

    for simulation_id, masked in tqdm(
        jobs,
        desc=f'Generating w={args.w}',
    ):
        save_path = generate_one_simulation(
            args=args,
            film=film,
            simulation_id=simulation_id,
            masked=masked,
        )
        print(f'Saved: {save_path}')

    print(
        f'Completed {args.masked} masked and {args.unmasked} unmasked '
        f'simulations for w={args.w}.'
    )


if __name__ == '__main__':
    args = build_parser().parse_args()
    validate_generation_request(args)
    model = prepare_model(args)
    generate_data(args, model)

