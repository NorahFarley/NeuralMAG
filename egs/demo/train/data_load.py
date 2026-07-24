import os
import glob
import torch
import random
import numpy as np
from tqdm import tqdm
from utils import *

def get_case_paths(paths):
    spin_paths = []
    Hd_paths = []
    for path in glob.glob(os.path.join(paths, '*')):
        try:
            if os.path.exists((path+f'/Spins.npy')) and os.path.exists((path+f'/Hds.npy')):
                spin_paths.append((path+f'/Spins.npy'))
                Hd_paths.append((path+f'/Hds.npy'))
        except ValueError:
            print(f"Invalid path: {path}")
    return spin_paths, Hd_paths

def path_split(paths, ntest, n128, ntrain, mode=None):
    random.seed(123)
    if mode=='eval128':
        spin_paths = []
        Hd_paths = []
        for path in paths:
            spin_path, Hd_path = get_case_paths(path)
            spin_paths.extend(spin_path)
            Hd_paths.extend(Hd_path)
        paired_paths = list(zip(spin_paths, Hd_paths))
        random.shuffle(paired_paths)
        test_data_paths, test_target_paths = zip(*paired_paths[:n128])
        print("test seed number:", len(test_data_paths), len(test_target_paths))
        return test_data_paths, test_target_paths
    else:
        spin_paths = []
        Hd_paths = []
        for path in paths:
            spin_path, Hd_path = get_case_paths(path)
            spin_paths.extend(spin_path)
            Hd_paths.extend(Hd_path)
        paired_paths = list(zip(spin_paths, Hd_paths))
        random.shuffle(paired_paths)
        train_data_paths, train_target_paths = zip(*paired_paths[ntest:ntrain])
        test_data_paths, test_target_paths = zip(*paired_paths[:ntest])
        print("train seed number:", len(train_data_paths))
        print("test seed number:", len(test_data_paths))
        return train_data_paths, train_target_paths, test_data_paths, test_target_paths


def combinfilter(x_paths, y_paths, cn):
    np.random.seed(123)
    selected_x = []
    selected_y = []
    a=0
    for x_path, y_path in zip(x_paths, y_paths):
        x_array = np.load(x_path, mmap_mode='r').transpose((0, 3, 1, 2))
        y_array = np.load(y_path, mmap_mode='r').transpose((0, 3, 1, 2))
        
        # randomly select half of the data
        indices = np.random.choice(x_array.shape[0], 500, replace=False)
        x_array = x_array[indices]
        y_array = y_array[indices]
        
        if cn <1000:
            _, winding_abs = winding_density(x_array)
            indices = np.where((0 <= winding_abs) & (winding_abs <= cn))
            selected_x.append(x_array[indices])
            selected_y.append(y_array[indices])
        else:
            selected_x.append(x_array)
            selected_y.append(y_array)
        a += x_array.shape[0]
    selected_x = np.concatenate(selected_x, axis=0)
    selected_y = np.concatenate(selected_y, axis=0)

    print('0<= core number <={}, selected_percent: {:.2f}'.format(cn, selected_x.shape[0]/a))
    print('selected x y shape: ', selected_x.shape, selected_y.shape, '\n')
    
    return selected_x, selected_y


def getdata(paths, ntest, n128, ntrain, cn, mode=None):
    if mode=='eval128':
        x_test_paths, y_test_paths = path_split(paths, ntest, n128, ntrain, mode)
        x_test, y_test = combinfilter(x_test_paths, y_test_paths, cn)
        return x_test, y_test
    else:
        x_train_paths, y_train_paths, x_test_paths, y_test_paths = path_split(paths, ntest, n128, ntrain)
        x_train, y_train = combinfilter(x_train_paths, y_train_paths, cn)
        x_test, y_test = combinfilter(x_test_paths, y_test_paths, cn)
        return x_train, y_train, x_test, y_test


def dataset_prepare(data_paths, ntest, n128, ntrain, cn, mode=None):
    print('loading data from: ', data_paths)
    if mode=='eval128':
        X_test, Y_test = getdata(data_paths, ntest, n128, ntrain, cn, mode)
        #prepare test set
        X_test_tensor = torch.from_numpy(X_test).float()
        y_test_tensor = torch.from_numpy(Y_test).float()
        test_dataset  = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)
        return test_dataset
    else:
        print_memory("Before getdata")
        print("Loading training dataset...", flush=True)
        X_train, Y_train, X_test, Y_test = getdata(data_paths, ntest, n128, ntrain, cn)
        print(f"Dataset loading complete for: {data_paths}", flush=True)
        print(X_train.shape, Y_train.shape, flush=True)
        print_memory("After getdata")
        #prepare training set
        X_train_tensor = torch.from_numpy(X_train).float()
        y_train_tensor = torch.from_numpy(Y_train).float()
        train_dataset  = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        #prepare test set
        X_test_tensor = torch.from_numpy(X_test).float()
        y_test_tensor = torch.from_numpy(Y_test).float()
        test_dataset  = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)
        return train_dataset, test_dataset

    """
Add these functions to data_load.py.

They leave the existing loader untouched and add a temporal loader for datasets
containing:
    Spins.npy
    Hds.npy
    Spins_prev.npy

Training TensorDataset items are returned as:
    (current_spin, current_hd, previous_spin)

Validation/test TensorDataset items remain:
    (current_spin, current_hd)

so the existing eval() loop does not need to be changed.
"""

import os
import random
import numpy as np
import torch


def get_case_paths_temporal(paths):
    """
    Recursively find temporal training cases.

    Recursive walking is intentional because the new generator stores cases under:
        wXX/masked/seedXXXXXX/
        wXX/unmasked/seedXXXXXX/
    """
    cases = []

    for root, _, files in os.walk(paths):
        required = {'Spins.npy', 'Hds.npy', 'Spins_prev.npy'}
        if required.issubset(files):
            cases.append(
                (
                    os.path.join(root, 'Spins.npy'),
                    os.path.join(root, 'Hds.npy'),
                    os.path.join(root, 'Spins_prev.npy'),
                )
            )

    cases.sort()
    return cases


def path_split_temporal(paths, ntest, n128, ntrain, mode=None):
    """
    Split complete (current, Hd, previous) cases together so temporal alignment
    cannot be broken by shuffling.
    """
    cases = []

    for path in paths:
        cases.extend(get_case_paths_temporal(path))

    if not cases:
        raise FileNotFoundError(
            "No temporal cases were found. Each seed directory must contain "
            "Spins.npy, Hds.npy, and Spins_prev.npy."
        )

    random.seed(123)
    random.shuffle(cases)

    if mode == 'eval128':
        selected = cases[:n128]
        print("test seed number:", len(selected))
        return selected

    train_cases = cases[ntest:ntrain]
    test_cases = cases[:ntest]

    print("train seed number:", len(train_cases))
    print("test seed number:", len(test_cases))

    return train_cases, test_cases


def _load_temporal_cases(cases, cn, include_previous):
    """
    Load aligned center/target/previous samples.

    The exact same per-trajectory random indices and any winding-number filter
    are applied to all temporal arrays.
    """
    rng = np.random.RandomState(123)

    selected_x = []
    selected_y = []
    selected_prev = []
    total_selected_before_core_filter = 0

    for x_path, y_path, prev_path in cases:
        x_array = np.load(x_path, mmap_mode='r').transpose((0, 3, 1, 2))
        y_array = np.load(y_path, mmap_mode='r').transpose((0, 3, 1, 2))
        prev_array = np.load(prev_path, mmap_mode='r').transpose((0, 3, 1, 2))

        if not (
            x_array.shape[0] == y_array.shape[0] == prev_array.shape[0]
        ):
            raise ValueError(
                "Temporal arrays have different sample counts in:\n"
                f"  {os.path.dirname(x_path)}"
            )

        if x_array.shape != prev_array.shape:
            raise ValueError(
                "Spins.npy and Spins_prev.npy have different shapes in:\n"
                f"  {os.path.dirname(x_path)}"
            )

        if x_array.shape[0] < 500:
            raise ValueError(
                f"{x_path} contains only {x_array.shape[0]} samples; "
                "the current NeuralMAG training pipeline expects 500."
            )

        # Same behavior as the existing loader: choose 500 samples per case.
        indices = rng.choice(x_array.shape[0], 500, replace=False)

        x_array = np.asarray(x_array[indices])
        y_array = np.asarray(y_array[indices])
        prev_array = np.asarray(prev_array[indices])

        total_selected_before_core_filter += x_array.shape[0]

        if cn < 1000:
            _, winding_abs = winding_density(x_array)
            core_indices = np.where(
                (0 <= winding_abs) & (winding_abs <= cn)
            )

            x_array = x_array[core_indices]
            y_array = y_array[core_indices]
            prev_array = prev_array[core_indices]

        selected_x.append(x_array)
        selected_y.append(y_array)

        if include_previous:
            selected_prev.append(prev_array)

    x = np.concatenate(selected_x, axis=0)
    y = np.concatenate(selected_y, axis=0)

    print(
        '0<= core number <={}, selected_percent: {:.2f}'.format(
            cn,
            x.shape[0] / total_selected_before_core_filter,
        )
    )
    print('selected x y shape: ', x.shape, y.shape, '\n')

    if not include_previous:
        return x, y

    prev = np.concatenate(selected_prev, axis=0)
    print('selected previous-spin shape: ', prev.shape, '\n')

    return x, y, prev


def getdata_temporal(paths, ntest, n128, ntrain, cn, mode=None):
    if mode == 'eval128':
        test_cases = path_split_temporal(
            paths, ntest, n128, ntrain, mode='eval128'
        )
        x_test, y_test = _load_temporal_cases(
            test_cases, cn, include_previous=False
        )
        return x_test, y_test

    train_cases, test_cases = path_split_temporal(
        paths, ntest, n128, ntrain
    )

    x_train, y_train, x_prev_train = _load_temporal_cases(
        train_cases, cn, include_previous=True
    )
    x_test, y_test = _load_temporal_cases(
        test_cases, cn, include_previous=False
    )

    return x_train, y_train, x_prev_train, x_test, y_test


def dataset_prepare_temporal(
    data_paths,
    ntest,
    n128,
    ntrain,
    cn,
    mode=None,
):
    """
    Temporal counterpart of dataset_prepare().

    Training samples:
        (m_t, Hd_t, m_{t-1})

    Test samples:
        (m_t, Hd_t)

    This keeps the existing evaluation code compatible.
    """
    print('loading temporal data from: ', data_paths)

    if mode == 'eval128':
        x_test, y_test = getdata_temporal(
            data_paths, ntest, n128, ntrain, cn, mode='eval128'
        )

        return torch.utils.data.TensorDataset(
            torch.from_numpy(x_test).float(),
            torch.from_numpy(y_test).float(),
        )

    (x_train, y_train, x_prev_train, x_test, y_test,) = getdata_temporal(data_paths, ntest, n128, ntrain, cn)

    train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(y_train).float(), torch.from_numpy(x_prev_train).float(),)

    test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test).float(),)

    return train_dataset, test_dataset


"""
Temporal loader additions for NeuralMAG rate-of-change weighted losses.

Training TensorDataset items:
    (m_t, Hd_t, m_(t-1), Hd_(t-1))

Test/eval TensorDataset items:
    (m_t, Hd_t)

This keeps the existing evaluation loop unchanged while making the previous
state and previous FFT demag field available during training.
"""

import os
import random
import numpy as np
import torch


def get_case_paths_temporal_physics(path):
    """
    Recursively find generated temporal seed directories containing:
        Spins.npy
        Hds.npy
        Spins_prev.npy
        Hds_prev.npy
    """
    cases = []

    for root, _, files in os.walk(path):
        required = {
            'Spins.npy',
            'Hds.npy',
            'Spins_prev.npy',
            'Hds_prev.npy',
        }

        if required.issubset(files):
            cases.append(
                (
                    os.path.join(root, 'Spins.npy'),
                    os.path.join(root, 'Hds.npy'),
                    os.path.join(root, 'Spins_prev.npy'),
                    os.path.join(root, 'Hds_prev.npy'),
                )
            )

    cases.sort()
    return cases


def path_split_temporal_physics(
    paths,
    ntest,
    n128,
    ntrain,
    mode=None,
):
    """
    Shuffle complete temporal cases as units so current/previous alignment
    can never be broken.
    """
    cases = []

    for path in paths:
        cases.extend(get_case_paths_temporal_physics(path))

    if not cases:
        raise FileNotFoundError(
            "No temporal seed directories were found. "
            "Expected Spins.npy, Hds.npy, Spins_prev.npy, and Hds_prev.npy."
        )

    rng = random.Random(123)
    rng.shuffle(cases)

    if mode == 'eval128':
        selected = cases[:n128]
        print("test seed number:", len(selected))
        return selected

    train_cases = cases[ntest:ntrain]
    test_cases = cases[:ntest]

    print("train seed number:", len(train_cases))
    print("test seed number:", len(test_cases))

    return train_cases, test_cases


def _load_temporal_physics_cases(
    cases,
    cn,
    include_previous,
):
    """
    Load temporally aligned arrays.

    One index array is generated per seed and applied to ALL four arrays.
    Any existing winding-number filter is also applied to all four arrays
    together.
    """
    rng = np.random.RandomState(123)

    selected_x = []
    selected_y = []
    selected_x_prev = []
    selected_y_prev = []

    total_before_core_filter = 0

    for (
        x_path,
        y_path,
        x_prev_path,
        y_prev_path,
    ) in cases:

        x_array = np.load(
            x_path, mmap_mode='r'
        ).transpose((0, 3, 1, 2))

        y_array = np.load(
            y_path, mmap_mode='r'
        ).transpose((0, 3, 1, 2))

        x_prev_array = np.load(
            x_prev_path, mmap_mode='r'
        ).transpose((0, 3, 1, 2))

        y_prev_array = np.load(
            y_prev_path, mmap_mode='r'
        ).transpose((0, 3, 1, 2))

        sample_counts = {
            x_array.shape[0],
            y_array.shape[0],
            x_prev_array.shape[0],
            y_prev_array.shape[0],
        }

        if len(sample_counts) != 1:
            raise ValueError(
                "Temporal arrays have different sample counts in:\n"
                f"  {os.path.dirname(x_path)}"
            )

        if x_array.shape != x_prev_array.shape:
            raise ValueError(
                "Spins.npy and Spins_prev.npy shapes differ in:\n"
                f"  {os.path.dirname(x_path)}"
            )

        if y_array.shape != y_prev_array.shape:
            raise ValueError(
                "Hds.npy and Hds_prev.npy shapes differ in:\n"
                f"  {os.path.dirname(x_path)}"
            )

        n_available = x_array.shape[0]

        if n_available < 500:
            raise ValueError(
                f"{os.path.dirname(x_path)} contains only "
                f"{n_available} samples; expected at least 500."
            )

        # The generator currently already saves exactly 500 samples per case,
        # but keeping this selection preserves the behavior of the old loader.
        indices = rng.choice(
            n_available,
            500,
            replace=False,
        )

        x_array = np.asarray(x_array[indices])
        y_array = np.asarray(y_array[indices])
        x_prev_array = np.asarray(x_prev_array[indices])
        y_prev_array = np.asarray(y_prev_array[indices])

        total_before_core_filter += x_array.shape[0]

        if cn < 1000:
            # This calls the existing winding_density() in data_load.py/utils.py.
            # Use ONE filter derived from the CURRENT state and apply it to
            # current and previous arrays together.
            _, winding_abs = winding_density(x_array)

            keep = np.where(
                (0 <= winding_abs) & (winding_abs <= cn)
            )[0]

            x_array = x_array[keep]
            y_array = y_array[keep]
            x_prev_array = x_prev_array[keep]
            y_prev_array = y_prev_array[keep]

        selected_x.append(x_array)
        selected_y.append(y_array)

        if include_previous:
            selected_x_prev.append(x_prev_array)
            selected_y_prev.append(y_prev_array)

    x = np.concatenate(selected_x, axis=0)
    y = np.concatenate(selected_y, axis=0)

    print(
        '0<= core number <={}, selected_percent: {:.2f}'.format(
            cn,
            x.shape[0] / total_before_core_filter,
        )
    )
    print('selected current x y shape:', x.shape, y.shape)

    if not include_previous:
        return x, y

    x_prev = np.concatenate(selected_x_prev, axis=0)
    y_prev = np.concatenate(selected_y_prev, axis=0)

    print(
        'selected previous x y shape:',
        x_prev.shape,
        y_prev.shape,
        '\n',
    )

    return x, y, x_prev, y_prev


def getdata_temporal_physics(
    paths,
    ntest,
    n128,
    ntrain,
    cn,
    mode=None,
):
    if mode == 'eval128':
        test_cases = path_split_temporal_physics(
            paths,
            ntest,
            n128,
            ntrain,
            mode='eval128',
        )

        x_test, y_test = _load_temporal_physics_cases(
            test_cases,
            cn,
            include_previous=False,
        )

        return x_test, y_test

    train_cases, test_cases = path_split_temporal_physics(
        paths,
        ntest,
        n128,
        ntrain,
    )

    (
        x_train,
        y_train,
        x_prev_train,
        y_prev_train,
    ) = _load_temporal_physics_cases(
        train_cases,
        cn,
        include_previous=True,
    )

    x_test, y_test = _load_temporal_physics_cases(
        test_cases,
        cn,
        include_previous=False,
    )

    return (
        x_train,
        y_train,
        x_prev_train,
        y_prev_train,
        x_test,
        y_test,
    )


def dataset_prepare_temporal_physics(
    data_paths,
    ntest,
    n128,
    ntrain,
    cn,
    mode=None,
):
    """
    Temporal counterpart of dataset_prepare().

    TRAINING dataset items:
        current spin,
        current FFT Hdemag,
        previous spin,
        previous FFT Hdemag

    TEST dataset items:
        current spin,
        current FFT Hdemag
    """
    print('loading temporal physics data from:', data_paths)

    if mode == 'eval128':
        x_test, y_test = getdata_temporal_physics(
            data_paths,
            ntest,
            n128,
            ntrain,
            cn,
            mode='eval128',
        )

        return torch.utils.data.TensorDataset(
            torch.from_numpy(x_test).float(),
            torch.from_numpy(y_test).float(),
        )

    (
        x_train,
        y_train,
        x_prev_train,
        y_prev_train,
        x_test,
        y_test,
    ) = getdata_temporal_physics(
        data_paths,
        ntest,
        n128,
        ntrain,
        cn,
    )

    train_dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(x_train).float(),
        torch.from_numpy(y_train).float(),
        torch.from_numpy(x_prev_train).float(),
        torch.from_numpy(y_prev_train).float(),
    )

    test_dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(x_test).float(),
        torch.from_numpy(y_test).float(),
    )

    return train_dataset, test_dataset


