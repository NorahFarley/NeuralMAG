import os
import glob
import torch
import random
import numpy as np
from tqdm import tqdm
from utils import print_memory, winding_density
from collections import OrderedDict

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
        print_memory("Before loading 128 evaluation dataset")
        X_test, Y_test = getdata(data_paths, ntest, n128, ntrain, cn, mode)
        print_memory("After loading 128 evaluation NumPy arrays")
        #prepare test set
        X_test_tensor = torch.from_numpy(X_test).float()
        y_test_tensor = torch.from_numpy(Y_test).float()
        test_dataset  = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)
        print_memory("After preparing 128 evaluation TensorDataset")
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


def get_case_paths_temporal_physics(path):
    cases = []

    for root, _, files in os.walk(path):
        required = {"Spins.npy",
                    "Hds.npy",
                    "Spins_prev.npy",
                    "Hds_prev.npy"}

        if required.issubset(files):
            cases.append((os.path.join(root, "Spins.npy"),
                          os.path.join(root, "Hds.npy"),
                          os.path.join(root, "Spins_prev.npy"),
                          os.path.join(root, "Hds_prev.npy")))

    cases.sort()
    return cases


class TemporalPhysicsDataset(torch.utils.data.Dataset):
    """
    Lazy temporal dataset.

    Training:
        (m_t, Hd_t, m_(t-1), Hd_(t-1))

    Testing:
        (m_t, Hd_t)
    """

    def __init__(self, cases, include_previous, max_cached_cases=8):
        self.cases = list(cases)
        self.include_previous = include_previous
        self.max_cached_cases = max_cached_cases
        self._cache = OrderedDict()

        self.case_lengths = []

        for case in self.cases:
            shapes = [np.load(path, mmap_mode="r").shape for path in case]

            if not all(shape == shapes[0] for shape in shapes):
                raise ValueError("Temporal arrays have different shapes in "
                                 f"{os.path.dirname(case[0])}")

            self.case_lengths.append(shapes[0][0])

        self.cumulative_lengths = np.cumsum(self.case_lengths)

    def __len__(self):
        if len(self.cumulative_lengths) == 0:
            return 0

        return int(self.cumulative_lengths[-1])

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_cache"] = OrderedDict()
        return state

    def _open_case(self, case_index):
        if case_index in self._cache:
            arrays = self._cache.pop(case_index)
            self._cache[case_index] = arrays
            return arrays

        arrays = tuple(np.load(path, mmap_mode="r") for path in self.cases[case_index])

        self._cache[case_index] = arrays

        while len(self._cache) > self.max_cached_cases:
            self._cache.popitem(last=False)

        return arrays

    @staticmethod
    def _sample_to_tensor(array, sample_index):
        sample = (np.asarray(array[sample_index]).transpose(2, 0, 1).copy())

        return torch.from_numpy(sample).float()

    def __getitem__(self, index):
        case_index = int(
            np.searchsorted(self.cumulative_lengths, index, side="right"))

        previous_total = (0 
                          if case_index == 0
                          else int(self.cumulative_lengths[case_index - 1]))

        sample_index = index - previous_total

        (spins, hds, spins_prev, hds_prev) = self._open_case(case_index)

        x = self._sample_to_tensor(spins, sample_index)
        y = self._sample_to_tensor(hds, sample_index)

        if not self.include_previous:
            return x, y

        x_prev = self._sample_to_tensor(spins_prev, sample_index)
        y_prev = self._sample_to_tensor(hds_prev, sample_index)

        return x, y, x_prev, y_prev


def dataset_prepare_temporal_physics(data_paths, ntest, ntrain, cn=1000, include_previous_train=True):
    if cn < 1000:
        raise NotImplementedError("Use --cornum 1000 with the temporal loader. "
                                  "A winding filter must preserve temporal alignment.")

    cases = []

    for path in data_paths:
        cases.extend(get_case_paths_temporal_physics(path))

    if not cases:
        raise FileNotFoundError("No temporal cases found. Expected "
                                "Spins.npy, Hds.npy, Spins_prev.npy, "
                                "and Hds_prev.npy.")

    rng = random.Random(123)
    rng.shuffle(cases)

    if ntest > len(cases):
        raise ValueError(f"ntest={ntest}, but only "
                         f"{len(cases)} cases were found.")

    train_stop = min(ntrain, len(cases))

    train_cases = cases[ntest:train_stop]
    test_cases = cases[:ntest]

    if not train_cases:
        raise ValueError("No training cases remain after splitting.")

    print("train seed number:", len(train_cases))
    print("test seed number:", len(test_cases))

    return (TemporalPhysicsDataset(train_cases, include_previous=include_previous_train),
            TemporalPhysicsDataset(test_cases, include_previous=False))
