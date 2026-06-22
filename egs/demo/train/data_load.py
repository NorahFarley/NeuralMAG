import os
import glob
import torch
import random
import numpy as np
from tqdm import tqdm
from utils import winding_density

def get_case_paths(paths, spin_file='Spins.npy', hd_file='Hds.npy'):
    spin_paths = []
    Hd_paths = []
    for path in glob.glob(os.path.join(paths, '*')):
        try:
            if os.path.exists((path+f'/{spin_file}')) and os.path.exists((path+f'/{hd_file}')):
                spin_paths.append((path+f'/{spin_file}'))
                Hd_paths.append((path+f'/{hd_file}'))
        except ValueError:
            print(f"Invalid path: {path}")
    return spin_paths, Hd_paths

def path_split(paths, ntest, n128, ntrain, spin_file, hd_file, mode=None):
    random.seed(123)
    if mode=='eval128':
        spin_paths = []
        Hd_paths = []
        for path in paths:
            spin_path, Hd_path = get_case_paths(path, spin_file, hd_file)
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
            spin_path, Hd_path = get_case_paths(path, spin_file, hd_file)
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
    
    for x_path, y_path in zip(x_paths, y_paths):
        # Load the consecutive trajectories
        # Expected shape: (Frames, 6, 32, 32)
        x_array = np.load(x_path)
        y_array = np.load(y_path)
        
        # Iterate starting from index 1 so we always have a past state (i-1)
        for i in range(1, x_array.shape):
            # Stack m_{t-1} and m_t to create a 12-channel input
            stacked_x = np.concatenate((x_array[i-1], x_array[i]), axis=0) 
            
            selected_x.append(stacked_x)
            selected_y.append(y_array[i]) # Target is H_demag at current time t
            
    return np.array(selected_x), np.array(selected_y)


def getdata(paths, ntest, n128, ntrain, cn, spin_file, hd_file, mode=None):
    if mode=='eval128':
        x_test_paths, y_test_paths = path_split(paths, ntest, n128, ntrain, spin_file, hd_file, mode)
        x_test, y_test = combinfilter(x_test_paths, y_test_paths, cn)
        return x_test, y_test
    else:
        x_train_paths, y_train_paths, x_test_paths, y_test_paths = path_split(paths, ntest, n128, ntrain, spin_file, hd_file)
        x_train, y_train = combinfilter(x_train_paths, y_train_paths, cn)
        x_test, y_test = combinfilter(x_test_paths, y_test_paths, cn)
        return x_train, y_train, x_test, y_test


def dataset_prepare(data_paths, ntest, n128, ntrain, cn, spin_file='Spins.npy', hd_file='Hds.npy', mode=None):
    print('loading data from: ', data_paths)
    if mode=='eval128':
        X_test, Y_test = getdata(data_paths, ntest, n128, ntrain, cn, spin_file, hd_file, mode)
        #prepare test set
        X_test_tensor = torch.from_numpy(X_test).float()
        y_test_tensor = torch.from_numpy(Y_test).float()
        test_dataset  = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)
        return test_dataset
    else:
        X_train, Y_train, X_test, Y_test = getdata(data_paths, ntest, n128, ntrain, cn, spin_file, hd_file)
        #prepare training set
        X_train_tensor = torch.from_numpy(X_train).float()
        y_train_tensor = torch.from_numpy(Y_train).float()
        train_dataset  = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        #prepare test set
        X_test_tensor = torch.from_numpy(X_test).float()
        y_test_tensor = torch.from_numpy(Y_test).float()
        test_dataset  = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)
        return train_dataset, test_dataset

