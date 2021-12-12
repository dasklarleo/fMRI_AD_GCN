# dataloader: load the data from the matlab file
from numpy.core.fromnumeric import shape
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import os
import scipy.io as scio
import math


def load_data(data_folder_path, batch_size):
    data = []
    labels = []
    files = os.listdir(data_folder_path)
    for file in files:
        if 'DFC' in file:
            continue
        elif 'NC' in file:
            labels.append(1)
        elif 'AD' in file:
            labels.append(2)
        elif 'EMCI' in file:
            labels.append(3)
        elif 'LMCI' in file:
            labels.append(4)
        elif 'SMC' in file:
            labels.append(5)
        elif 'MCI' in file:
            labels.append(6)
        ROI_BOLDs = scio.loadmat(data_folder_path+'/'+file)['ROI_ts']
        data.append(ROI_BOLDs)

    data = np.array(data)
    labels = np.array(labels)
    # randomly shuffle the data
    state = np.random.get_state()
    np.random.shuffle(data)
    np.random.set_state(state)
    np.random.shuffle(labels)
    # cut the data into training data and valiadation data
    data_len = data.shape[0]
    train_len = math.ceil(data_len*0.8)
    train_data = data[0:train_len]
    valiad_data = data[train_len:data_len]
    train_labels = labels[0:train_len]
    valiad_labels = labels[train_len:data_len]
    # put the data in the dataloader
    train_data = torch.from_numpy(train_data)
    train_labels = torch.from_numpy(train_labels)
    valiad_data = torch.from_numpy(valiad_data)
    valiad_labels = torch.from_numpy(valiad_labels)
    train_dataset = TensorDataset(train_data, train_labels)
    valid_dataset = TensorDataset(valiad_data, valiad_labels)
    # dataloader
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=valid_dataset,
                              batch_size=batch_size, shuffle=True)
    return train_loader, valid_loader


if __name__ == '__main__':
    data, labels = load_data("/home/leosher/data/pet/fMRI_BOLD/par100")
    print(len(data))
    print(data)
    print(labels)
