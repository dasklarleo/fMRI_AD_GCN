# dataloader: load the data from the matlab file
from numpy.core.fromnumeric import shape
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import os
import scipy.io as scio
import math


def load_data(data_train_folder_path,data_test_folder_path,batch_size):
    data_train = []
    data_test=[]
    labels_train = []
    labels_test=[]

    files = os.listdir(data_train_folder_path)
    for file in files:
        if 'DFC' in file:
            continue
        if 'NC' in file:
            labels_train.append(0)
        elif 'EMCI' in file:
            labels_train.append(1)
        '''
        elif 'AD' in file:
            labels.append(2)
        elif 'LMCI' in file:
            labels.append(3)
        elif 'SMC' in file:
            labels.append(4)
        elif 'MCI' in file:
            labels.append(5)
        '''
        ROI_BOLDs = scio.loadmat(data_train_folder_path+'/'+file)['ROI_ts']
        data_train.append(ROI_BOLDs)

    data_train = np.array(data_train)
    labels_train = np.array(labels_train)
    labels_train=labels_train.reshape((labels_train.shape[0]))
    # randomly shuffle the data
    state = np.random.get_state()
    np.random.shuffle(data_train)
    np.random.set_state(state)
    np.random.shuffle(labels_train)
###################################################
##############TEST   DATA##########################
    files = os.listdir(data_test_folder_path)
    for file in files:
        if 'DFC' in file:
            continue
        if 'NC' in file:
            labels_test.append(0)
        elif 'EMCI' in file:
            labels_test.append(1)
        '''
        elif 'AD' in file:
            labels.append(2)
        elif 'LMCI' in file:
            labels.append(3)
        elif 'SMC' in file:
            labels.append(4)
        elif 'MCI' in file:
            labels.append(5)
        '''
        ROI_BOLDs = scio.loadmat(data_test_folder_path+'/'+file)['ROI_ts']
        data_test.append(ROI_BOLDs)

    data_test = np.array(data_test)
    labels_test = np.array(labels_test)
    labels_test=labels_test.reshape((data_test.shape[0]))
    # randomly shuffle the data
    state = np.random.get_state()
    np.random.shuffle(data_test)
    np.random.set_state(state)
    np.random.shuffle(labels_test)





    print("total train num: ",labels_train.shape[0]," eMCI num: ",(labels_train> 0).sum()," NC num: ",(labels_train<1).sum())
    print("total test num: ",labels_test.shape[0]," eMCI num: ",(labels_test >0).sum()," NC num: ",(labels_test<1).sum())
    # put the data in the dataloader
    train_data = torch.from_numpy(data_train)
    train_labels = torch.from_numpy(labels_train)

    test_data = torch.from_numpy(data_test)
    test_labels = torch.from_numpy(labels_test)

    train_dataset = TensorDataset(train_data, train_labels)
    test_dataset = TensorDataset(test_data, test_labels)
    # dataloader
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size, shuffle=True,drop_last=True)
    valid_loader = DataLoader(dataset=test_dataset,
                              batch_size=batch_size, shuffle=True,drop_last=True)
    return train_loader, valid_loader


if __name__ == '__main__':
    data, labels = load_data("/home/leosher/data/pet/fMRI_BOLD/par100",16)
    print(len(data))
    print(data)
    print(labels)
