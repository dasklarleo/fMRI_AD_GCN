# dataloader: load the data from the matlab file
from numpy.core.fromnumeric import shape
import torch
from torch.cuda import random
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
from scipy.fftpack import fft,ifft
import matplotlib.pyplot as plt
from matplotlib.pylab import mpl
import os
import scipy.io as scio
import math
import random


def load_data(data_train_folder_path,data_test_folder_path,batch_size):
    data_train = []
    data_test=[]
    labels_train = []
    labels_test=[]

    files = os.listdir(data_train_folder_path)
    random.shuffle(files)
    for file in files:

        ROI_BOLDs = scio.loadmat(data_train_folder_path+'/'+file)['ROI_ts'].T
            
        for i in range(100):
            ROI_BOLDs_i_avg=np.mean(ROI_BOLDs[i])
            ROI_BOLDs_i_std=np.std(ROI_BOLDs[i])
            ROI_BOLDs[i]=ROI_BOLDs[i]-ROI_BOLDs_i_avg
            ROI_BOLDs[i]=ROI_BOLDs[i]/ROI_BOLDs_i_std

        data_train.append(ROI_BOLDs)


    data_train = np.array(data_train)
    labels_train = np.array(labels_train)
    labels_train=labels_train.reshape((labels_train.shape[0]))
    # randomly shuffle the data

    print(data_train.shape)
###################################################
##############TEST   DATA##########################
###################################################
    files = os.listdir(data_test_folder_path)
    for file in files:
        ROI_BOLDs = scio.loadmat(data_test_folder_path+'/'+file)['ROI_ts'].T

        for i in range(100):
            ROI_BOLDs_i_avg=np.mean(ROI_BOLDs[i])
            ROI_BOLDs_i_std=np.std(ROI_BOLDs[i])
            ROI_BOLDs[i]=ROI_BOLDs[i]-ROI_BOLDs_i_avg
            ROI_BOLDs[i]=ROI_BOLDs[i]/ROI_BOLDs_i_std

        data_test.append(ROI_BOLDs)


    data_test = np.array(data_test)
    labels_test = np.array(labels_test)
    labels_test=labels_test.reshape((data_test.shape[0]))
    print(data_test.shape)
    # randomly shuffle the data

    print("total train num: ",labels_train.shape[0]," eMCI num: ",(labels_train> 0).sum()," NC num: ",(labels_train<1).sum())
    print("total test num: ",labels_test.shape[0]," eMCI num: ",(labels_test >0).sum()," NC num: ",(labels_test<1).sum())
    # put the data in the dataloader
    train_data = torch.from_numpy(data_train).double()
    train_labels = torch.from_numpy(labels_train)

    test_data = torch.from_numpy(data_test).double()
    test_labels = torch.from_numpy(labels_test)

    train_dataset = TensorDataset(train_data, train_labels)
    test_dataset = TensorDataset(test_data, test_labels)
    # dataloader
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size, shuffle=True,drop_last=False)
    valid_loader = DataLoader(dataset=test_dataset,
                              batch_size=batch_size, shuffle=True,drop_last=False)
    return train_loader, valid_loader


if __name__ == '__main__':
    pass
