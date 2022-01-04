# dataloader: load the data from the matlab file
from numpy.core.fromnumeric import shape
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import os
import scipy.io as scio
import math

def delete(folder_path,keyword):
    for filename in (os.listdir(folder_path)):
        if keyword in filename:
            os.remove(folder_path+filename)


if __name__ == '__main__':
    delete("/home/leosher/data/pet/fMRI_BOLD/two_300/NC/",'DFC')
    delete("/home/leosher/data/pet/fMRI_BOLD/two_300/MCI/",'DFC')
    delete("/home/leosher/data/pet/fMRI_BOLD/two_300/eMCI/",'DFC')
    delete("/home/leosher/data/pet/fMRI_BOLD/two_300/LMCI/",'DFC')
    delete("/home/leosher/data/pet/fMRI_BOLD/two_300/smc/",'DFC')
