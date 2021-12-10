#dataloader: load the data from the matlab file
import torch
from torch.utils.data.sampler import SubsetRandomSampler
import random
import sys
import numpy as np
import pickle
import os
import csv
import scipy.io as scio
def load_data(data_folder_path,atlas):
    for file in data_folder_path:
        ROICorrelation = scio.loadmat(data_folder_path+file)['ROICorrelation']
        ROICorrelation=ROICorrelation.reshape(1,-1)
        ROICorrelation=np.column_stack((ROICorrelation,1))
        data=np.row_stack((data,ROICorrelation))


