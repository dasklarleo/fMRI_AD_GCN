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
def percentage2n(eigVals,percentage=0.99):
    sortArray=np.sort(eigVals)   #升序
    sortArray=sortArray[-1::-1]  #逆转，即降序
    arraySum=sum(sortArray)
    tmpSum=0
    num=0
    for i in sortArray:
        tmpSum+=i
        num+=1
        if tmpSum>=arraySum*percentage:
            return num

def load_data(data_train_folder_path,data_test_folder_path,batch_size):
    data_train = []
    data_test=[]
    labels_train = []
    labels_test=[]
    percentage=0.99
    files = os.listdir(data_train_folder_path)
    random.shuffle(files)
    for file in files:
        if 'DFC' in file:
            continue
        if 'NC' in file:#70%
            labels_train.append(0)
        elif 'MCI' in file:#30%
            labels_train.append(1)

        ROI_BOLDs = scio.loadmat(data_train_folder_path+'/'+file)['ROI_ts'].T
        mu=np.mean(ROI_BOLDs,axis=0)
        sigma=np.std(ROI_BOLDs,axis=0)    
        for i in range(ROI_BOLDs.shape[1]):
            #ROI_BOLDs[:,i]=(ROI_BOLDs[:,i]-mu[i])/sigma[i]
            ROI_BOLDs[:,i]=(ROI_BOLDs[:,i]-mu[i])
        #n=percentage2n(eigVals,percentage)
        covMat=np.cov(ROI_BOLDs,rowvar=0)    #求协方差矩阵,return ndarray；若rowvar非0，一列代表一个样本，为0，一行代表一个样本
        eigVals,eigVects=np.linalg.eig(np.mat(covMat))#求特征值和特征向量,特征向量是按列放的，即一列代表一个特征向量
        eigValIndice=np.argsort(eigVals)            #对特征值从小到大排序
        n_eigValIndice=eigValIndice[-1:-(64+1):-1]   #最大的n个特征值的下标
        n_eigVect=eigVects[:,n_eigValIndice]        #最大的n个特征值对应的特征向量
        lowDDataMat=ROI_BOLDs*n_eigVect               #低维特征空间的数据
    
        data_train.append(lowDDataMat)


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
        print(file)
        if 'DFC' in file:
            continue
        if 'NC' in file:
            labels_test.append(0)
        elif 'MCI' in file:
            labels_test.append(1)
        ROI_BOLDs = scio.loadmat(data_test_folder_path+'/'+file)['ROI_ts'].T

        mu=np.mean(ROI_BOLDs,axis=0)
        sigma=np.std(ROI_BOLDs,axis=0)    
        for i in range(ROI_BOLDs.shape[1]):
            #ROI_BOLDs[:,i]=(ROI_BOLDs[:,i]-mu[i])/sigma[i]
            ROI_BOLDs[:,i]=(ROI_BOLDs[:,i]-mu[i])

        covMat=np.cov(ROI_BOLDs,rowvar=0)    #求协方差矩阵,return ndarray；若rowvar非0，一列代表一个样本，为0，一行代表一个样本
        eigVals,eigVects=np.linalg.eig(np.mat(covMat))#求特征值和特征向量,特征向量是按列放的，即一列代表一个特征向量
        eigValIndice=np.argsort(eigVals)            #对特征值从小到大排序
        n_eigValIndice=eigValIndice[-1:-(64+1):-1]   #最大的n个特征值的下标
        n_eigVect=eigVects[:,n_eigValIndice]        #最大的n个特征值对应的特征向量
        lowDDataMat=ROI_BOLDs*n_eigVect               #低维特征空间的数据
        
        '''if file=='EMCI_002_S_2043_201109_06.mat':
            matDict={"ROI_BOLDs":ROI_BOLDs}
            scio.savemat("/home/leosher/桌面/ROI_BOLD_signals.mat",matDict)
            matDict={"ROI_BOLDs":lowDDataMat}
            scio.savemat("/home/leosher/桌面/ROI_BOLD_signals_PCA.mat",matDict)'''
        data_test.append(lowDDataMat)


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
