# dataloader: load the data from the matlab file
from numpy.core.fromnumeric import shape
import torch
from torch.cuda import random
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import os
import scipy.io as scio
import math
import random
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

def check_accuracy(loader,model):
    num_correct=0
    num_samples=0
    TP_total=0
    FN_total=0
    FP_total=0
    TN_total=0
    model.eval()
    for x,y in (loader):
        x=x.to(device=device)
        y=y.to(device=device)
        scores=model(x) #[data_num,5]
        _,predictions=scores.max(1)# we just care about the index of the maxium value but not the value of the maxium
        predictions_0=copy.deepcopy(predictions)
        predictions_1=copy.deepcopy(predictions)
        predictions_0[predictions_0==1]=100
        predictions_1[predictions_1==0]=100
        #1 预测正确的数量
        predictions_1_correct_num=(predictions_1==y).sum()
        #0 预测正确的数量
        predictions_0_correct_num=(predictions_0==y).sum()#TN
        #1 的总数量
        total_num_1=(y[y==1].sum())
        #0 的总数量
        total_num_0=(y.shape[0]-total_num_1)
        TP=predictions_1_correct_num.item()
        TN=predictions_0_correct_num.item()
        FN=(total_num_1-predictions_1_correct_num).item()
        FP=(total_num_0-predictions_0_correct_num).item()
        TP_total=TP_total+TP
        TN_total=TN_total+TN
        FP_total=FP_total+FP
        FN_total=FN_total+FN
        num_correct+=(predictions==y).sum()
        num_samples+=predictions.size(0)
    print(f'Got {TP_total+TN_total}/{TP_total+FP_total+TN_total+FN_total} with accuracy: {float(TP_total+TN_total)/float(TP_total+FP_total+TN_total+FN_total)*100:.2f}')
    print(f'Got {TP_total}/{TP_total+FN_total} with sensitivity(call back): {float(TP_total)/float(TP_total+FN_total)*100:.2f}')
    print(f'Got {TN_total}/{TN_total+FP_total} with specifity: {float(TN_total)/float(FP_total+TN_total)*100:.2f}')
    call=float(TP_total)/float(TP_total+FN_total)*100
    spe=float(TN_total)/float(FP_total+TN_total)*100
    print(f'Got F1 score: {(2*call*spe)/(call+spe):.2f}')

def load_data(data_train_folder_path,data_test_folder_path,batch_size):
    data_train = []
    data_test=[]
    labels_train = []
    labels_test=[]

    files = os.listdir(data_train_folder_path)
    random.shuffle(files)
    for file in files:
        if 'DFC' in file:
            continue
        if 'NC' in file:#70%
            labels_train.append(1)
        elif 'MCI' in file:#30%
            labels_train.append(0)
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

    print(data_train.shape)
###################################################
##############TEST   DATA##########################
###################################################
    files = os.listdir(data_test_folder_path)
    for file in files:
        if 'DFC' in file:
            continue
        if 'NC' in file:
            labels_test.append(1)
        elif 'MCI' in file:
            labels_test.append(0)
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
    print(data_test.shape)
    # randomly shuffle the data

    print("total train num: ",labels_train.shape[0]," NC num: ",(labels_train> 0).sum()," MCI num: ",(labels_train<1).sum())
    print("total test num: ",labels_test.shape[0]," NC num: ",(labels_test >0).sum()," MCI num: ",(labels_test<1).sum())
    # put the data in the dataloader
    train_data = torch.from_numpy(data_train)
    train_labels = torch.from_numpy(labels_train)

    test_data = torch.from_numpy(data_test)
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
    # load data
    train_loader,test_loader = load_data("/home/leosher/data/pet/fMRI_BOLD/par100/NC_MCI/train","/home/leosher/data/pet/fMRI_BOLD/par100/NC_MCI/test",batch_size=16)
    net=torch.load("/home/leosher/桌面/project/fMRI_AD_GCN/model/modelNC_eMCI").to(device)
    check_accuracy(train_loader,net)
    check_accuracy(test_loader,net)