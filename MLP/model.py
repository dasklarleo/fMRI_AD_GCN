import math
import torch
import random
from torch import serialization
from torch import autograd
from torch.autograd import Variable, variable
import torch.nn
import numpy as np
import torch.nn.functional as F
from torch.nn.modules import padding, transformer
from torch.nn.modules.activation import ReLU

# Method


class Transofrmer(torch.nn.Module):
    def __init__(self, BatchSize, ROIs, BOLDs, q):
        super(Transofrmer, self).__init__()
        self.BatchSize = BatchSize
        self.ROIs = ROIs
        self.q = q



        self.linear_q = torch.nn.Linear(BOLDs, q)
        #self.linear_k = torch.nn.Linear(BOLDs, q)
        self.relu_layer = torch.nn.ReLU(inplace=False)

        self.bn=torch.nn.BatchNorm1d(130)

    def forward(self, x,threshold):
        # affine and get Q & K
        x = x.double()

        Q = self.linear_q(x).reshape(
            (-1, self.ROIs, self.q))
        K = self.linear_q(x).reshape(
            (-1, self.q,self.ROIs))

        relation = torch.matmul(Q, K).double()

        #relation = relation/self.q
        relation=self.bn(relation)
        relation = F.softmax(relation,dim=2)
        relation =relation - threshold

        #print(threshold)
        relation = self.relu_layer(relation)
        return relation


class VanillaGCN(torch.nn.Module):
    def __init__(self, features_in, features_out):
        super(VanillaGCN, self).__init__()
        self.W = torch.nn.Linear(features_in, features_out)
        self.batch_normalization=torch.nn.BatchNorm1d(130)        
    def forward(self, Adjancy_Matrix, features):

        output = F.relu(
            self.batch_normalization(
                self.W(torch.matmul(Adjancy_Matrix, features))),
                         inplace=False)

        return output

class VanillaResGCN(torch.nn.Module):
    def __init__(self, features_in, features_out):
        super(VanillaResGCN, self).__init__()
        self.W = torch.nn.Linear(features_in, features_out)
        self.batch_normalization=torch.nn.BatchNorm1d(130)
    def forward(self, Adjancy_Matrix, features):
        output1 = F.relu(
            self.batch_normalization(
                                    self.W(torch.matmul(Adjancy_Matrix, features))),
                        inplace=False)
                        # ResNet
        return output1 + features
'''
fusion the infomations in different heads
    - inputs: [batch_size,heads,ROIs,features]
    - outputs: [batch_size,1,ROIs,features]
'''
class Fusion(torch.nn.Module):
    def __init__(self, heads) -> None:
        super(Fusion, self).__init__()
        self.heads = heads
        self.conv = torch.nn.Conv2d(in_channels=heads,
                                    out_channels=1,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0)

    def forward(self, output):
        return (self.conv(output).squeeze())


'''
Output the classfication informations
    - inputs: [batch_size,ROIs,features]
    - outputs: [batch_size,predict_labels]
'''


class MLP(torch.nn.Module):
    def __init__(self, heads):
        super(MLP, self).__init__()
        self.MLP1 = torch.nn.Linear(130*2,2)
        self.MLP2 = torch.nn.Linear(16, 2)
        #self.MLP3 = torch.nn.Linear(16,2)

    def forward(self, fusion_output):

        fusion_output1 = fusion_output.reshape(fusion_output.shape[0],-1)
        predict_labels = F.relu(self.MLP1(fusion_output1), inplace=False)
        #predict_labels1 = F.relu(self.MLP2(predict_labels), inplace=False)
        #predict_labels2 = F.relu(self.MLP3(predict_labels1), inplace=False)
        return predict_labels


class whole_network(torch.nn.Module):
    def __init__(self, batch_size, ROI_nums, BOLD_nums, q, 
                 feature_nums):
        super(whole_network, self).__init__()
        self.batch_size = batch_size
        self.feature_nums = feature_nums
        self.ROI_nums = ROI_nums
        self.BOLD_nums = BOLD_nums  # BOLD信号采样的个数
        self.q = q
        self.threshold = torch.tensor(0.005)
        self.threshold = torch.nn.parameter.Parameter(self.threshold,requires_grad=True)
        self.transofrmer = Transofrmer(batch_size, ROI_nums, BOLD_nums,
                                       q)

        # vanilla GCN
        self.vanilla_gcn_layer1 = VanillaGCN(feature_nums, 16)
        self.vanilla_gcn_layer2 = VanillaResGCN(16, 16)
        self.vanilla_gcn_layer3 = VanillaGCN(16, 4)
        self.vanilla_gcn_layer4 = VanillaResGCN(4 ,4)
        self.vanilla_gcn_layer5 = VanillaGCN(4, 2)
        self.vanilla_gcn_layer6 = VanillaResGCN(2, 2)
        self.vanilla_gcn_layer7 = VanillaGCN(32, 8)
        self.vanilla_gcn_layer8 = VanillaResGCN(8, 8)
        self.vanilla_gcn_layer9 = VanillaGCN(8, 2)
        self.vanilla_gcn_layer10 = VanillaResGCN(2, 2)

        self.mlp = MLP(feature_nums)

        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight,nonlinearity='relu')
            elif isinstance(m, (torch.nn.BatchNorm1d)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m,torch.nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight,nonlinearity='relu')
                torch.nn.init.constant_(m.bias, 0.0)   
                    
    def forward(self, BOLD_signals):  # BOLD_signals[batch_size,ROIs,features]
        BOLD_signals = torch.where(torch.isnan(BOLD_signals),
                                   torch.full_like(BOLD_signals, 0),
                                   BOLD_signals)

        # relation[batch_size,heads,ROIs,ROIs]
        relation_matrix= self.transofrmer(BOLD_signals,self.threshold)


        vanilla_GCN_output1 = self.vanilla_gcn_layer1(relation_matrix,BOLD_signals)
        vanilla_GCN_output2 = self.vanilla_gcn_layer2(relation_matrix,vanilla_GCN_output1) 
        vanilla_GCN_output3 = self.vanilla_gcn_layer3(relation_matrix,vanilla_GCN_output2) 
        vanilla_GCN_output4 = self.vanilla_gcn_layer4(relation_matrix,vanilla_GCN_output3) 
        vanilla_GCN_output5 = self.vanilla_gcn_layer5(relation_matrix,vanilla_GCN_output4) 
        vanilla_GCN_output6 = self.vanilla_gcn_layer6(relation_matrix,vanilla_GCN_output5)
        '''
        vanilla_GCN_output7 = self.vanilla_gcn_layer7(relation_matrix,vanilla_GCN_output6) 
        vanilla_GCN_output8 = self.vanilla_gcn_layer8(relation_matrix,vanilla_GCN_output7) 
        vanilla_GCN_output9 = self.vanilla_gcn_layer9(relation_matrix,vanilla_GCN_output8) 
        vanilla_GCN_output10 = self.vanilla_gcn_layer10(relation_matrix,vanilla_GCN_output9)
        '''         
        # predict_labels:[batch_size,labels]

        predict_labels = (self.mlp(vanilla_GCN_output6))
        '''predict_labels=vanilla_GCN_output6.reshape(-1,200)
        print(predict_labels.shape)
        predict_labels=F.adaptive_avg_pool1d(predict_labels,1)
        print(predict_labels.shape)
        predict_labels=predict_labels.reshape(-1,1)'''
        return predict_labels
