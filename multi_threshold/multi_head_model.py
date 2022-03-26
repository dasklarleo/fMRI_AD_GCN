from cmath import sqrt
import math
from tkinter.font import BOLD
from turtle import forward, shape
import torch
import random
from torch import autograd
from torch.autograd import Variable, variable
import torch.nn
import numpy as np
import torch.nn.functional as F
from torch.nn.modules import padding, transformer
from torch.nn.modules.activation import ReLU

# Method

class BuildCorrelation(torch.nn.Module):
    def __init__(self) -> None:
        super(BuildCorrelation,self).__init__()
        self.LinearProjectionQ=torch.nn.Linear(64,32)
        self.LinearProjectionK=torch.nn.Linear(64,32)
        self.LinearProjectionV=torch.nn.Linear(64,32)
    def forward(self,BOLDSignals,EmptyCorrelations):
        Q=self.LinearProjectionQ(BOLDSignals)
        K=self.LinearProjectionK(BOLDSignals)
        V=self.LinearProjectionV(BOLDSignals)
        K_T=torch.permute(K,(0,2,1))
        S=torch.bmm(Q,K_T)
        S=S/32
        A=F.softmax(S,dim=2)
        Feature=torch.bmm(A,V)
        for i in range(BOLDSignals.shape[0]):
            EmptyCorrelations[i]=torch.corrcoef(Feature[i])
        return(EmptyCorrelations)
        pass



class VanillaGCN(torch.nn.Module):
    def __init__(self, features_in, features_out):
        super(VanillaGCN, self).__init__()
        self.W = torch.nn.Linear(features_in, features_out)
        self.batch_normalization=torch.nn.BatchNorm1d(100)        
    def forward(self, Adjancy_Matrix, features):

        output = torch.tanh(
            self.batch_normalization(
                self.W(torch.matmul(Adjancy_Matrix, features))))

        return output



class GAP(torch.nn.Module):
    def __init__(self):
        super(GAP,self).__init__()
        self.gap=torch.nn.AdaptiveAvgPool1d(1)
    def forward(self,data):
        gap1=torch.squeeze(self.gap(data))
        gap2=torch.squeeze(self.gap(gap1))
        return gap2

class whole_network(torch.nn.Module):
    def __init__(self, batch_size, ROI_nums, BOLD_nums, q, 
                 feature_nums):
        super(whole_network, self).__init__()
        self.batch_size = batch_size
        self.feature_nums = feature_nums
        self.ROI_nums = ROI_nums
        self.BOLD_nums = BOLD_nums  # BOLD信号采样的个数
        self.q = q
        self.threshold = torch.tensor([[[0.6]],[[0.4]],[[-0.2]]])
        self.threshold = torch.nn.parameter.Parameter(self.threshold,requires_grad=True)
        self.relation_draw=torch.rand((100,100)).double()


        self.BuildCorrelation=BuildCorrelation()

        # vanilla GCN
        self.vanilla_gcn_layer1_1 = VanillaGCN(feature_nums, 32)
        self.vanilla_gcn_layer2_1 = VanillaGCN(32, 16)
        self.vanilla_gcn_layer3_1 = VanillaGCN(16, 8)
        self.vanilla_gcn_layer4_1 = VanillaGCN(8, 4)
        self.vanilla_gcn_layer5_1 = VanillaGCN(4, 2)

        self.vanilla_gcn_layer1_2 = VanillaGCN(feature_nums, 16)
        self.vanilla_gcn_layer3_2 = VanillaGCN(16, 4)
        self.vanilla_gcn_layer5_2 = VanillaGCN(4, 2)

        self.FinalLinear=torch.nn.Linear(200,2)


        self.GAP=GAP()
        #self.final_linear=torch.nn.Linear(200,2)
        self.avg2d=torch.nn.AdaptiveAvgPool2d((1,1))


        self.correlation_learn=None
        self.correlation=None

    
        for m in self.modules():
            if isinstance(m, torch.nn.Conv1d):
                torch.nn.init.kaiming_normal_(m.weight,nonlinearity='relu')
            elif isinstance(m, (torch.nn.BatchNorm1d)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m,torch.nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight,nonlinearity='relu')
                torch.nn.init.constant_(m.bias, 0.0)  
        
                    
    def forward(self, BOLD_signals):  # BOLD_signals[batch_size,ROIs,features]
        BOLD_signals=BOLD_signals.double()

        self.correlation=torch.rand((BOLD_signals.shape[0],100,100)).double().cuda()

        self.correlation=self.BuildCorrelation(BOLD_signals,self.correlation)
        

        
        vanilla_GCN_output1_1 = self.vanilla_gcn_layer1_1(self.correlation,BOLD_signals)
        vanilla_GCN_output2_1 = self.vanilla_gcn_layer2_1(self.correlation,vanilla_GCN_output1_1)
        vanilla_GCN_output3_1 = self.vanilla_gcn_layer3_1(self.correlation,vanilla_GCN_output2_1)
        vanilla_GCN_output4_1 = self.vanilla_gcn_layer4_1(self.correlation,vanilla_GCN_output3_1) 
        vanilla_GCN_output5_1 = self.vanilla_gcn_layer5_1(self.correlation,vanilla_GCN_output4_1) 


        vanilla_GCN_output5_1_squeze=torch.reshape(vanilla_GCN_output5_1,(vanilla_GCN_output5_1.shape[0],-1))

        '''        
        gap2=self.GAP(vanilla_GCN_output5_2)
        gap3=self.GAP(vanilla_GCN_output5_3)
        gap4=self.GAP(vanilla_GCN_output5_4)
        '''


        predict_labels = (self.FinalLinear(vanilla_GCN_output5_1_squeze))

        return predict_labels.float()
