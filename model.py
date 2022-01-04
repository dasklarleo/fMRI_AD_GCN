import math
import torch
import random
from torch import serialization
from torch import autograd
from torch.autograd import Variable
import torch.nn
import numpy as np
import torch.nn.functional as F
from torch.nn.modules import padding, transformer
from torch.nn.modules.activation import ReLU

# Method


class Transofrmer(torch.nn.Module):
    def __init__(self, BatchSize, ROIs, heads, BOLDs, q):
        super(Transofrmer, self).__init__()
        self.BatchSize = BatchSize
        self.ROIs = ROIs
        self.heads = heads
        self.q = q

        self.div_q = torch.tensor(q)
        self.threshold = torch.tensor(0.01)
        self.threshold = Variable(self.threshold.double(),
                                  requires_grad=True).to(
                                      "cuda") 

        self.linear_q = torch.nn.Linear(BOLDs, heads * q)
        self.linear_k = torch.nn.Linear(BOLDs, heads * q)
        self.relu_layer = torch.nn.ReLU(inplace=False)
        self.sofmax_layer = torch.nn.Softmax(dim=3)

    def forward(self, x):
        # affine and get Q & K
        """
        遇到了这个问题RuntimeError: expected scalar type Double but found Float
        尝试:net=net.double()
        """
        x = x.double()
        Q = self.linear_q(x)
        Q = Q.reshape((self.BatchSize, self.ROIs, self.heads, self.q))
        K = self.linear_k(x).reshape(
            (self.BatchSize, self.ROIs, self.heads, self.q))
        # reallign to make Q:[BatchSize,heads,ROIs,q],K:[BatchSize,heads,q,ROIs]
        Q = Q.permute(0, 2, 1, 3).double()
        K = K.permute(0, 2, 3, 1).double()


        relation = torch.matmul(Q, K).double()

        relation = relation/(self.div_q)

        relation = F.softmax(relation,dim=3)
        relation =relation - self.threshold


        relation = self.relu_layer(relation)
        return relation

# from [Batchsize,ROIs,features] to [Batchsize,heads,ROIs,features] by repeating it heads times


class VanillaGCN(torch.nn.Module):
    def __init__(self, features_in, features_out, dilated_parameter_k):
        super(VanillaGCN, self).__init__()
        self.dilated_parameter_k = dilated_parameter_k
        self.W = torch.nn.Linear(features_in, features_out)
        self.batch_normalization=torch.nn.BatchNorm1d(130)        
    def forward(self, Adjancy_Matrix, features):

        output = F.relu(self.batch_normalization(self.W(torch.matmul(Adjancy_Matrix, features))),
                         inplace=False)

        return output

class VanillaResGCN(torch.nn.Module):
    def __init__(self, features_in, features_out, dilated_parameter_k):
        super(VanillaResGCN, self).__init__()
        self.dilated_parameter_k = dilated_parameter_k
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
        self.MLP1 = torch.nn.Linear(130*16,512)
        self.MLP2 = torch.nn.Linear(512, 32)
        self.MLP3 = torch.nn.Linear(32,2)

    def forward(self, fusion_output):

        fusion_output1 = fusion_output.reshape(fusion_output.shape[0],-1)
        predict_labels = F.relu(self.MLP1(fusion_output1), inplace=False)
        predict_labels1 = F.relu(self.MLP2(predict_labels), inplace=False)
        predict_labels2 = F.relu(self.MLP3(predict_labels1), inplace=False)
        return predict_labels2


class whole_network(torch.nn.Module):
    def __init__(self, batch_size, ROI_nums, heads, BOLD_nums, q, SE_parameter,
                 feature_nums, dilated_parameter_k):
        super(whole_network, self).__init__()
        self.heads = heads
        self.batch_size = batch_size
        self.feature_nums = feature_nums
        self.ROI_nums = ROI_nums
        self.BOLD_nums = BOLD_nums  # BOLD信号采样的个数
        self.q = q
        self.softmax_layer=torch.nn.Softmax(dim=3)
        self.transofrmer = Transofrmer(batch_size, ROI_nums, heads, BOLD_nums,
                                       q)

        # vanilla GCN
        self.vanilla_gcn_layer1 = VanillaGCN(feature_nums, 256, 0)
        self.vanilla_gcn_layer2 = VanillaResGCN(256, 256, 0)
        self.vanilla_gcn_layer3 = VanillaGCN(256, 128, 0)
        self.vanilla_gcn_layer4 = VanillaResGCN(128, 128, 0)
        self.vanilla_gcn_layer5 = VanillaGCN(128, 64, 0)
        self.vanilla_gcn_layer6 = VanillaResGCN(64, 64, 0)
        self.vanilla_gcn_layer7 = VanillaGCN(64, 32, 0)
        self.vanilla_gcn_layer8 = VanillaResGCN(32, 32, 0)
        self.vanilla_gcn_layer9 = VanillaGCN(32, 16, 0)
        self.vanilla_gcn_layer10 = VanillaResGCN(16, 16, 0)

        self.fusion = Fusion(heads)
        self.mlp = MLP(feature_nums)

    def forward(self, BOLD_signals):  # BOLD_signals[batch_size,ROIs,features]
        BOLD_signals = torch.where(torch.isnan(BOLD_signals),
                                   torch.full_like(BOLD_signals, 0),
                                   BOLD_signals)

        # relation[batch_size,heads,ROIs,ROIs]
        relation_matrix= self.transofrmer(BOLD_signals)

        relation_matrix=self.fusion(relation_matrix)

        vanilla_GCN_output1 = self.vanilla_gcn_layer1(relation_matrix,BOLD_signals)
        vanilla_GCN_output2 = self.vanilla_gcn_layer2(relation_matrix,vanilla_GCN_output1) 
        vanilla_GCN_output3 = self.vanilla_gcn_layer3(relation_matrix,vanilla_GCN_output2) 
        vanilla_GCN_output4 = self.vanilla_gcn_layer4(relation_matrix,vanilla_GCN_output3) 
        vanilla_GCN_output5 = self.vanilla_gcn_layer5(relation_matrix,vanilla_GCN_output4) 
        vanilla_GCN_output6 = self.vanilla_gcn_layer6(relation_matrix,vanilla_GCN_output5) 
        vanilla_GCN_output7 = self.vanilla_gcn_layer7(relation_matrix,vanilla_GCN_output6) 
        vanilla_GCN_output8 = self.vanilla_gcn_layer8(relation_matrix,vanilla_GCN_output7) 
        vanilla_GCN_output9 = self.vanilla_gcn_layer9(relation_matrix,vanilla_GCN_output8) 
        vanilla_GCN_output10 = self.vanilla_gcn_layer10(relation_matrix,vanilla_GCN_output9)        
        # predict_labels:[batch_size,labels]

        predict_labels = torch.softmax(self.mlp(vanilla_GCN_output10),dim=1)
        return predict_labels
