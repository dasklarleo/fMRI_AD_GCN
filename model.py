import math
import torch
import random
from torch import serialization
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
        self.threshold = torch.tensor(0.1)
        self.threshold = Variable(self.threshold.float(), requires_grad=True)
        self.linear_q = torch.nn.Linear(BOLDs, heads * q)
        self.linear_k = torch.nn.Linear(BOLDs, heads * q)

    def forward(self, x):
        # affine and get Q & K
        Q = self.linear_q(x)
        """遇到了这个问题RuntimeError: expected scalar type Double but found Float
            尝试：net=net.double()
        """
        Q = Q.reshape(
            (self.BatchSize, self.ROIs, self.heads, self.q))
        K = self.linear_k(x).reshape(
            (self.BatchSize, self.ROIs, self.heads, self.q))
        # reallign to make Q:[BatchSize,heads,ROIs,q],K:[BatchSize,heads,q,ROIs]
        Q = Q.permute(0, 2, 1, 3)
        K = K.permute(0, 2, 3, 1)

        # calculate the relation
        relation = torch.matmul(Q, K)
        relation = relation/torch.sqrt(self.div_q)

        # moved ReLU and softmax
        relation = F.softmax(relation, dim=3)

        relation -= self.threshold
        relation = F.relu(relation)
        return relation


# In the squeeze module what squeeze algorithm should I use?


class SE_Block(torch.nn.Module):
    def __init__(self, SE_Parameter, heads):
        super(SE_Block, self).__init__()
        self.SE_Parameter = SE_Parameter
        self.heads = heads
        self.W_1 = torch.nn.Linear(
            self.heads, self.heads // self.SE_Parameter)
        self.W_2 = torch.nn.Linear(
            self.heads // self.SE_Parameter, self.heads)

    def forward(self, relation):
        # GAP:[1,1,batch_size,heads]
        GAP = F.adaptive_avg_pool2d(relation, (1, 1)).permute(2, 3, 0, 1)
        # bottle_neck:[1,1,batch_size,heads/r]
        bottle_neck = F.relu(self.W_1(GAP))
        # output
        output = F.sigmoid(self.W_2(bottle_neck))
        for i in range(relation.shape[0]):
            for j in range(relation.shape[1]):
                relation[i, j] *= output[0, 0, i, j].long()
        return relation


# The first time we call the GCN module, we need to make the features
# from [Batchsize,ROIs,features] to [Batchsize,heads,ROIs,features] by repeating it heads times


class GCN(torch.nn.Module):
    def __init__(self, features, dilated_parameter_k):
        super(GCN, self).__init__()
        self.dilated_parameter_k = dilated_parameter_k
        self.W = torch.nn.Linear(features, features)
        # GCN with dilated kernel

    def forward(self, Adjancy_Matrix, features):
        _, index = torch.sort(Adjancy_Matrix, descending=True, dim=3)
        index[index == 0] = -1
        (batch_size, heads, ROIs, _) = Adjancy_Matrix.shape

        prob = random.random()
        # calculate the index
        # dilated convolution sample 10 points
        if (prob < 0.8):
            for i in range(5):
                index[index == (i + pow(2, self.dilated_parameter_k))] = -1
            index[index != -1] = 0
            index[index == -1] = 1

        # random convolution sample points sample 10 points (each batchsize is sampled with different points)
        if (prob >= 0.8):
            random_conv = torch.rand(batch_size, heads, ROIs, ROIs).to("cuda")
            random_conv[random_conv >= 0.9] = 1
            random_conv[random_conv < 1] = 0
            index = index * random_conv
        # 确保只有1而没有2 从而防止对角线出现2
        for i in range(index.shape[0]):
            for j in range(index.shape[1]):
                index[i, j] += torch.eye(ROIs).long().to("cuda")
        index[index >= 1] = 1
        # 没有添加D^(-1/2)归一化
        output1 = F.relu(
            (self.W(
                torch.matmul(
                    torch.mul(
                        Adjancy_Matrix, index),features))))
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
                                    kernel_size=1,
                                    stride=1,
                                    padding=0)

    def forward(self, output):
        return (self.conv(output))


'''
Output the classfication informations
    - inputs: [batch_size,ROIs,features]
    - outputs: [batch_size,predict_labels]
'''


class MLP(torch.nn.Module):
    def __init__(self, heads):
        super(MLP, self).__init__()
        self.MLP1 = torch.nn.Linear(130, 64)
        self.MLP2=torch.nn.Linear(64,16)
        self.MLP3=torch.nn.Linear(16,5)

    def forward(self, fusion_output):
        fusion_output=F.adaptive_avg_pool1d(fusion_output,1)
        fusion_output=fusion_output.reshape(fusion_output.shape[0],fusion_output.shape[1])
        print(fusion_output.shape)
        predict_labels = F.relu(self.MLP1(fusion_output))
        predict_labels = F.relu(self.MLP2(predict_labels))
        predict_labels = F.relu(self.MLP3(predict_labels))
        return predict_labels


class whole_network(torch.nn.Module):
    def __init__(self, batch_size, ROI_nums, heads, BOLD_nums, q, SE_parameter,
                 feature_nums, dilated_parameter_k):
        super(whole_network, self).__init__()
        self.heads = heads
        self.batch_size = batch_size
        self.feature_nums = feature_nums
        self.ROI_nums = ROI_nums
        self.BOLD_nums = BOLD_nums  # BOLD信号采样的个数
        self.transofrmer = Transofrmer(batch_size, ROI_nums, heads, BOLD_nums,
                                       q)
        self.se = SE_Block(SE_parameter, heads)
        self.gcn_layer1 = GCN(feature_nums, dilated_parameter_k)
        self.gcn_layer2 = GCN(feature_nums, dilated_parameter_k)
        self.gcn_layer3 = GCN(feature_nums, dilated_parameter_k)
        self.gcn_layer4 = GCN(feature_nums, dilated_parameter_k)
        self.gcn_layer5 = GCN(feature_nums, dilated_parameter_k)
        self.gcn_layer6 = GCN(feature_nums, dilated_parameter_k)
        self.gcn_layer7 = GCN(feature_nums, dilated_parameter_k)
        self.gcn_layer8 = GCN(feature_nums, dilated_parameter_k)
        self.fusion = Fusion(heads)
        self.mlp = MLP(feature_nums)

    def forward(self, BOLD_signals):  # BOLD_signals[batch_size,ROIs,features]

        # relation[batch_size,heads,ROIs,ROIs]
        relation = self.transofrmer(BOLD_signals)
        relation = self.se(relation)  # relation[batch_size,heads,ROIs,ROIs]

        # BOLD_signals[batch_size,1,ROI_nums,BOLD_nums]
        BOLD_signals = BOLD_signals.reshape(self.batch_size, 1, self.ROI_nums,
                                            self.BOLD_nums)

        BOLD_signals = torch.cat(
            (
                BOLD_signals,
                BOLD_signals,
                BOLD_signals,
                BOLD_signals,
                BOLD_signals,
                BOLD_signals,
                BOLD_signals,
                BOLD_signals,
            ),
            dim=1)  # BOLD_signals[batch_size,8(heads),ROI_nums,BOLD_nums]
        GCN_output = self.gcn_layer1(relation, BOLD_signals)
        GCN_output = self.gcn_layer2(relation, GCN_output)
        GCN_output = self.gcn_layer3(relation, GCN_output)
        GCN_output = self.gcn_layer4(relation, GCN_output)
        GCN_output = self.gcn_layer5(relation, GCN_output)
        GCN_output = self.gcn_layer6(relation, GCN_output)
        GCN_output = self.gcn_layer7(relation, GCN_output)
        # GCN_output:[batch_size,heads,ROIs,features]
        GCN_output = self.gcn_layer8(relation, GCN_output)

        # fusion_output:[batch_size,1,ROIs,features]
        print(GCN_output.shape)
        fusion_output = self.fusion(GCN_output)
        fusion_output = fusion_output.reshape(self.batch_size, self.ROI_nums,
                                              self.BOLD_nums)
        print(fusion_output.shape)
        # predict_labels:[batch_size,labels]
        predict_labels = self.mlp(fusion_output)
        print(predict_labels.shape)
        return predict_labels
