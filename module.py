import torch
import random
import torch.nn
import numpy as np
import torch.nn.functional as F
from torch.nn.modules import padding
from torch.nn.modules.activation import ReLU

# Method
class Transofrmer(torch.nn.Module):
    def __init__(self,BatchSize,ROIs,heads,BOLDs,q):
        super(Transofrmer,self).__init__()
        self.BatchSize=BatchSize
        self.ROIs=ROIs
        self.heads=heads
        self.q=q
        self.linear_q=torch.nn.Linear(BOLDs,heads*q)
        self.linear_k=torch.nn.Linear(BOLDs,heads*q)

    def forward(self,x):
        # affine and get Q & K
        Q=self.linear_k(x).view((self.BatchSize,self.ROIs,self.heads,self.q))
        K=self.linear_k(x).view((self.BatchSize,self.ROIs,self.heads,self.q))
        # reallign to make Q:[BatchSize,heads,ROIs,q],K:[BatchSize,heads,q,ROIs]
        Q=torch.transpose(Q,(self.BatchSize,self.heads,self.ROIs,self.q))
        K=torch.transpose(K,(self.BatchSize,self.heads,self.q,self.ROIs))

        # calculate the relation
        relation=Q.dot(K)/torch.sqrt(self.q)
        # moved ReLU and softmax
        threshold=torch.randn(1)
        relation-=threshold
        relation=F.relu(relation)
        relation=F.softmax(relation,dim=1)

        return relation
#In the squeeze module what squeeze algorithm should I use?
class SE_Block(torch.nn.module):
    def __init__(self,SE_Parameter,heads):
        super(SE_Block,self).__init__()
        self.SE_Parameter=SE_Parameter
        self.heads=heads
        self.W_1=torch.nn.Linear((self.heads,self.heads/self.SE_Parameter))
        self.W_2=torch.nn.Linear((self.heads/self.SE_Parameter,self.heads))
    def forward(self,relation):
        #GAP:[1,1,batch_size,heads]
        GAP=F.adaptive_avg_pool2d(relation,(1,1)).permute(2,3,0,1)
        #bottle_neck:[1,1,batch_size,heads/r]
        bottle_neck=F.relu(self.W_1(GAP))
        #output
        output=F.sigmoid(self.W_2(bottle_neck))
        return output

# The first time we call the GCN module, we need to make the features from [Batchsize,ROIs,features] to [Batchsize,heads,ROIs,features] by repeating it heads times
class GCN(torch.nn.module):
    def __init__(self,features,dilated_parameter_k):
        super(GCN,self).__init__()
        self.dilated_parameter_k=dilated_parameter_k
        self.W=torch.nn.Linear(features,features)
        #GCN with dilated kernel
    def forward(self,Adjancy_Matrix,features):
        _,index=torch.sort(Adjancy_Matrix,dim=3)
        index[index==0]=-1

        prob=random.random()
    #calculate the index 
        # dilated convolution sample 10 points
        if(prob<0.8):
            for i in range(5):
                index[index==(i+pow(2,self.dilated_parameter_k))]=-1
            index[index!=-1]=0
            index[index==-1]=1

        # random convolution sample points sample 10 points (each batchsize is sampled with different points)
        if(prob>=0.8):
            pass
            (batch_size,heads,ROIs,_)=Adjancy_Matrix.shape
            samples=10
            random_conv=torch.rand(batch_size,heads,ROIs,ROIs)
            random_conv[random_conv>=0.9]=1
            random_conv[random_conv<1]=0
            index=index*random_conv
        index+=torch.eye(ROIs,ROIs)
        #确保只有1而没有2 从而防止对角线出现2
        index[index>=1]=1
        #没有添加D^(-1/2)归一化
        output1=F.relu((self.W(Adjancy_Matrix*index*features)))
        #ResNet
        return output1+features

#fusion the infomations in different heads
class fusion(torch.nn.modules):
    def __init__(self,heads,batch_size) -> None:
        super().__init__(fusion,self)
        self.heads=heads
        self.conv=torch.nn.Conv2d(in_channels=heads,out_channels=1,stride=1,padding=0)
    def forward(self,output):
        return (self.conv(output))
class MLP(torch.nn.modules):
    def __init__(self):
        super().__init__()
        pass
    def forward(self):
        pass


