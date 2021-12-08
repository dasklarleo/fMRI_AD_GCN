import torch
import torch.nn
import torch.nn.functional as F
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
class GCN(torch.nn.module):
    def __init__(self,BOLD_signals,dilated_parameter_k):
        super().__init__()
        self.BOLD_signals=BOLD_signals
        self.dilated_parameter_k=dilated_parameter_k
        #GCN with dilated kernel
    def forward(self,Adjancy_Matrix,features):
        
        pass
class Net1(torch.nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 3, 1, 1)
        self.dense1 = torch.nn.Linear(32 * 3 * 3, 128)
        self.dense2 = torch.nn.Linear(128, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv(x)), 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.dense1(x))
        x = self.dense2(x)
        return x

print("Method 1:")
model1 = Net1()
print(model1)