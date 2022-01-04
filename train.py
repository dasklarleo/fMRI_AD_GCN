'''
In this file the hyper-paramters will be set
The training loops would also be set
'''
from os import access
import dataloader
import model
import logging
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
print(device)
# set parameters
batch_size = 16
ROI_nums = 130
heads = 2
BOLD_nums = 300
q = 128
SE_parameter = 4
feature_nums = 300
dilated_parameter_k = 2
learning_rate = 0.0001
num_epoches = 100

logging.basicConfig(level=logging.INFO,
filename='./log.txt',
filemode='w',
format='%(asctime)s-%(message)s')

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    # 也可以判断是否为conv2d，使用相应的初始化方式 
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
     # 是否为批归一化层
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

def check_accuracy(loader,model):

    num_correct=0
    num_samples=0
    model.eval()
    for x,y in (loader):
        x=x.to(device=device)
        y=y.to(device=device)

        scores=model(x) #[data_num,5]
        _,predictions=scores.max(1)# we just care about the index of the maxium value but not the value of the maxium
        num_correct+=(predictions==y).sum()
        num_samples+=predictions.size(0)
    print(f'Got {num_correct}/{num_samples} with accuracy{float(num_correct)/float(num_samples)*100:.2f}')
    logging.info(num_correct/num_samples)


if __name__ == '__main__':


    torch.autograd.set_detect_anomaly(True)
    CUDA_LAUNCH_BLOCKING=1
    # load data
    #train_loader,valid_loader = dataloader.load_data("/home/leosher/data/pet/fMRI_BOLD/par"+str(BOLD_nums), batch_size)
    train_loader,valid_loader = dataloader.load_data("/home/leosher/data/pet/fMRI_BOLD/train","/home/leosher/data/pet/fMRI_BOLD/test",batch_size=batch_size)
    # initialize the network
    net = model.whole_network(batch_size=batch_size,
                              ROI_nums=ROI_nums,
                              heads=heads,
                              BOLD_nums=BOLD_nums,
                              q=q,
                              SE_parameter=SE_parameter,
                              feature_nums=feature_nums,
                              dilated_parameter_k=dilated_parameter_k).to(device)
    net = net.double()
    net.apply(weight_init)
    # loss and optimizer
    criterion=nn.CrossEntropyLoss()
    optimizer=optim.Adam(net.parameters(),lr=learning_rate)
    loss_all=[]
    # train the network
    for epoch in range(100):# one epoch means that the net has seen all the data in the dataset

        for batch_idx,(data,target) in enumerate(train_loader):
            data=data.to(device)
            target=target.to(device)
    # forward
            scores=net(data)
            loss=criterion(scores,target)
            record_info='epoch: '+str(epoch)+" iteration: "+str(batch_idx)+" loss: "+str(loss.item())
            loss_all.append(loss.item())
            logging.info(record_info)
    # backward
            optimizer.zero_grad()
            loss.backward()
    # gradient desecnt or adam step
            optimizer.step()
        if epoch%10==0:
            print("*"*10)
            print("epoch:", epoch)
            check_accuracy(train_loader,net)
            check_accuracy(valid_loader,net)
            
    plt.plot(loss_all)
    plt.show()