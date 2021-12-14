'''
In this file the hyper-paramters will be set
The training loops would also be set
'''
from os import access
import dataloader
import model
import torch
import torch.nn as nn
import torch.optim as optim
# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
print(device)
# set parameters
batch_size = 16
ROI_nums = 130
heads = 8
BOLD_nums = 300
q = 128
SE_parameter = 4
feature_nums = 300
dilated_parameter_k = 2
learning_rate = 0.001
num_epoches = 100

if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    CUDA_LAUNCH_BLOCKING=1
    # load data
    train_loader,valid_loader = dataloader.load_data("/home/leosher/data/pet/fMRI_BOLD/par"+str(BOLD_nums), batch_size)
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
    # loss and optimizer
    criterion=nn.CrossEntropyLoss()
    optimizer=optim.Adam(net.parameters(),lr=learning_rate)
    # train the network
    for epoch in range(num_epoches):# one epoch means that the net has seen all the data in the dataset
        for batch_idx,(data,target) in enumerate(train_loader):
            data=data.to(device=device)
            target=target.to(device=device)
    # forward
            scores=net(data)
            loss=criterion(scores,target)
    # backward
            optimizer.zero_grad()
            print("*"*12)
            loss.backward()
    # gradient desecnt or adam step
            optimizer.step()
# check the accuracy
def check_accuracy(loader,model):
    if loader.dataset.train:
        print("checking accuracy on the training dataset")
    else:
        print("checking accuracy on the validation dataset")
    num_correct=0
    num_samples=0
    model.eval()

    with torch.no_grad():
        for x,y in (loader):
            x=x.to(device=device)
            y=y.to(device=device)

            scores=model(x) #[data_num,5]
            _,predictions=scores.max(1)# we just care about the index of the maxium value but not the value of the maxium
            num_correct+=(predictions==y).sum()
            num_samples+=predictions.size(0)
        print(f'Got {num_correct}/{num_samples} with accuracy{float(num_correct)/float(num_samples)*100:.2f}')
    model.train()
    #return acc
check_accuracy(train_loader,model)
check_accuracy(valid_loader,model)