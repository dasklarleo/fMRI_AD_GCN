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
from tensorboardX import SummaryWriter
# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
print(device)
# set parameters
batch_size = 16
ROI_nums = 300
BOLD_nums = 130
q = 32
feature_nums = 130
learning_rate = 0.0001
num_epoches = 100

logging.basicConfig(level=logging.INFO,
filename='./log.txt',
filemode='w',
format='%(asctime)s-%(message)s')


def check_accuracy(loader,model):

    num_correct=0
    num_samples=0
    model.eval()
    for x,y in (loader):
        x=x.to(device=device)
        y=y.to(device=device)

        scores=model(x) #[data_num,5]
        _,predictions=scores.max(1)# we just care about the index of the maxium value but not the value of the maxium
        print(predictions)
        predictions_1=(predictions==y and predictions[predictions==1]).sum()
        predictions_0=(predictions==y and predictions[predictions==0]).sum()
        print(predictions_0,predictions_1)
        num_correct+=(predictions==y).sum()
        num_samples+=predictions.size(0)
    print(f'Got {num_correct}/{num_samples} with accuracy{float(num_correct)/float(num_samples)*100:.2f}')
    logging.info(num_correct/num_samples)

def get_logger(logger_name,log_file,level=logging.INFO):
	logger = logging.getLogger(logger_name)
	formatter = logging.Formatter('%(asctime)s-%(message)s')
	fileHandler = logging.FileHandler(log_file, mode='a')
	fileHandler.setFormatter(formatter)

	logger.setLevel(level)
	logger.addHandler(fileHandler)

	return logger

if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    CUDA_LAUNCH_BLOCKING=1

    # load data
    train_loader,valid_loader = dataloader.load_data("/home/leosher/data/pet/fMRI_BOLD/train","/home/leosher/data/pet/fMRI_BOLD/test",batch_size=batch_size)
    log_backward='/home/leosher/桌面/project/fMRI_AD_GCN/log/log_gradient.log'
    log_loss='/home/leosher/桌面/project/fMRI_AD_GCN/log/log_loss.log'
    log_backward=get_logger('log_backward',log_backward)
    log_loss=get_logger('log_loss',log_loss)
    # initialize the network
    net = model.whole_network(batch_size=batch_size,
                              ROI_nums=ROI_nums,
                              BOLD_nums=BOLD_nums,
                              q=q,
                              feature_nums=feature_nums).to(device)
    net = net.double()
    # loss and optimizer
    criterion=nn.CrossEntropyLoss()
    optimizer=optim.Adam(net.parameters(),lr=learning_rate)
    loss_all=[]
    loss_average=0
    # train the network
    for epoch in range(100):# one epoch means that the net has seen all the data in the dataset
        for batch_idx,(data,target) in enumerate(train_loader):
            net.train()
            data=data.to(device)
            target=target.to(device)
    # forward
            scores=net(data)
            loss=criterion(scores,target)
            record_info='epoch: '+str(epoch)+" iteration: "+str(batch_idx)+" loss: "+str(loss.item())
            log_loss.info(record_info)
            loss_average=loss_average+loss.item()

    # backward
            optimizer.zero_grad()
            loss.backward()

            for name, parms in net.named_parameters():	
                log_backward.info('epoch: '+str(epoch)+' batch index: '+str(batch_idx)+'-->name '+str(name)+' -->grad_requirs: '+str(parms.requires_grad)+\
		 ' -->grad_value: '+str(parms.grad))

            optimizer.step()
        loss_average=loss_average/(batch_idx+1)
        loss_all.append(loss_average)
        loss_average=0
        if epoch%10==0:
            print("*"*10)
            print("epoch:", epoch)
            check_accuracy(train_loader,net)
            check_accuracy(valid_loader,net)
            
    plt.plot(loss_all)
    plt.savefig('/home/leosher/桌面/project/fMRI_AD_GCN/testblueline.jpg')
