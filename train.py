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
import copy
from sklearn.metrics import roc_auc_score
#from tensorboardX import SummaryWriter
torch.autograd.set_detect_anomaly(True)
CUDA_LAUNCH_BLOCKING=1

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
print(device)
# set parameters
batch_size = 8
ROI_nums = 100
BOLD_nums = 130
q = 16
feature_nums = 130
learning_rate = 0.001
num_epoches = 10

sens=[]
specifity=[]
acc=[]
f1=[]
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
    acc.append(float(TP_total+TN_total)/float(TP_total+FP_total+TN_total+FN_total)*100)
    specifity.append(float(TN_total)/float(FP_total+TN_total)*100)
    sens.append(float(TP_total)/float(TP_total+FN_total)*100)
    print(f'Got {TP_total+TN_total}/{TP_total+FP_total+TN_total+FN_total} with accuracy: {float(TP_total+TN_total)/float(TP_total+FP_total+TN_total+FN_total)*100:.2f}')
    print(f'Got {TP_total}/{TP_total+FN_total} with sensitivity(call back): {float(TP_total)/float(TP_total+FN_total)*100:.2f}')
    print(f'Got {TN_total}/{TN_total+FP_total} with specifity: {float(TN_total)/float(FP_total+TN_total)*100:.2f}')
    call=float(TP_total)/float(TP_total+FN_total)*100
    spe=float(TN_total)/float(FP_total+TN_total)*100
    f1.append((2*call*spe)/(call+spe))
    print(f'Got F1 score: {(2*call*spe)/(call+spe):.2f}')
def get_logger(logger_name,log_file,level=logging.INFO):
	logger = logging.getLogger(logger_name)
	formatter = logging.Formatter('%(asctime)s-%(message)s')
	fileHandler = logging.FileHandler(log_file, mode='w')
	fileHandler.setFormatter(formatter)
	logger.setLevel(level)
	logger.addHandler(fileHandler)
	return logger

if __name__ == '__main__':
    # load data
    train_loader,test_loader = dataloader.load_data("/home/leosher/data/pet/fMRI_BOLD/par100/NC_eMCI/train","/home/leosher/data/pet/fMRI_BOLD/par100/NC_eMCI/test",batch_size=batch_size)
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
    for epoch in range(50):# one epoch means that the net has seen all the data in the dataset
        for batch_idx,(data,target) in enumerate(train_loader):
            net.train()
            data=data.to(device)
            target=target.to(device)
            # forward
            scores=net(data)
            loss=criterion(scores,target)
            print(loss)
            record_info='epoch: '+str(epoch)+" iteration: "+str(batch_idx)+" loss: "+str(loss.item())
            #log_loss.info(record_info)
            loss_average=loss_average+loss.item()
            # backward
            optimizer.zero_grad()
            loss.backward()
            '''
            for name, parms in net.named_parameters():	
                log_backward.info('epoch: '+str(epoch)+' batch index: '+str(batch_idx)+'-->name '+str(name)+' -->grad_requirs: '+str(parms.requires_grad)+\
		            ' -->grad_value: '+str(parms.grad))
            '''
            optimizer.step()
            
        loss_average=loss_average/(batch_idx+1)
        loss_all.append(loss_average)
        loss_average=0
        if epoch%2==0:
            print("*"*10)
            print("epoch:", epoch)
            check_accuracy(test_loader,net)
    plt.plot(sens)
    plt.savefig('/home/leosher/桌面/project/fMRI_AD_GCN/sens.jpg')
    plt.figure()
    plt.plot(acc)
    plt.savefig('/home/leosher/桌面/project/fMRI_AD_GCN/acc.jpg')
    plt.figure()
    plt.plot(specifity)
    plt.savefig('/home/leosher/桌面/project/fMRI_AD_GCN/specifity.jpg')
    plt.figure()
    plt.figure()      
    plt.plot(loss_all)
    plt.savefig('/home/leosher/桌面/project/fMRI_AD_GCN/loss.jpg')
    plt.figure()
    plt.plot(f1)
    plt.savefig('/home/leosher/桌面/project/fMRI_AD_GCN/f1.jpg')
    check_accuracy(train_loader,net)  