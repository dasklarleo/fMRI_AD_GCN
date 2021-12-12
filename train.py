'''
In this file the hyper-paramters will be set
The training loops would also be set
'''
import dataloader
import model
if __name__ == '__main__':
    pass
    dataloader("/home/leosher/data/pet/fMRI_BOLD/par100", 16)
    net = model.whole_network(batch_size=16, ROI_nums=130, heads=8, BOLD_nums=1000, q=128, SE_parameter=4,
                              feature_nums=1000, dilated_parameter_k=2)
    
