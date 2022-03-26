# dataloader: load the data from the matlab file

import os

from sklearn.metrics import precision_recall_curve


def delete(folder_path,keyword):
    for filename in (os.listdir(folder_path)):
        if keyword in filename:
            os.remove(folder_path+filename)
def record_subjects(folder_path):
    name_list=[]
    for filename in (os.listdir(folder_path)):
        index1=filename.find('S_')
        index2=filename.find('.mat')
        name_list.append(filename[index1+2:index2-10])
    return list(set(name_list))

if __name__ == '__main__':
    delete("/home/leosher/桌面/eMCI/",'DFC')