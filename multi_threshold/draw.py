from os import access

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import seaborn as sns
import numpy as np
import seaborn as sns
import scipy.io as scio
import torch.nn.functional as F


ROI_BOLDs = scio.loadmat("/home/leosher/data/pet/fMRI_BOLD/par100/NC_eMCI/test/EMCI_130_S_4417_201302_05.mat")['ROI_ts'].T
ROI_BOLDs=torch.from_numpy(ROI_BOLDs)

correlation=torch.corrcoef(ROI_BOLDs)

sns.set()

sns.heatmap(correlation,cmap="hot")
plt.show()

sns.set()
correlation_2=F.threshold(correlation,0.105,0)
sns.heatmap(correlation_2,cmap="hot")
plt.show()

sns.set()
correlation_1=F.threshold(correlation,0.274,0)
sns.heatmap(correlation_1,cmap="hot")
plt.show()


sns.set()
correlation_3=F.threshold(correlation,0.425,0)
sns.heatmap(correlation_3,cmap="hot")
plt.show()
sns.set()
correlation_4=F.threshold(correlation,0.637,0)
sns.heatmap(correlation_4,cmap="hot")
plt.show()