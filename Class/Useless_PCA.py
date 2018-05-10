# PCA部分
# Author：lindongding
# Time：2018年5月2日

import pandas as pd
import numpy as np
import sys
import torch
import torch.utils.data as Data
import torch.nn as nn
import torch.nn.functional as F
import warnings
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.decomposition import PCA
torch.manual_seed(1)
warnings.filterwarnings("ignore")
torch.cuda.set_device(1)

train_data = pd.read_csv('./DataSet/data_all_6MaxMinScale.csv',header=None)
t = train_data.values
pca = PCA(n_components=0.99999)
p = pca.fit_transform(t)
# print (p.explained_variance_ratio_)
# print(p.components_)
pre_pandas = pd.DataFrame(data=p)
train_label = pd.read_csv('./DataSet/train_label.csv',header=None)
train_count = len(train_label)
X = pre_pandas.values[:train_count, :]
y = train_label.replace(-1,0).values.squeeze()
test_X  = pre_pandas.values[train_count:, :]