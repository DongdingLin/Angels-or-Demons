# NN部分
# Author：lindongding
# Time：2018年5月2日
# PS: 直接运行就GG

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
torch.manual_seed(2018)
warnings.filterwarnings("ignore")
torch.cuda.set_device(0)

data_all = pd.read_csv('./DataSet/Full_data_Number_Final.csv',header=None)
train_label = pd.read_csv('./DataSet/train_label.csv',header=None)
train_count = len(train_label)
train_data = data_all.iloc[:train_count, :]
test_data = data_all.iloc[train_count:, :]

# 参数设置
EPOCH=100
BATCH_SIZE=256
SHUFFLE=True
NUM_WORKERS=2
CUDA=True
Save_Model=False
FEATURE=X_train.shape[1]
LABEL_COUNT=2
TRAIN_SIZE=112405
TEST_SIZE=28101
POSITIVE_LABEL_SIZE=330
NEGATIVE_LABEL_SIZE=112075
LR=1e-3 # Learning Rate
POSITIVE_RATE=float(POSITIVE_LABEL_SIZE/TRAIN_SIZE)
NEGATIVE_RATE=float(NEGATIVE_LABEL_SIZE/TRAIN_SIZE)
KFold_K=5

# 模型设置
class MyNet(nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(MyNet, self).__init__()
        self.hidden1 = nn.Linear(n_feature, n_hidden)
        self.bn = nn.BatchNorm1d(num_features=n_hidden)
        self.hidden2 = nn.Linear(n_hidden, int(n_hidden/2))
        self.bn2 = nn.BatchNorm1d(num_features=int(n_hidden/2))
        self.dropout = nn.Dropout(0.8)
#         self.hidden3 = nn.Linear(n_hidden, int(n_hidden/2))
        self.output  = nn.Linear(int(n_hidden/2), n_output)
    
    def forward(self, x):
        x = x.cuda() if CUDA is True else x
        x = self.hidden1(x)
#         x = self.dropout(x)
        x = F.relu(x)
        x = self.bn(x)
        x = self.hidden2(x)
#         x = self.dropout(x)
        x = F.relu(x)
        x = self.bn2(x)
#         x = F.relu(self.hidden3(x))
        x = self.output(x)
        return x


# Cross Validation
X = train_data.values
y = train_label.replace(-1,0).values.squeeze()
skf = StratifiedKFold(n_splits=KFold_K, shuffle=True)
Round = 1
for train_index, test_index in skf.split(X, y):
    # generate train data and validate data
    X_train, X_vali = X[train_index], X[test_index]
    y_train, y_vali = y[train_index], y[test_index]
    
    # Neural Network
    net = MyNet(FEATURE, int(FEATURE*2), LABEL_COUNT)
    print (net)
    net = net.cuda() if CUDA is True else net
    weight = torch.Tensor([POSITIVE_RATE, NEGATIVE_RATE]).cuda() if CUDA is True else torch.Tensor([POSITIVE_RATE, NEGATIVE_RATE])
    loss_func = nn.CrossEntropyLoss(weight=weight)
	# optimizer = torch.optim.SGD(net.parameters(), lr=LR, momentum=0.8)
    optimizer = torch.optim.Adam(net.parameters(), lr=LR)
    scheduler = ReduceLROnPlateau(optimizer, 'min')
	# generate DataLoader Format data and loader
    d = torch.from_numpy(X_train)
    l = torch.from_numpy(y_train)
    torch_dataset = Data.TensorDataset(data_tensor=d, target_tensor=l)
    loader = Data.DataLoader(
        dataset=torch_dataset,  
        batch_size=BATCH_SIZE, 
        shuffle=SHUFFLE,
        num_workers=NUM_WORKERS,
    )
    
    # Training
    for epoch in range(EPOCH):
        begin_time = time.time()
        # print ('Epoch:', epoch+1, '   Training...')
        Count_Loss = 0
        for step, (batch_x, batch_y) in enumerate(loader):
            epoch_b_x = Variable(batch_x).cuda() if CUDA else Variable(batch_x)
            epoch_b_y = Variable(batch_y).cuda() if CUDA else Variable(batch_y)
            epoch_out = net(epoch_b_x.float())
            loss = loss_func(epoch_out, epoch_b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            Count_Loss += loss.cpu().data.numpy()[0]
            # if step%1000 == 0:
                # print ("iter:%d---time:%d  loss:%f"%(step,time.time()-begin_time, loss.cpu().data.numpy()[0]))
        if (epoch+1) % 1 == 0:
            print ("Epoch:%d is over, use %.2fs, loss: %f"%(epoch+1,(time.time()-begin_time), Count_Loss))
            pre_dict = net(Variable(torch.from_numpy(X_train)).cuda().float()).data
            vali_auc = metrics.roc_auc_score(y_train,pre_dict[:,1])
            scheduler.step(vali_auc)
            print ("Train_AUC score is %.8f"%(vali_auc))
            pre_dict = net(Variable(torch.from_numpy(X_vali)).float()).data
            vali_auc = metrics.roc_auc_score(y_vali,pre_dict[:,1])
#             scheduler.step(vali_auc)
            print ("Vali_AUC  score is %.8f"%(vali_auc))
            
    # Validating
    pre_dict = net(Variable(torch.from_numpy(X_vali)).float()).data
    vali_auc = metrics.roc_auc_score(y_vali,pre_dict[:,1])
    print ("The Round %d is Over. AUC score is %.8f\n"%(Round, vali_auc))
    
    # Save the Model Parameters
    if Save_Model:
        model_file_name = "FC_Round_"+str(Round)+".pkl"
        torch.save(net.state_dict(), model_file_name)
    Round += 1    
    

# Result
test = Variable(torch.from_numpy(test_data.values))
outputs = net(test.float())
_, predicted = torch.max(outputs.data, 1)
predicted_list = predicted.cpu().numpy().tolist()
print ("-1 rate: %.5f\n1  count: %.5f\n"%(float(predicted_list.count(0))/len(predicted_list), float(predicted_list.count(1))/len(predicted_list)))
print ("Train_label distribution:\n-1 rate: %.5f\n1  rate: %.5f\n"%(NEGATIVE_RATE,POSITIVE_RATE))


# Save to CSV File
pred_pandas = pd.DataFrame(data=predicted.cpu().numpy()).replace(0,-1)
index = pd.DataFrame(data=np.arange(1, 28102))
pred_pandas = pd.concat([index, pred_pandas], axis=1)
pred_pandas.to_csv('./NN_Predict.csv', header=['Id', 'label'], index=False)