# Stacking部分
# Author：lindongding
# Time：2018年5月2日
# PS: 直接运行就GG
import pandas as pd
import numpy as np
import sys
import warnings
import time
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier,BaggingClassifier,ExtraTreesClassifier,GradientBoostingClassifier
from sklearn import metrics
from sklearn import svm
from sklearn.datasets import dump_svmlight_file
from xgboost import plot_importance
from matplotlib import pyplot
from sklearn import metrics
import pickle
from collections import Counter
import lightgbm as lgb
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMRegressor
warnings.filterwarnings("ignore")

# 将特征工程得到的所有特征加进来
data_all = pd.read_csv('./DataSet/data_all_After_90%off_10%mean_ym_numeric.csv',header=None)
YM = pd.read_csv('./DataSet/data_all_After_90%off_10%mean_ym_string_DateYM.csv',header=None)
col0_3 = pd.read_csv('./DataSet/data_all_After_90%off_10%mean_ym_string_Col0-3.csv', header=None)
col4half = pd.read_csv('./DataSet/data_all_After_90%off_10%mean_ym_string_Col4half.csv', header=None)
col4half2 = pd.read_csv('./DataSet/data_all_After_90%off_10%mean_ym_string_Col4half2.csv', header=None)
col5half = pd.read_csv('./DataSet/data_all_After_90%off_10%mean_ym_string_Col5half.csv', header=None)
col5half2 = pd.read_csv('./DataSet/data_all_After_90%off_10%mean_ym_string_Col5half2.csv', header=None)
col6 = pd.read_csv('./DataSet/data_all_After_90%off_10%mean_ym_string_Col6.csv', header=None)
col7 = pd.read_csv('./DataSet/data_all_After_90%off_10%mean_ym_string_Col7.csv', header=None)
col8half = pd.read_csv('./DataSet/data_all_After_90%off_10%mean_ym_string_Col8half.csv', header=None)
col8half2 = pd.read_csv('./DataSet/data_all_After_90%off_10%mean_ym_string_Col8half2.csv', header=None)
col9half = pd.read_csv('./DataSet/data_all_After_90%off_10%mean_ym_string_Col9half.csv', header=None)
col9half2 = pd.read_csv('./DataSet/data_all_After_90%off_10%mean_ym_string_Col9half2.csv', header=None)
col10half = pd.read_csv('./DataSet/data_all_After_90%off_10%mean_ym_string_Col10half.csv', header=None)
col10half2 = pd.read_csv('./DataSet/data_all_After_90%off_10%mean_ym_string_Col10half2.csv', header=None)
col11half = pd.read_csv('./DataSet/data_all_After_90%off_10%mean_ym_string_Col11half.csv', header=None)
col11half2 = pd.read_csv('./DataSet/data_all_After_90%off_10%mean_ym_string_Col11half2.csv', header=None)
col12half = pd.read_csv('./DataSet/data_all_After_90%off_10%mean_ym_string_Col12half.csv', header=None)
col12half2 = pd.read_csv('./DataSet/data_all_After_90%off_10%mean_ym_string_Col12half2.csv', header=None)
col13half = pd.read_csv('./DataSet/data_all_After_90%off_10%mean_ym_string_Col13half.csv', header=None)
col13half2 = pd.read_csv('./DataSet/data_all_After_90%off_10%mean_ym_string_Col13half2.csv', header=None)
col14half = pd.read_csv('./DataSet/data_all_After_90%off_10%mean_ym_string_Col14half.csv', header=None)
col14half2 = pd.read_csv('./DataSet/data_all_After_90%off_10%mean_ym_string_Col14half2.csv', header=None)
col15half = pd.read_csv('./DataSet/data_all_After_90%off_10%mean_ym_string_Col15half.csv', header=None)
col15half2 = pd.read_csv('./DataSet/data_all_After_90%off_10%mean_ym_string_Col15half2.csv', header=None)
col16half = pd.read_csv('./DataSet/data_all_After_90%off_10%mean_ym_string_Col16half.csv', header=None)
col16half2 = pd.read_csv('./DataSet/data_all_After_90%off_10%mean_ym_string_Col16half2.csv', header=None)
col17half = pd.read_csv('./DataSet/data_all_After_90%off_10%mean_ym_string_Col17half.csv', header=None)
col17half2 = pd.read_csv('./DataSet/data_all_After_90%off_10%mean_ym_string_Col17half2.csv', header=None)
col18half = pd.read_csv('./DataSet/data_all_After_90%off_10%mean_ym_string_Col18half.csv', header=None)
col18half2 = pd.read_csv('./DataSet/data_all_After_90%off_10%mean_ym_string_Col18half2.csv', header=None)
col19half = pd.read_csv('./DataSet/data_all_After_90%off_10%mean_ym_string_Col19half.csv', header=None)
col19half2 = pd.read_csv('./DataSet/data_all_After_90%off_10%mean_ym_string_Col19half2.csv', header=None)
col20half = pd.read_csv('./DataSet/data_all_After_90%off_10%mean_ym_string_Col20half.csv', header=None)
col20half2 = pd.read_csv('./DataSet/data_all_After_90%off_10%mean_ym_string_Col20half2.csv', header=None)
col21half = pd.read_csv('./DataSet/data_all_After_90%off_10%mean_ym_string_Col21half.csv', header=None)
col21half2 = pd.read_csv('./DataSet/data_all_After_90%off_10%mean_ym_string_Col21half2.csv', header=None)
col22half = pd.read_csv('./DataSet/data_all_After_90%off_10%mean_ym_string_Col22half.csv', header=None)
col22half2 = pd.read_csv('./DataSet/data_all_After_90%off_10%mean_ym_string_Col22half2.csv', header=None)
col23half = pd.read_csv('./DataSet/data_all_After_90%off_10%mean_ym_string_Col23half.csv', header=None)
col23half2 = pd.read_csv('./DataSet/data_all_After_90%off_10%mean_ym_string_Col23half2.csv', header=None)
col25qu1 = pd.read_csv('./DataSet/data_all_After_90%off_10%mean_ym_string_Col25quarter1.csv', header=None)
col25qu2 = pd.read_csv('./DataSet/data_all_After_90%off_10%mean_ym_string_Col25quarter2.csv', header=None)
col25qu3 = pd.read_csv('./DataSet/data_all_After_90%off_10%mean_ym_string_Col25quarter3.csv', header=None)
col25qu4 = pd.read_csv('./DataSet/data_all_After_90%off_10%mean_ym_string_Col25quarter4.csv', header=None)
col26qu1 = pd.read_csv('./DataSet/data_all_After_90%off_10%mean_ym_string_Col26quarter1.csv', header=None)
col26qu2 = pd.read_csv('./DataSet/data_all_After_90%off_10%mean_ym_string_Col26quarter2.csv', header=None)
col26qu3 = pd.read_csv('./DataSet/data_all_After_90%off_10%mean_ym_string_Col26quarter3.csv', header=None)
col26qu4 = pd.read_csv('./DataSet/data_all_After_90%off_10%mean_ym_string_Col26quarter4.csv', header=None)
factorize = pd.read_csv('./DataSet/data_all_After_90%off_10%mean_ym_string_Factorize.csv', header=None)
train_label = pd.read_csv('./DataSet/Last_Train_Label.csv',header=None)

#生成数据集
list_t = [
    data_all,                                     #61.6
          YM,                                     #61.6
          col0_3,                                 #57.8
          col4half, col4half2,                    #59
          col5half,col5half2,                     #58
          col6,                                   #57.6
          col7,                                   #57.6
          col8half,col8half2,                     #55
          col9half,col9half2,                     #55
          col10half,col10half2,                   #55.5
          col11half,col11half2,                   #55.5
          col12half, col12half2,                  #55.8
          col13half, col13half2,                  #55.8
          col14half, col14half2,                  #56.7
          col15half, col15half2,                  #56.7
          col16half, col16half2,                  #50.5
          col17half, col17half2,                  #50.5
          col18half, col18half2,                  #51.8
          col19half, col19half2,                  #51.2
          col20half, col20half2,                  #52
          col21half, col21half2,                  #50.5
          col22half, col22half2,                  #47.2
          col23half, col23half2,                  #46.6
          col25qu1,col25qu2,col25qu3,col25qu4,    #76.5
          col26qu1,col26qu2,col26qu3,col26qu4,    #64
          factorize,                              #85
         ]
data_all = pd.concat(list_t,axis=1)
data_all.columns = range(len(data_all.columns))
train_count = len(train_label)
train_data = data_all.iloc[:train_count, :]
test_data = data_all.iloc[train_count:, :]
X_set = train_data.values
y_set = train_label.replace(-1,0).values.squeeze()
test = test_data.values

# Stacking第一层
def model_fit_pred(model, train_set, train_label, vali_set, vali_label, test_set):
    if model == 'RF':
        RF_model = RandomForestClassifier(bootstrap=True,n_jobs=6,max_depth=12,n_estimators=170)
        RF_model.fit(train_set, train_label)
        if RF_model.predict_proba(vali_set).shape[1] == 2:
            train_pred = RF_model.predict_proba(vali_set)[:,1]
        else:
            train_pred = RF_model.predict_proba(vali_set)[:,0]
            
        if RF_model.predict_proba(test_set).shape[1] == 2:
            test_pred = RF_model.predict_proba(test_set)[:,1]
        else:
            test_pred = RF_model.predict_proba(test_set)[:,0]
    elif model == 'XGBoost1':
        param_dict = {'tree_method':'gpu_hist'}
        xgb_model = XGBClassifier(
            learning_rate=0.1,n_estimators=300,max_depth=10,min_child_weight=1,gamma=0,
            subsample=1,colsample_bytree=1,objective= 'binary:logistic',nthread=10,
            scale_pos_weight=1,**param_dict)
        xgb_model.fit(train_set, train_label)
        train_pred = xgb_model.predict_proba(vali_set)[:,1]
        test_pred = xgb_model.predict_proba(test_set)[:,1]
    elif model == 'XGBoost2':
        param_dict = {'tree_method':'gpu_hist'}
        xgb_model = XGBClassifier(
            learning_rate=0.1,n_estimators=200,max_depth=9,min_child_weight=1,gamma=0,
            subsample=1,colsample_bytree=1,objective= 'binary:logistic',nthread=10,
            scale_pos_weight=1,**param_dict)
        xgb_model.fit(train_set, train_label)
        train_pred = xgb_model.predict_proba(vali_set)[:,1]
        test_pred = xgb_model.predict_proba(test_set)[:,1]
    elif model == 'XGBoost3':
        param_dict = {'tree_method':'gpu_hist'}
        xgb_model = XGBClassifier(
            learning_rate=0.1,n_estimators=200,max_depth=8,min_child_weight=1,gamma=0,
            subsample=1,colsample_bytree=1,objective= 'binary:logistic',nthread=10,
            scale_pos_weight=1,**param_dict)
        xgb_model.fit(train_set, train_label)
        train_pred = xgb_model.predict_proba(vali_set)[:,1]
        test_pred = xgb_model.predict_proba(test_set)[:,1]
    elif model == 'XGBoost4':
        param_dict = {'tree_method':'gpu_hist'}
        xgb_model = XGBClassifier(
            learning_rate=0.1,n_estimators=200,max_depth=11,min_child_weight=1,gamma=0,
            subsample=1,colsample_bytree=1,objective= 'binary:logistic',nthread=10,
            scale_pos_weight=1,**param_dict)
        xgb_model.fit(train_set, train_label)
        train_pred = xgb_model.predict_proba(vali_set)[:,1]
        test_pred = xgb_model.predict_proba(test_set)[:,1]
    elif model == 'XGBoost5':
        param_dict = {'tree_method':'gpu_hist'}
        xgb_model = XGBClassifier(
            learning_rate=0.1,n_estimators=200,max_depth=12,min_child_weight=1,gamma=0,
            subsample=1,colsample_bytree=1,objective= 'binary:logistic',nthread=10,
            scale_pos_weight=1,**param_dict)
        xgb_model.fit(train_set, train_label)
        train_pred = xgb_model.predict_proba(vali_set)[:,1]
        test_pred = xgb_model.predict_proba(test_set)[:,1]
    elif model == 'LGBM':
        lgb_model = LGBMClassifier(
            num_leaves=30,boosting_type='gbdt',max_depth=-1,learning_rate=0.01,n_estimators=1200,
            subsample_for_bin=10000,objective='binary',min_split_gain=0.,min_child_weight=1e-3,
            min_child_samples=40,subsample=.5,subsample_freq=1,
            colsample_bytree=1.,reg_alpha=0.,reg_lambda=0.,n_jobs=12,silent=True)
        lgb_model.fit(train_set, train_label)
        train_pred = lgb_model.predict_proba(vali_set)[:,1]
        test_pred = lgb_model.predict_proba(test_set)[:,1]
    return train_pred, test_pred

# stacking第二层
def stacking(model_list, X, y, test_set):
    # First level learning model
    # Create new data
    new_train_matrix = list()
    new_test_matrix  = list()
    for model in model_list:
        skf = StratifiedKFold(n_splits=5,shuffle=True, random_state=1377)
        new_train_col = np.zeros(len(y))
        new_test_col = np.zeros(len(test_set))
        for train_index, test_index in skf.split(X, y):
            X_train, X_vali = X[train_index], X[test_index]
            y_train, y_vali = y[train_index], y[test_index]
            print (model)
            train_pred, test_pred = model_fit_pred(model, X_train, y_train, X_vali, y_vali, test_set)
            new_train_col[test_index] = train_pred
            new_test_col = new_test_col+test_pred
        new_test_col = new_test_col/5
        new_train_matrix.append(new_train_col)
        new_test_matrix.append(new_test_col)
    # Second level learning model
    new_train_matrix = np.array(new_train_matrix)
    new_test_matrix = np.array(new_test_matrix)
    new_train_matrix,new_test_matrix = new_train_matrix.T,new_test_matrix.T
    param_dict = {
        'tree_method':'gpu_hist'
    }
    layer2_model = XGBClassifier(silent=False,
            learning_rate=0.1,n_estimators=100,max_depth=3,min_child_weight=1,gamma=0,
            subsample=1,colsample_bytree=1,objective= 'binary:logistic',nthread=10,
            scale_pos_weight=1,seed=1337,**param_dict)
#     layer2_model = LGBMClassifier(
#             num_leaves=31,boosting_type='gbdt',max_depth=3,learning_rate=0.1,n_estimators=150,
#             subsample_for_bin=50000,objective='binary',min_split_gain=0.,min_child_weight=1e-3,
#             min_child_samples=20,subsample=1.,subsample_freq=1,
#             colsample_bytree=1.,reg_alpha=0.,reg_lambda=0.,n_jobs=12,silent=True)
    layer2_model.fit(new_train_matrix, y)
    pred_result = layer2_model.predict_proba(new_test_matrix)[:,1]
    return pred_result

# stacking交叉验证
skf = StratifiedKFold(n_splits=5,shuffle=True)
X_set = pd.DataFrame(X_set).fillna('-1').values
test = pd.DataFrame(test).fillna('-1').values
for train_index, test_index in skf.split(X_set, y_set):
    X_train, X_vali = X_set[train_index], X_set[test_index]
    y_train, y_vali = y_set[train_index], y_set[test_index]
    model_l = [
    'RF', 
    'XGBoost1', 
    'XGBoost2', 
    'XGBoost3', 
    'LGBM',
    ]
    resu = stacking(model_l,X_train,y_train,X_vali)
    vali_auc = metrics.roc_auc_score(y_vali,resu)
    print ("vali_auc:"+str(vali_auc))

# Stacking最终结果
X_set = pd.DataFrame(X_set).fillna('-1').values
test = pd.DataFrame(test).fillna('-1').values
model_l = [
    'RF', 
    'XGBoost1', 
    'XGBoost2', 
    'XGBoost3', 
    'LGBM',
    ]
resu = stacking(model_l,X_set,y_set,test)
index = pd.DataFrame(data=np.arange(1, 28102))
pred_pandas=pd.DataFrame(data=resu)
pred_pandas = pd.concat([index, pred_pandas], axis=1)
pred_pandas.to_csv('./Stacking_3.csv', header=['Id', 'label'], index=False)


# 其他stacking 的基模型测试
rf_X = pd.DataFrame(X).fillna('-1').values
skf = StratifiedKFold(n_splits=5,shuffle=True, random_state=1377)
RF_model = RandomForestClassifier(bootstrap=True,n_jobs=6,max_depth=12, random_state=1377,n_estimators=200)
Ada_model = AdaBoostClassifier(random_state=1377,n_estimators=200)
Bag_model = BaggingClassifier(n_estimators=200,random_state=1377,n_jobs=6)
Extra_model = ExtraTreesClassifier(n_estimators=200,bootstrap=True,random_state=1377,n_jobs=6)
Grad_model  = GradientBoostingClassifier(n_estimators=300, random_state=1377)
OCSVM_model = svm.OneClassSVM(random_state=1377)
SVC_model = svm.SVC(random_state=1377)
SVR_model = svm.SVR()
list_model = [
   'RF','Ada','Extra','Grad','OCSVM','SVC','SVR'
]
for m in list_model:
    for train_index, test_index in skf.split(rf_X, y):
        X_train, X_vali = rf_X[train_index], rf_X[test_index]
        y_train, y_vali = y[train_index], y[test_index]
        if m == 'RF':
            RF_model.fit(X_train, y_train)
            y_vali_pred = RF_model.predict_proba(X_vali)[:,1]
        elif m == 'Ada':
            Ada_model.fit(X_train, y_train)
            y_vali_pred = Ada_model.predict_proba(X_vali)[:,1]
        elif m == 'Bag':
            Bag_model.fit(X_train, y_train)
            y_vali_pred = Bag_model.predict_proba(X_vali)[:,1]
        elif m == 'Extra':
            Extra_model.fit(X_train, y_train)
            y_vali_pred = Extra_model.predict_proba(X_vali)[:,1]
        elif m == 'Grad':
            Grad_model.fit(X_train, y_train)
            y_vali_pred = Grad_model.predict_proba(X_vali)[:,1]
        elif m == 'OCSVM':
            OCSVM_model.fit(X_train, y_train)
            y_vali_pred = OCSVM_model.predict(X_vali)
        elif m == 'SVC':
            SVC_model.fit(X_train, y_train)
            y_vali_pred = SVC_model.predict(X_vali)
        elif m == 'SVR':
            SVR_model.fit(X_train, y_train)
            y_vali_pred = SVR_model.predict(X_vali)
        vali_auc = metrics.roc_auc_score(y_vali,y_vali_pred)
        print ("model: "+str(m)+"\t"+"vali_auc:"+str(vali_auc))