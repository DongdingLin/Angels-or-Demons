# XGBoost部分
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
from sklearn import metrics
from sklearn.datasets import dump_svmlight_file
from xgboost import plot_importance
from matplotlib import pyplot
from sklearn import metrics
from xgboost import XGBClassifier
import xgboost as xgb
import pickle
warnings.filterwarnings("ignore")

# 添加数据
data_all = pd.read_csv('./DataSet2/data_all_After_90%off_10%mean_ym_numeric.csv',header=None)
YM = pd.read_csv('./DataSet2/data_all_After_90%off_10%mean_ym_string_DateYM.csv',header=None)
col0_3 = pd.read_csv('./DataSet2/data_all_After_90%off_10%mean_ym_string_Col0-3.csv', header=None)
col4half = pd.read_csv('./DataSet2/data_all_After_90%off_10%mean_ym_string_Col4half.csv', header=None)
col4half2 = pd.read_csv('./DataSet2/data_all_After_90%off_10%mean_ym_string_Col4half2.csv', header=None)
col5half = pd.read_csv('./DataSet2/data_all_After_90%off_10%mean_ym_string_Col5half.csv', header=None)
col5half2 = pd.read_csv('./DataSet2/data_all_After_90%off_10%mean_ym_string_Col5half2.csv', header=None)
col6 = pd.read_csv('./DataSet2/data_all_After_90%off_10%mean_ym_string_Col6.csv', header=None)
col7 = pd.read_csv('./DataSet2/data_all_After_90%off_10%mean_ym_string_Col7.csv', header=None)
col8half = pd.read_csv('./DataSet2/data_all_After_90%off_10%mean_ym_string_Col8half.csv', header=None)
col8half2 = pd.read_csv('./DataSet2/data_all_After_90%off_10%mean_ym_string_Col8half2.csv', header=None)
col9half = pd.read_csv('./DataSet2/data_all_After_90%off_10%mean_ym_string_Col9half.csv', header=None)
col9half2 = pd.read_csv('./DataSet2/data_all_After_90%off_10%mean_ym_string_Col9half2.csv', header=None)
col10half = pd.read_csv('./DataSet2/data_all_After_90%off_10%mean_ym_string_Col10half.csv', header=None)
col10half2 = pd.read_csv('./DataSet2/data_all_After_90%off_10%mean_ym_string_Col10half2.csv', header=None)
col11half = pd.read_csv('./DataSet2/data_all_After_90%off_10%mean_ym_string_Col11half.csv', header=None)
col11half2 = pd.read_csv('./DataSet2/data_all_After_90%off_10%mean_ym_string_Col11half2.csv', header=None)
col12half = pd.read_csv('./DataSet2/data_all_After_90%off_10%mean_ym_string_Col12half.csv', header=None)
col12half2 = pd.read_csv('./DataSet2/data_all_After_90%off_10%mean_ym_string_Col12half2.csv', header=None)
col13half = pd.read_csv('./DataSet2/data_all_After_90%off_10%mean_ym_string_Col13half.csv', header=None)
col13half2 = pd.read_csv('./DataSet2/data_all_After_90%off_10%mean_ym_string_Col13half2.csv', header=None)
col14half = pd.read_csv('./DataSet2/data_all_After_90%off_10%mean_ym_string_Col14half.csv', header=None)
col14half2 = pd.read_csv('./DataSet2/data_all_After_90%off_10%mean_ym_string_Col14half2.csv', header=None)
col15half = pd.read_csv('./DataSet2/data_all_After_90%off_10%mean_ym_string_Col15half.csv', header=None)
col15half2 = pd.read_csv('./DataSet2/data_all_After_90%off_10%mean_ym_string_Col15half2.csv', header=None)
col16half = pd.read_csv('./DataSet2/data_all_After_90%off_10%mean_ym_string_Col16half.csv', header=None)
col16half2 = pd.read_csv('./DataSet2/data_all_After_90%off_10%mean_ym_string_Col16half2.csv', header=None)
col17half = pd.read_csv('./DataSet2/data_all_After_90%off_10%mean_ym_string_Col17half.csv', header=None)
col17half2 = pd.read_csv('./DataSet2/data_all_After_90%off_10%mean_ym_string_Col17half2.csv', header=None)
col18half = pd.read_csv('./DataSet2/data_all_After_90%off_10%mean_ym_string_Col18half.csv', header=None)
col18half2 = pd.read_csv('./DataSet2/data_all_After_90%off_10%mean_ym_string_Col18half2.csv', header=None)
col19half = pd.read_csv('./DataSet2/data_all_After_90%off_10%mean_ym_string_Col19half.csv', header=None)
col19half2 = pd.read_csv('./DataSet2/data_all_After_90%off_10%mean_ym_string_Col19half2.csv', header=None)
col20half = pd.read_csv('./DataSet2/data_all_After_90%off_10%mean_ym_string_Col20half.csv', header=None)
col20half2 = pd.read_csv('./DataSet2/data_all_After_90%off_10%mean_ym_string_Col20half2.csv', header=None)
col21half = pd.read_csv('./DataSet2/data_all_After_90%off_10%mean_ym_string_Col21half.csv', header=None)
col21half2 = pd.read_csv('./DataSet2/data_all_After_90%off_10%mean_ym_string_Col21half2.csv', header=None)
col22half = pd.read_csv('./DataSet2/data_all_After_90%off_10%mean_ym_string_Col22half.csv', header=None)
col22half2 = pd.read_csv('./DataSet2/data_all_After_90%off_10%mean_ym_string_Col22half2.csv', header=None)
col23half = pd.read_csv('./DataSet2/data_all_After_90%off_10%mean_ym_string_Col23half.csv', header=None)
col23half2 = pd.read_csv('./DataSet2/data_all_After_90%off_10%mean_ym_string_Col23half2.csv', header=None)
col25qu1 = pd.read_csv('./DataSet2/data_all_After_90%off_10%mean_ym_string_Col25quarter1.csv', header=None)
col25qu2 = pd.read_csv('./DataSet2/data_all_After_90%off_10%mean_ym_string_Col25quarter2.csv', header=None)
col25qu3 = pd.read_csv('./DataSet2/data_all_After_90%off_10%mean_ym_string_Col25quarter3.csv', header=None)
col25qu4 = pd.read_csv('./DataSet2/data_all_After_90%off_10%mean_ym_string_Col25quarter4.csv', header=None)
col26qu1 = pd.read_csv('./DataSet2/data_all_After_90%off_10%mean_ym_string_Col26quarter1.csv', header=None)
col26qu2 = pd.read_csv('./DataSet2/data_all_After_90%off_10%mean_ym_string_Col26quarter2.csv', header=None)
col26qu3 = pd.read_csv('./DataSet2/data_all_After_90%off_10%mean_ym_string_Col26quarter3.csv', header=None)
col26qu4 = pd.read_csv('./DataSet2/data_all_After_90%off_10%mean_ym_string_Col26quarter4.csv', header=None)
factorize = pd.read_csv('./DataSet2/data_all_After_90%off_10%mean_ym_string_Factorize.csv', header=None)
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
X = train_data.values
y = train_label.replace(-1,0).values.squeeze()
test = test_data.values

# CV调树数
def modelfit(alg, data, target, cv_folds=5, early_stopping_rounds=50):
    xgb_param = alg.get_xgb_params()
    xgtrain = xgb.DMatrix(data=data, label=target)
    skf_cv = StratifiedKFold(n_splits=cv_folds, shuffle=True)
    cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'],folds=skf_cv,seed=1377,
        metrics='auc',early_stopping_rounds=early_stopping_rounds, show_stdv=True, verbose_eval=True)
    alg.set_params(n_estimators=cvresult.shape[0])
    print (cvresult.shape[0])

# XGBoost调参
param_dict = {'tree_method':'gpu_hist',
             'gpu_id':1,
             }
xgb1 = XGBClassifier(
        learning_rate=0.1,
        n_estimators=500,
        max_depth=6,
        min_child_weight=1,
        gamma=0,
        subsample=1,
        colsample_bytree=1,
        objective= 'binary:logistic',
        nthread=10,
        scale_pos_weight=1,
        seed=1377,
        **param_dict)
modelfit(xgb1, X, y)

# CVSearch
X = train_data.values
y = train_label.replace(-1,0).values.squeeze()
param_test1 = {
    'max_depth':range(3,6,1),
    'min_child_weight':range(1,6,1),
    'subsample':[i/10.0 for i in range(6,10,2)],
    'colsample_bytree':[i/10.0 for i in range(6,10)],
    'reg_alpha':[0, 0.001, 0.005, 0.01, 0.05],
    'max_depth':[3],
}
print (param_test1)
skf = StratifiedKFold(n_splits=7, shuffle=True)
param_dict = {'tree_method':'gpu_hist','silent':0}

gsearch1 = GridSearchCV(param_grid=param_test1,scoring='roc_auc',n_jobs=4,
    iid=False,cv=skf,
    estimator=XGBClassifier(learnaing_rate =0.2,n_estimators=200,max_depth=5,n_jobs=4,
    min_child_weight=1,gamma=0.5,objective='binary:logistic',nthread=1,seed=27,**param_dict))
begin = time.time()
gsearch1.fit(X, y)
print ("use time:%ds"%(time.time()-begin))
print (gsearch1.grid_scores_)
print (gsearch1.best_params_)
print (gsearch1.best_score_)


# Predict
X = train_data.values
y = train_label.replace(-1,0).values.squeeze()
X_test = test_data.values

param_dict = {'tree_method':'gpu_hist','silent':0,'gpu_id':1, 
                  'max_depth':9,'gamma':0,
                  'min_child_weight':1,
                 'subsample':1,'colsample_bytree':1,
                 'scale_pos_weight':1}
model = XGBClassifier(n_estimators=190,
                    nthread=10,seed=27,
                    objective='binary:logistic',
                    n_jobs=6,
                    learning_rate=0.1,
                    **param_dict)
eval_set=[(X, y)]
eval_metric='auc'
model.fit(X_train, y_train,
          eval_set=eval_set, 
          eval_metric=eval_metric)
pickle.dump(model, open("Final_1.xgb", "wb"))
test_predict = model.predict_proba(X_test)[:,1]
pred_pandas = pd.DataFrame(data=test_predict).replace(0,-1)
index = pd.DataFrame(data=np.arange(1, 28102))
pred_pandas = pd.concat([index, pred_pandas], axis=1)
pred_pandas.to_csv('./Xgboost_Predict.csv', header=['Id', 'label'], index=False)