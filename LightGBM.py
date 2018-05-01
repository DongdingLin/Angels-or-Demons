# LightGBM部分
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
import pickle
from collections import Counter
import lightgbm as lgb
from lightgbm import LGBMClassifier
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

# 构建数据集
list_t = [                                        #单一特征结果
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

# 搜索最佳树数
def modelfit(alg, data, target, cv_folds=5, early_stopping_rounds=150):
    lgb_param = alg.get_params()
    lgtrain = lgb.Dataset(data=data, label=target)
    cvresult = lgb.cv(lgb_param, lgtrain, num_boost_round=alg.get_params()['n_estimators'],
                      nfold=5,stratified=True,shuffle=True,metrics='auc',
                      seed=1377,
                      init_model=None,
                      early_stopping_rounds=early_stopping_rounds, show_stdv=True, verbose_eval=True)


# 调参部分
param_dict = {"device":"cpu",
              "threads":"12",
#               "gpu_device_id":1,
#               'scale_pos_weight':y.tolist().count(0)/y.tolist().count(1)
             }
lgb_para = LGBMClassifier(
        num_leaves=30, #30
        boosting_type='gbdt',
        max_depth=-1,
        learning_rate=0.01,
        n_estimators=2000,
        subsample_for_bin=8000,
        objective='binary',
        min_split_gain=0.,
        min_child_weight=1e-3,
        min_child_samples=40,
        subsample=.6,
        subsample_freq=1,
        colsample_bytree=.8,
        reg_alpha=0.15,
        reg_lambda=0.,
        random_state=1377,
        n_jobs=12,
        silent=False,
#         class_weight='is_unbalanced',
        **param_dict)
modelfit(lgb_para, X, y)

# 最终结果
lgb_final = LGBMClassifier(
        num_leaves=30, #30
        boosting_type='gbdt',
        max_depth=-1,
        learning_rate=0.01,
        n_estimators=1000,
        subsample_for_bin=10000,
        objective='binary',
        min_split_gain=0.,
        min_child_weight=1e-3,
        min_child_samples=40,
        subsample=.5, # baozha
        subsample_freq=1,
        colsample_bytree=.8,
        reg_alpha=0.2,
        reg_lambda=0.,
#         random_state=1377,
        n_jobs=12,
        silent=False,
#         class_weight='is_unbalanced',
        **param_dict)
lgb_final.fit(X=X, y=y, eval_metric='auc',verbose=True)
pred = lgb2.predict_proba(X=test)

index = pd.DataFrame(data=np.arange(1, 28102))
pred_pandas=pd.DataFrame(data=pred[:,1])
pred_pandas = pd.concat([index, pred_pandas], axis=1)
pred_pandas.to_csv('./LGBM_Predict.csv', header=['Id', 'label'], index=False)