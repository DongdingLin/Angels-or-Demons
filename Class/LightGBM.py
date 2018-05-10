# Class for LightGBM
# Author：lindongding
# Time：May.9.2018
import pandas as pd
import numpy as np
import sys,warnings,time,os
from sklearn.model_selection import StratifiedKFold,KFold,GridSearchCV
from sklearn import metrics
from xgboost import plot_importance
from matplotlib import pyplot
import lightgbm as lgb
from lightgbm import LGBMClassifier,LGBMRegressor
warnings.filterwarnings("ignore")

class LightGBM:
	def __init__(self, data_path):
		self.data_path = os.path.abspath(data_path)
		self.train_data = None
		self.test_data = None
		self.label = None
		self.__LoadData()
		self.lgbm_trained = None

	def __LoadData(self):
		feather_path_list = list([None for i in range(52)])
		feather_path_list[0] = self.data_path+"/SpiltToNumeric/numeric_data.csv"
		feather_path_list[1] = self.data_path+"/DealAllString/time_0_factorize.csv"
		feather_path_list[2] = self.data_path+"/DealAllString/time_1_factorize.csv"
		feather_path_list[3] = self.data_path+"/DealAllString/time_2_factorize.csv"
		feather_path_list[4] = self.data_path+"/DealAllString/Col0-3.csv"
		feather_path_list[5] = self.data_path+"/DealAllString/Col4half.csv"
		feather_path_list[6] = self.data_path+"/DealAllString/Col4half2.csv"
		feather_path_list[7] = self.data_path+"/DealAllString/Col5half.csv"
		feather_path_list[8] = self.data_path+"/DealAllString/Col5half2.csv"
		feather_path_list[9] = self.data_path+"/DealAllString/Col6.csv"
		feather_path_list[10] = self.data_path+"/DealAllString/Col7.csv"
		feather_path_list[11] = self.data_path+"/DealAllString/Col8half.csv"
		feather_path_list[12] = self.data_path+"/DealAllString/Col8half2.csv"
		feather_path_list[13] = self.data_path+"/DealAllString/Col9half.csv"
		feather_path_list[14] = self.data_path+"/DealAllString/Col9half2.csv"
		feather_path_list[15] = self.data_path+"/DealAllString/Col10half.csv"
		feather_path_list[16] = self.data_path+"/DealAllString/Col10half2.csv"
		feather_path_list[17] = self.data_path+"/DealAllString/Col11half.csv"
		feather_path_list[18] = self.data_path+"/DealAllString/Col11half2.csv"
		feather_path_list[19] = self.data_path+"/DealAllString/Col12half.csv"
		feather_path_list[20] = self.data_path+"/DealAllString/Col12half2.csv"
		feather_path_list[21] = self.data_path+"/DealAllString/Col13half.csv"
		feather_path_list[22] = self.data_path+"/DealAllString/Col13half2.csv"
		feather_path_list[23] = self.data_path+"/DealAllString/Col14half.csv"
		feather_path_list[24] = self.data_path+"/DealAllString/Col14half2.csv"
		feather_path_list[25] = self.data_path+"/DealAllString/Col15half.csv"
		feather_path_list[26] = self.data_path+"/DealAllString/Col15half2.csv"
		feather_path_list[27] = self.data_path+"/DealAllString/Col16half.csv"
		feather_path_list[28] = self.data_path+"/DealAllString/Col16half2.csv"
		feather_path_list[29] = self.data_path+"/DealAllString/Col17half.csv"
		feather_path_list[30] = self.data_path+"/DealAllString/Col17half2.csv"
		feather_path_list[31] = self.data_path+"/DealAllString/Col18half.csv"
		feather_path_list[32] = self.data_path+"/DealAllString/Col18half2.csv"
		feather_path_list[33] = self.data_path+"/DealAllString/Col19half.csv"
		feather_path_list[34] = self.data_path+"/DealAllString/Col19half2.csv"
		feather_path_list[35] = self.data_path+"/DealAllString/Col20half.csv"
		feather_path_list[36] = self.data_path+"/DealAllString/Col20half2.csv"
		feather_path_list[37] = self.data_path+"/DealAllString/Col21half.csv"
		feather_path_list[38] = self.data_path+"/DealAllString/Col21half2.csv"
		feather_path_list[39] = self.data_path+"/DealAllString/Col22half.csv"
		feather_path_list[40] = self.data_path+"/DealAllString/Col22half2.csv"
		feather_path_list[41] = self.data_path+"/DealAllString/Col23half.csv"
		feather_path_list[42] = self.data_path+"/DealAllString/Col23half2.csv"
		feather_path_list[43] = self.data_path+"/DealAllString/Col24quarter1.csv"
		feather_path_list[44] = self.data_path+"/DealAllString/Col24quarter2.csv"
		feather_path_list[45] = self.data_path+"/DealAllString/Col24quarter3.csv"
		feather_path_list[46] = self.data_path+"/DealAllString/Col24quarter4.csv"
		feather_path_list[47] = self.data_path+"/DealAllString/Col25quarter1.csv"
		feather_path_list[48] = self.data_path+"/DealAllString/Col25quarter2.csv"
		feather_path_list[49] = self.data_path+"/DealAllString/Col25quarter3.csv"
		feather_path_list[50] = self.data_path+"/DealAllString/Col25quarter4.csv"
		feather_path_list[51] = self.data_path+"/DealAllString/All_Factorize.csv"

		for i in feather_path_list:
			if not os.path.exists(i):
				print ("Not found data. Run the Preprocess.Auto function please.")
				return
		train_label = pd.read_csv(self.data_path+"/Standardlize/label.csv",header=None)
		feather_list = list()
		for i in feather_path_list:
			temp = pd.read_csv(i, header=None)
			feather_list.append(temp)

		all_data = pd.concat(feather_list,axis=1)
		all_data.columns = range(len(all_data.columns))
		train_count = len(train_label)
		train_data = all_data.iloc[:train_count, :]
		test_data = all_data.iloc[train_count:, :]
		self.train_data = train_data.values
		self.test_data = test_data.values
		self.label = train_label.replace(-1,0).values.squeeze()

	def train(self):
		param_dict = {"device":"cpu","threads":"12"}
		lgb_final = LGBMClassifier(
	        num_leaves=40, boosting_type='gbdt', learning_rate=0.01, n_estimators=1400, subsample_for_bin=10000,
	        objective='binary', min_split_gain=0., min_child_weight=1e-3, min_child_samples=40,subsample=.6, 
	        subsample_freq=1, colsample_bytree=.8, reg_alpha=0.15, reg_lambda=0., n_jobs=12, silent=False,random_state=1377,
	        **param_dict)
		lgb_final.fit(X=self.train_data, y=self.label, eval_metric='auc',verbose=True)
		self.lgbm_trained = lgb_final

	def predict(self):
		pred = self.lgbm_trained.predict_proba(X=self.test_data)
		index = pd.DataFrame(data=np.arange(1, len(self.test_data)+1))
		pred_pandas=pd.DataFrame(data=pred[:,1])
		pred_pandas = pd.concat([index, pred_pandas], axis=1)
		pred_pandas.to_csv('./LGBM_Predict.csv', header=['Id', 'label'], index=False)

	def CV(self):
		param_dict = {"device":"cpu","threads":"12"}
		lgb_para = LGBMClassifier(
			num_leaves=40, boosting_type='gbdt', max_depth=-1, learning_rate=0.01, n_estimators=2000,
			subsample_for_bin=10000, objective='binary', min_split_gain=0., min_child_weight=1e-3, 
			min_child_samples=40, subsample=.6, subsample_freq=1, colsample_bytree=.8,
			reg_alpha=0.15, reg_lambda=0., random_state=1377, n_jobs=12, silent=False, **param_dict)
		lgb_param = lgb_para.get_params()
		lgtrain = lgb.Dataset(data=self.train_data, label=self.label)
		lgb.cv(lgb_param, lgtrain, num_boost_round=lgb_para.get_params()['n_estimators'],
					nfold=5,stratified=True,shuffle=True,metrics='auc',
					seed=1377,
					init_model=None,
					early_stopping_rounds=150, show_stdv=True, verbose_eval=True)
