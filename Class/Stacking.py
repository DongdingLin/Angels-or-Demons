# Class for LightGBM
# Author：lindongding
# Time：May.10.2018
import pandas as pd
import numpy as np
import sys,warnings,time,os
from sklearn.model_selection import StratifiedKFold,KFold,GridSearchCV
import lightgbm as lgb
from lightgbm import LGBMClassifier,LGBMRegressor
from xgboost import XGBClassifier
warnings.filterwarnings("ignore")

class Stacking:
	def __init__(self, data_path):
		self.data_path = os.path.abspath(data_path)
		self.train_data = None
		self.test_data = None
		self.label = None
		self.__LoadData()

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

	def __model_fit_pred(self, model, para_dict, train_set, train_label, vali_set, vali_label, test_set):
		if model == 'RF':
			RF_model = RandomForestClassifier(bootstrap=para_dict["bootstrap"],n_jobs=para_dict["n_jobs"], max_depth=para_dict["max_depth"], n_estimators=para_dict["n_estimators"])
			RF_model.fit(train_set, train_label)
			if RF_model.predict_proba(vali_set).shape[1] == 2:
				train_pred = RF_model.predict_proba(vali_set)[:,1]
			else:
				train_pred = RF_model.predict_proba(vali_set)[:,0]
				
			if RF_model.predict_proba(test_set).shape[1] == 2:
				test_pred = RF_model.predict_proba(test_set)[:,1]
			else:
				test_pred = RF_model.predict_proba(test_set)[:,0]
		elif model == 'XGBoost':
			param_dict = {'tree_method':'gpu_hist'}
			xgb_model = XGBClassifier(
				learning_rate=para_dict["learning_rate"], n_estimators=para_dict["n_estimators"], max_depth=para_dict["max_depth"],subsample=para_dict["subsample"], nthread=para_dict["nthread"], **param_dict)
			xgb_model.fit(train_set, train_label)
			train_pred = xgb_model.predict_proba(vali_set)[:,1]
			test_pred = xgb_model.predict_proba(test_set)[:,1]
		elif model == 'LGBM':
			lgb_model = LGBMClassifier(
				num_leaves=para_dict["num_leaves"], learning_rate=para_dict["learning_rate"], n_estimators=para_dict["n_estimators"], subsample_for_bin=para_dict["subsample_for_bin"], min_child_samples=para_dict["min_child_samples"], subsample=para_dict["subsample"], n_jobs=para_dict["n_jobs"])
			lgb_model.fit(train_set, train_label)
			train_pred = lgb_model.predict_proba(vali_set)[:,1]
			test_pred = lgb_model.predict_proba(test_set)[:,1]
		return train_pred, test_pred
	
	# stacking第二层
	def __stacking(self, model_list, model_para_list, X, y, test_set):
		new_train_matrix = list()
		new_test_matrix  = list()
		for i in range(len(model_list)):
			skf = StratifiedKFold(n_splits=5,shuffle=True, random_state=1377)
			new_train_col = np.zeros(len(y))
			new_test_col = np.zeros(len(test_set))
			for train_index, test_index in skf.split(X, y):
				X_train, X_vali = X[train_index], X[test_index]
				y_train, y_vali = y[train_index], y[test_index]

				train_pred, test_pred = self.__model_fit_pred(model_list[i], model_para_list[i], X_train, y_train, X_vali, y_vali, test_set)
				new_train_col[test_index] = train_pred
				new_test_col = new_test_col+test_pred
			new_test_col = new_test_col/5
			new_train_matrix.append(new_train_col)
			new_test_matrix.append(new_test_col)
		# Second level learning model
		new_train_matrix = np.array(new_train_matrix)
		new_test_matrix = np.array(new_test_matrix)
		new_train_matrix,new_test_matrix = new_train_matrix.T,new_test_matrix.T
		param_dict = {'tree_method':'gpu_hist'}
		layer2_model = XGBClassifier(silent=False, learning_rate=0.1, n_estimators=100, nthread=10, seed=1337, **param_dict)
		layer2_model.fit(new_train_matrix, y)
		pred_result = layer2_model.predict_proba(new_test_matrix)[:,1]
		return pred_result

	def predict(self):
		train_data = pd.DataFrame(self.train_data).fillna('-1').values
		test_data = pd.DataFrame(self.test_data).fillna('-1').values
		model_list = ['RF', 'XGBoost', 'LGBM']
		model_para_list = [{"bootstrap":True, "n_jobs":6, "max_depth":12, "n_estimators":170},
		{"learning_rate":0.1, "n_estimators":300, "max_depth":10, "subsample":1,"nthread":10},
		{"num_leaves":40, "learning_rate":0.01,"n_estimators":1200, "subsample_for_bin":10000, "min_child_samples":40, "subsample":.5,"n_jobs":12}]
		result = self.__stacking(model_list, model_para_list, train_data, self.label, test_data)
		index = pd.DataFrame(data=np.arange(1, len(self.test_data)+1))
		pred_pandas=pd.DataFrame(data=result)
		pred_pandas = pd.concat([index, pred_pandas], axis=1)
		pred_pandas.to_csv('./Stacking.csv', header=['Id', 'label'], index=False)