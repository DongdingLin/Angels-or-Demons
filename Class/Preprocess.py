# Class for preprocess
# Author：lindongding
# Time：May.9.2018
import pandas as pd
import numpy as np
import sys, os
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
import warnings
warnings.filterwarnings("ignore")

class Preprocess:

	def __init__(self, data_path):
		self.data_path = os.path.abspath(data_path)

	# 规范化数据部分
	def Standardlize(self):
		# Split label and feature, label in line 246
		train_file_path = self.data_path + "/train.csv"
		test_file_path = self.data_path + "/test.csv"

		train_data = pd.read_csv(train_file_path, header=None)
		test_data = pd.read_csv(test_file_path, header=None)
		train_data.drop(247, axis=1,inplace=True)
		test_data.drop(246, axis=1,inplace=True)
		
		label = train_data.iloc[:, 246]
		train_data = train_data.iloc[:, 0:246]
		all_data = pd.concat([train_data, test_data], axis=0)

		out_path = self.data_path+"/Standardlize"
		if not os.path.exists(out_path):
			os.makedirs(out_path)
		label.to_csv(out_path+"/label.csv",header=False, index=False)
		all_data.to_csv(out_path+"/all_data.csv", header=False, index=False)
		train_data.to_csv(out_path+"/train_data.csv",header=False, index=False)
		test_data.to_csv(out_path+"/test_data.csv",header=False, index=False)
		print ("Standardlize Finished!")

	# 清洗数据,去掉缺失值大于90%的列，以及无信息量的列
	def DropUselessData(self):
		all_data_path = self.data_path+"/Standardlize/all_data.csv"
		if not os.path.exists(all_data_path):
			print ("Not found all data. Run the Standardlize function please.")
			return
		out_path = self.data_path+"/DropUselessData"
		if not os.path.exists(out_path):
			os.makedirs(out_path)

		all_data = pd.read_csv(all_data_path, header=None)
		rowsOfTotal= len(all_data)
		rowsOfNonMiss = rowsOfTotal * (1-0.9)
		all_data.dropna(thresh=rowsOfNonMiss,axis=1,inplace=True)
		for i in all_data.columns:
		    if all_data.loc[:,i].value_counts().count() == 1:
		        all_data.drop(i, axis=1, inplace=True)
		all_data.to_csv(out_path+"/all_data.csv", header=False, index=False)
		print ("Drop Useless Data Finished!")

	# 将数据分为字符和数值，完整与不完整
	def SpiltToNumDefect(self):
		all_data_path = self.data_path+"/DropUselessData/all_data.csv"
		if not os.path.exists(all_data_path):
			print ("Not found data. Run the DropUselessData function please.")
			return
		out_path = self.data_path+"/SpiltToNumDefect"
		if not os.path.exists(out_path):
			os.makedirs(out_path)
		all_data = pd.read_csv(all_data_path, header=None)

		full_data_string = pd.DataFrame()
		full_data_number = pd.DataFrame()
		defect_data_string = pd.DataFrame()
		defect_data_number = pd.DataFrame()
		for i in all_data.columns:
		    ty = all_data.loc[:,i]
		    type_  = str(ty.dtype)
		    if type_ != 'float64' and type_ != 'int64':
		        if ty.isnull().any() == False:
		            full_data_string = pd.concat([full_data_string, pd.DataFrame(ty)], axis=1)
		        else:
		            defect_data_string = pd.concat([defect_data_string, pd.DataFrame(ty)], axis=1)
		    else:
		        if ty.isnull().any() == False:
		            full_data_number = pd.concat([full_data_number, pd.DataFrame(ty)], axis=1)
		        else:
		            defect_data_number = pd.concat([defect_data_number, pd.DataFrame(ty)], axis=1)
		full_data_string.to_csv(out_path+"/full_data_string.csv", header=False, index=False)
		full_data_number.to_csv(out_path+"/full_data_number.csv", header=False, index=False)
		defect_data_string.to_csv(out_path+"/defect_data_string.csv", header=False, index=False)
		defect_data_number.to_csv(out_path+"/defect_data_number.csv", header=False, index=False)
		print ("SpiltToNumDefect Finished!")

	# 用决策树的方法填充数据
	def __Fill_Data_Number(self, series1, known_pd, type_):
	    unknown_pd = pd.DataFrame(data=series1)
	    all_pd = pd.concat([unknown_pd, known_pd], axis=1)
	    all_pd.columns = range(all_pd.shape[1])
	    columns_0 = all_pd.columns[0]
	    y_train = all_pd[columns_0][all_pd[columns_0].notnull()].values
	    x_train = all_pd[all_pd[columns_0].notnull()].drop([columns_0],axis=1).values
	    x_test = all_pd[all_pd[columns_0].isnull()].drop([columns_0],axis=1).values
	    if type_ == "classifier":
	        rfc = RandomForestClassifier().fit(x_train, y_train.astype('str'))
	    else:
	        rfc = RandomForestRegressor().fit(x_train, y_train)
	    all_pd[columns_0][all_pd[columns_0].isnull()] = (rfc.predict(x_test)).astype('float')
	    return all_pd.loc[:,columns_0]

	# 填补数值缺失部分，从缺失最小的开始填补，填完就加到full data里
	def FillDefectNumber(self):
		defect_data_path = self.data_path+"/SpiltToNumDefect/defect_data_number.csv"
		full_data_path = self.data_path+"/SpiltToNumDefect/full_data_number.csv"
		if not os.path.exists(defect_data_path) or not os.path.exists(full_data_path):
			print ("Not found data. Run the SpiltToNumDefect function please.")
			return 
		out_path = self.data_path+"/FillDefectNumber"
		if not os.path.exists(out_path):
			os.makedirs(out_path)
		defect_data = pd.read_csv(defect_data_path, header=None)
		full_data = pd.read_csv(full_data_path, header=None)

		min_max_scaler = preprocessing.MinMaxScaler()
		while defect_data.size != 0:
		    min_max_data = min_max_scaler.fit_transform(full_data.values)
		    full_data = pd.DataFrame(data=min_max_data)
		    # Find the most least defect column
		    min_col_r = 100
		    min_col_name = None
		    for i in defect_data.columns:
		        ty = defect_data.loc[:, i]
		        if ty.value_counts().count() == 1:
		            defect_data.drop(i, axis=1, inplace=True)
		            continue
		        d=len(ty)-ty.count()
		        r=(d/len(ty))*100
		        if r < min_col_r and r != 0:
		            min_col_r = r
		            min_col_name = i
		    # use RandomForest
		    if min_col_r < 10:
		        if (defect_data.loc[:,min_col_name].value_counts().count() <= 100):
		            temp = self.__Fill_Data_Number(defect_data.loc[:,min_col_name], full_data, "classifier")
		        else:
		            temp = self.__Fill_Data_Number(defect_data.loc[:,min_col_name], full_data, "regressor")
		        defect_data.drop(min_col_name, axis=1, inplace=True)
		        defect_data.columns = range(defect_data.shape[1])
		        full_data = pd.concat([full_data, temp], axis=1)
		        full_data.columns = range(full_data.shape[1])
		    else:
		        defect_data.fillna(defect_data.mean(), inplace=True)
		        full_data = pd.concat([full_data, defect_data], axis=1)
		        full_data.columns = range(full_data.shape[1])
		        defect_data.drop(defect_data.columns, axis=1, inplace=True)
		min_max_data = min_max_scaler.fit_transform(full_data.values)
		full_data = pd.DataFrame(data=min_max_data)
		full_data.to_csv(out_path+"/Full_data_Number.csv",header=False, index=False)
		print ("Fill Defect Number Finished!")

	# 检测数据中缺失的部分及其比例
	def CheckData(self, path=None):
		if path is None:
			all_data_path = self.data_path+"/DropUselessData/all_data.csv"
		else:
			all_data_path = path+"/all_data.csv"
		if not os.path.exists(all_data_path):
			print ("Not found data. Run the DropUselessData function please.")
			return 
		all_data = pd.read_csv(all_data_path, header=None)
		for i in all_data.columns:
		    d=len(all_data)-all_data.loc[:,i].count()
		    r=(d/len(all_data))*100
		    ty = str(all_data.loc[:,i].dtype)
		    rate='%.2f%%' % r
		    print('字段名为:',str(i).ljust(1), '字段类型:', ty.ljust(6),'数量:',str(d).ljust(4),'占比：',rate,'字段形式:', all_data.loc[0, i])

	# 缺失值<10%的以均值填充
	def FillMeanNumberLess10Pec(self):
		all_data_path = self.data_path+"/DropUselessData/all_data.csv"
		if not os.path.exists(all_data_path):
			print ("Not found data. Run the DropUselessData function please.")
			return 
		out_path = self.data_path+"/FillMeanNumberLess10Pec"
		if not os.path.exists(out_path):
			os.makedirs(out_path)
		all_data = pd.read_csv(all_data_path, header=None)

		for i in all_data.columns:
		    d=len(all_data)-all_data.loc[:,i].count()
		    r=(d/len(all_data))*100
		    if d != 0 and r < 10 and str(all_data.loc[:,i].dtype) != 'object':
		        mean_ = all_data.loc[:, i].mean()
		        all_data.loc[:, i].fillna(mean_, inplace=True)
		all_data.to_csv(out_path+"/all_data.csv", header=False, index=False)
		print ("Fill with Mean Number Less than 10 Percent Finished!")


	# 以众数填充
	def FillMode(self):
		all_data_path = self.data_path+"/FillMeanNumberLess10Pec/all_data.csv"
		if not os.path.exists(all_data_path):
			print ("Not found data. Run the FillMeanNumberLess10Pec function please.")
			return
		out_path = self.data_path+"/FillMode"
		if not os.path.exists(out_path):
			os.makedirs(out_path)
		all_data = pd.read_csv(all_data_path, header=None)
		for i in all_data.columns:
			word = all_data.iloc[:, i].mode().loc[0]
			all_data.iloc[:, i].fillna(word, inplace=True)
		all_data.to_csv(out_path+"/all_data.csv", header=False, index=False)
		print ("Fill with Mode Number Finished!")

	# 处理时间列，方法为距离最早的时间的天数,年月,年月日
	def DealDate(self):
		all_data_path = self.data_path+"/FillMeanNumberLess10Pec/all_data.csv"
		if not os.path.exists(all_data_path):
			print ("Not found data. Run the FillMeanNumberLess10Pec function please.")
			return
		out_path = self.data_path+"/DealDate"
		if not os.path.exists(out_path):
			os.makedirs(out_path)
		all_data = pd.read_csv(all_data_path, header=None)
		time_t = pd.to_datetime(all_data.iloc[:,180], format='%Y-%m-%d-%H.%M.%S.%f')
		all_data.drop(180, 1, inplace=True)
		all_data.to_csv(out_path+"/all_data_drop_date.csv", header=False, index=False)
		min_t = min(time_t)
		list_day = [(i-min_t).days for i in time_t]
		date_file = pd.DataFrame(list_day)
		date_file.to_csv(out_path+"/method1.csv", header=False, index=False)

		time_1 = time_t.dt.strftime('%Y-%m')
		date_file = pd.DataFrame(time_1)
		date_file.to_csv(out_path+"/method2.csv",  header=False, index=False)

		time_2 = time_t.dt.strftime('%Y-%m-%d')
		date_file = pd.DataFrame(time_2)
		date_file.to_csv(out_path+"/method3.csv",  header=False, index=False)
		print ("Deal with Date Finished!")

	# 将数据分为数值与非数值的（含缺失值）
	def SpiltToNumeric(self):
		all_data_path = self.data_path+"/DealDate/all_data_drop_date.csv"
		if not os.path.exists(all_data_path):
			print ("Not found data. Run the DealDate function please.")
			return
		out_path = self.data_path+"/SpiltToNumeric"
		if not os.path.exists(out_path):
			os.makedirs(out_path)
		all_data = pd.read_csv(all_data_path, header=None)

		all_data_numeric = pd.DataFrame()
		for i in all_data.columns:
		    ty = str(all_data.loc[:,i].dtype)
		    if ty != 'object':
		        all_data_numeric = pd.concat([all_data_numeric, all_data.loc[:,i]], axis=1)
		        all_data.drop(i, axis=1, inplace=True)
		all_data_numeric.to_csv(out_path+"/numeric_data.csv",  header=False, index=False)
		all_data.to_csv(out_path+"/string_data.csv",  header=False, index=False)
		print ("Spilt To Numeric Finished!")

	# 处理所有字符段
	def DealAllString(self):
		all_data_path = self.data_path+"/SpiltToNumeric/string_data.csv"
		if not os.path.exists(all_data_path):
			print ("Not found data. Run the SpiltToNumeric function please.")
			return
		time_0_path = self.data_path+"/DealDate/method1.csv"
		time_1_path = self.data_path+"/DealDate/method2.csv"
		time_2_path = self.data_path+"/DealDate/method3.csv"
		if not os.path.exists(time_0_path) or not os.path.exists(time_1_path) or not os.path.exists(time_2_path):
			print ("Not found data. Run the DealDate function please.")
			return
		out_path = self.data_path+"/DealAllString"
		if not os.path.exists(out_path):
			os.makedirs(out_path)

		all_data = pd.read_csv(all_data_path, header=None)
		time_0 = pd.read_csv(time_0_path, header=None)
		time_1 = pd.read_csv(time_1_path, header=None)
		time_2 = pd.read_csv(time_2_path, header=None)

		pd.DataFrame(pd.factorize(time_0.loc[:,0])[0]).to_csv(out_path+"/time_0_factorize.csv",  header=False, index=False)
		pd.get_dummies(time_0.loc[:,0]).to_csv(out_path+"/time_0_dummies.csv",  header=False, index=False)
		pd.DataFrame(pd.factorize(time_1.loc[:,0])[0]).to_csv(out_path+"/time_1_factorize.csv",  header=False, index=False)
		pd.get_dummies(time_1.loc[:,0]).to_csv(out_path+"/time_1_dummies.csv",  header=False, index=False)
		pd.DataFrame(pd.factorize(time_2.loc[:,0])[0]).to_csv(out_path+"/time_2_factorize.csv",  header=False, index=False)
		pd.get_dummies(time_2.loc[:,0]).to_csv(out_path+"/time_2_dummies.csv",  header=False, index=False)

		# 处理0-3列
		pd.get_dummies(all_data.loc[:,0:3]).to_csv(out_path+"/Col0-3.csv",  header=False, index=False)
		# 处理第6，7列
		for i in list([6,7]):
		    pd.get_dummies(all_data.loc[:,i]).to_csv(out_path+'/Col'+str(i)+'.csv',  header=False, index=False)

		# 处理第4,5,8,9列
		for i in list([4, 5, 8, 9]):
		    pd.get_dummies(all_data.loc[:,i].str[0:2]).to_csv(out_path+'/Col'+str(i)+'half.csv',  header=False, index=False)
		    pd.get_dummies(all_data.loc[:,i].str[2:4]).to_csv(out_path+'/Col'+str(i)+'half2.csv',  header=False, index=False)

		# 处理第10,11列
		for i in range(10,12,1):
		    pd.get_dummies(all_data.loc[:,i].str.upper()[:].str[0:2]).to_csv(out_path+'/Col'+str(i)+'half.csv',  header=False, index=False)
		    pd.get_dummies(all_data.loc[:,i].str.upper()[:].str[2:]).to_csv(out_path+'/Col'+str(i)+'half2.csv',  header=False, index=False)

		# 处理12-17列
		for i in range(12,18,1):
		    col_temp = all_data.loc[:,i].str.extract('([a-zA-Z]*)([\d]*)')
		    pd.get_dummies(col_temp.loc[:,0]).to_csv(out_path+'/Col'+str(i)+'half.csv',  header=False, index=False)
		    pd.get_dummies(col_temp.loc[:,1]).to_csv(out_path+'/Col'+str(i)+'half2.csv',  header=False, index=False)

		# 处理18-23列
		for i in range(18,24,1):
		    col_temp = all_data.loc[:,i].str.extract('([a-zA-Z]*)([\d]*)')
		    pd.get_dummies(col_temp.loc[:,0]).to_csv(out_path+'/Col'+str(i)+'half.csv',  header=False, index=False)
		    col_temp.loc[:,1].to_csv(out_path+'/Col'+str(i)+'half2.csv',  header=False, index=False)

		# 24-25列
		col_temp = all_data.loc[:,24].str.extract('([a-zA-Z]*)([\d]*)([a-zA-Z]*)([\d]*)')
		pd.get_dummies(col_temp.loc[:,0]).to_csv(out_path+'/Col24quarter1.csv',  header=False, index=False)
		col_temp.loc[:,1].to_csv(out_path+'/Col24quarter2.csv',  header=False, index=False)
		pd.get_dummies(col_temp.loc[:,2]).to_csv(out_path+'/Col24quarter3.csv',  header=False, index=False)
		col_temp.loc[:,3].to_csv(out_path+'/Col24quarter4.csv',  header=False, index=False)

		col_temp = all_data.loc[:,25].str.extract('([a-zA-Z]*)([\d]*)-([a-zA-Z]*)-([\d]*)')
		pd.get_dummies(col_temp.loc[:,0]).to_csv(out_path+'/Col25quarter1.csv',  header=False, index=False)
		col_temp.loc[:,1].to_csv(out_path+'/Col25quarter2.csv',  header=False, index=False)
		pd.get_dummies(col_temp.loc[:,2]).to_csv(out_path+'/Col25quarter3.csv',  header=False, index=False)
		col_temp.loc[:,3].to_csv(out_path+'/Col25quarter4.csv',  header=False, index=False)

		# 将所有字符直接factorize
		temp = pd.DataFrame()
		for i in all_data.columns:
		    t = pd.DataFrame(pd.factorize(all_data[i])[0])
		    temp = pd.concat([temp, t], axis=1)
		temp.to_csv(out_path+'/All_Factorize.csv',  header=False, index=False)
		print ("Deal All String Finished!")

	# 自动化
	def Auto(self):
		self.Standardlize()
		self.DropUselessData()
		self.FillMeanNumberLess10Pec()
		self.DealDate()
		self.SpiltToNumeric()
		self.DealAllString()