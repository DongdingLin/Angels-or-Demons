# 数据特征处理部分
# Author：lindongding
# Time：2018年5月2日
# PS：直接运行是不成功的，每一部分都是一种做法
# PPS：主要学习的是特征处理的思想
import pandas as pd
import numpy as np
import sys
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
warnings.filterwarnings("ignore")

# 规范化数据部分
pan_train = pd.read_csv("./DataSet/train.csv", header=None)
pan_test = pd.read_csv("./DataSet/test.csv", header=None)
pan_train.drop([pan_train.columns[247]], axis=1,inplace=True)
pan_test.drop([pan_test.columns[246]], axis=1,inplace=True)
pan_train_label = pan_train.iloc[:, 246]
pan_train_data = pan_train.iloc[:, 0:246]
data_all = pd.concat([pan_train_data, pan_test], axis=0)

pan_train_label.to_csv('./DataSet/train_label.csv',header=False, index=False)
data_all.to_csv('./DataSet/data_all.csv', header=False, index=False)
pan_train_data.to_csv('./DataSet/train_data.csv',header=False, index=False)
pan_test.to_csv('./DataSet/test_data.csv',header=False, index=False)

# 清洗数据,去掉缺失值大于90%的列，以及无信息量的列
data_all = pd.read_csv('./DataSet/data_all.csv', header=None)
rowsOfTotal= len(data_all)
rowsOfNonMiss = rowsOfTotal * (1-0.9)
data_all.dropna(thresh=rowsOfNonMiss,axis=1,inplace=True)
for i in data_all.columns:
    if data_all.loc[:,i].value_counts().count() == 1:
        data_all.drop(i, axis=1, inplace=True)
data_all.columns = range(data_all.shape[1])

# 将数据分为字符和数值，完整与不完整
full_data_string = pd.DataFrame()
full_data_number = pd.DataFrame()
defect_data_string = pd.DataFrame()
defect_data_number = pd.DataFrame()
for i in data_all.columns:
    ty = data_all.loc[:,i]
    type_  = str(ty.dtype)
    if type_ != 'float64' and type_ != 'int64':
        if ty.isnull().any() == False:
            if (full_data_string.size == 0):
                full_data_string = pd.DataFrame(ty)
            else:
                full_data_string = pd.concat([full_data_string, pd.DataFrame(ty)], axis=1)
        else:
            if (defect_data_string.size == 0):
                defect_data_string = pd.DataFrame(ty)
            else:
                defect_data_string = pd.concat([defect_data_string, pd.DataFrame(ty)], axis=1)
    else:
        if ty.isnull().any() == False:
            if (full_data_number.size == 0):
                full_data_number = pd.DataFrame(ty)
            else:
                full_data_number = pd.concat([full_data_number, pd.DataFrame(ty)], axis=1)
        else:
            if (defect_data_number.size == 0):
                defect_data_number = pd.DataFrame(ty)
            else:
                defect_data_number = pd.concat([defect_data_number, pd.DataFrame(ty)], axis=1)
full_data_string.to_csv('./DataSet/full_data_string.csv', header=False, index=False)
full_data_number.to_csv('./DataSet/full_data_number.csv', header=False, index=False)
defect_data_string.to_csv('./DataSet/defect_data_string.csv', header=False, index=False)
defect_data_number.to_csv('./DataSet/defect_data_number.csv', header=False, index=False)

# 填补数值缺失部分，从缺失最小的开始填补，填完就加到full data里
min_max_scaler = preprocessing.MinMaxScaler()
while defect_data_number.size != 0:
    min_max_data = min_max_scaler.fit_transform(full_data_number.values)
    full_data_number = pd.DataFrame(data=min_max_data)
    # Find the most least defect column
    min_col_r = 100
    min_col_name = None
    for i in defect_data_number.columns:
        ty = defect_data_number.loc[:, i]
        if ty.value_counts().count() == 1:
            defect_data_number.drop(i, axis=1, inplace=True)
            continue
        d=len(ty)-ty.count()
        r=(d/len(ty))*100
        if r < min_col_r and r != 0:
            min_col_r = r
            min_col_name = i
    # use RandomForest
    if min_col_r < 10:
        print (min_col_name)
        print (min_col_r)
        print (defect_data_number.loc[:,min_col_name].value_counts().count())
        if (defect_data_number.loc[:,min_col_name].value_counts().count() <= 100):
            temp = Fill_Data_Number(defect_data_number.loc[:,min_col_name], full_data_number, "classifier")
        else:
            temp = Fill_Data_Number(defect_data_number.loc[:,min_col_name], full_data_number, "regressor")
        defect_data_number.drop(min_col_name, axis=1, inplace=True)
        defect_data_number.columns = range(defect_data_number.shape[1])
        full_data_number = pd.concat([full_data_number, temp], axis=1)
        full_data_number.columns = range(full_data_number.shape[1])
    else:
        defect_data_number.fillna(defect_data_number.mean(), inplace=True)
        full_data_number = pd.concat([full_data_number, defect_data_number], axis=1)
        full_data_number.columns = range(full_data_number.shape[1])
        defect_data_number.drop(defect_data_number.columns, axis=1, inplace=True)
min_max_data = min_max_scaler.fit_transform(full_data_number.values)
full_data_number = pd.DataFrame(data=min_max_data)
full_data_number.to_csv('./DataSet/Full_data_Number_Final.csv',header=False, index=False)

# 用决策树的方法填充数据
def Fill_Data_Number(series1, known_pd, type_):
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

# 检测数据中缺失的部分及其比例
for i in data_all.columns:
    d=len(data_all)-data_all.loc[:,i].count()
    r=(d/len(data_all))*100
    ty = str(data_all.loc[:,i].dtype)
    rate='%.2f%%' % r
    print('字段名为:',str(i).ljust(1), '字段类型:', ty.ljust(6),'数量:',str(d).ljust(4),'占比：',rate,'字段形式:', data_all.loc[0, i])

# 缺失值<10%的以均值填充
for i in data_all.columns:
    d=len(data_all)-data_all.loc[:,i].count()
    r=(d/len(data_all))*100
    rate='%.2f%%' % r
    if d != 0 and r < 10 and str(data_all.loc[:,i].dtype) != 'object':
        mean_ = data_all.loc[:, i].mean()
        data_all.loc[:, i].fillna(mean_, inplace=True)
data_all.columns = range(data_all.shape[1])

# 以众数填充
word = data_all.iloc[:, i].mode().loc[0]
data_all.iloc[:, i].fillna(word, inplace=True)

# 处理时间列，方法为距离最早的时间的天数
data_all = pd.read_csv('./DataSet/data_all.csv',header=None)
time_t = pd.to_datetime(data_all.iloc[:,198], format='%Y-%m-%d-%H.%M.%S.%f')
data_all.drop(198, 1, inplace=True)
min_t = min(time_t)
list_day = [(i-min_t).days for i in time_t]
days = pd.DataFrame(list_day)
data_all = pd.concat([data_all, days], axis=1)
data_all.to_csv('./DataSet/data_all_3Full_with_day.csv', header=False, index=False)

# 处理时间列，以月份划分，转化成Year-Month
time_t = pd.to_datetime(data_all.loc[:,180], format='%Y-%m-%d-%H.%M.%S.%f').dt.strftime('%Y-%m')
data_all.loc[:, 180] = time_t
data_all.to_csv('./DataSet2/data_all_After_90%off_10%mean_ym.csv',  header=False, index=False)
# 处理时间列，以年月日划分
time_t = pd.to_datetime(data_all.loc[:,180], format='%Y-%m-%d-%H.%M.%S.%f').dt.strftime('%Y-%m-%d')
data_all.loc[:, 180] = time_t
data_all.to_csv('./DataSet2/data_all_After_90%off_10%mean_ym.csv',  header=False, index=False)

# 将数据分为数值与非数值的（含缺失值）
data_all_numeric = pd.DataFrame()
for i in data_all.columns:
    ty = str(data_all.loc[:,i].dtype)
    if ty != 'object':
        data_all_numeric = pd.concat([data_all_numeric, data_all.loc[:,i]], axis=1)
        data_all.drop(i, axis=1, inplace=True)
# 还原列名
data_all.columns = range(data_all.shape[1]) 
data_all_numeric.to_csv('./DataSet2/data_all_After_90%off_10%mean_ym_numeric.csv',  header=False, index=False)
data_all.to_csv('./DataSet2/data_all_After_90%off_10%mean_ym_string.csv',  header=False, index=False)

# one-hot矩阵，factorize，处理字符串列
data_all = pd.read_csv('./DataSet2/data_all_After_90%off_10%mean_ym_string.csv', header=None)

# 处理时间列，直接dummies，factorize
temp = pd.get_dummies(data_all.loc[:,24])
temp.to_csv('./DataSet2/data_all_After_90%off_10%mean_ym_string_DateYM.csv',  header=False, index=False)
t = pd.DataFrame(pd.factorize(time_t)[0])
t.to_csv('./DataSet2/data_all_After_90%off_10%mean_ym_string_DateYM.csv.csv',  header=False, index=False)

# 处理0-3列
temp = pd.get_dummies(data_all.loc[:,0:3])
temp.to_csv('./DataSet2/data_all_After_90%off_10%mean_ym_string_Col0-3.csv',  header=False, index=False)

# 处理第6，7列
for i in list([6,7]):
    temp = pd.get_dummies(data_all.loc[:,i])
    temp.to_csv('./DataSet2/data_all_After_90%off_10%mean_ym_string_Col'+str(i)+'.csv',  header=False, index=False)

# 处理第4,5,8,9列
for i in list([4, 5, 8, 9]):
    temp = pd.get_dummies(data_all.loc[:,i].str[0:2])
    temp.to_csv('./DataSet2/data_all_After_90%off_10%mean_ym_string_Col'+str(i)+'half.csv',  header=False, index=False)
    temp = pd.get_dummies(data_all.loc[:,i].str[2:4])
    temp.to_csv('./DataSet2/data_all_After_90%off_10%mean_ym_string_Col'+str(i)+'half2.csv',  header=False, index=False)

# 处理第10,11列
for i in range(10,12,1):
    temp = pd.get_dummies(data_all.loc[:,i].str.upper()[:].str[0:2])
    temp.to_csv('./DataSet2/data_all_After_90%off_10%mean_ym_string_Col'+str(i)+'half.csv',  header=False, index=False)
    temp = pd.get_dummies(data_all.loc[:,i].str.upper()[:].str[2:])
    temp.to_csv('./DataSet2/data_all_After_90%off_10%mean_ym_string_Col'+str(i)+'half2.csv',  header=False, index=False)

# 处理12-17列
for i in range(12,18,1):
    col_temp = data_all.loc[:,i].str.extract('([a-zA-Z]*)([\d]*)')
    temp = pd.get_dummies(col_temp.loc[:,0])
    temp.to_csv('./DataSet2/data_all_After_90%off_10%mean_ym_string_Col'+str(i)+'half.csv',  header=False, index=False)
    temp = pd.get_dummies(col_temp.loc[:,1])
    temp.to_csv('./DataSet2/data_all_After_90%off_10%mean_ym_string_Col'+str(i)+'half2.csv',  header=False, index=False)

# 处理28-23列
for i in range(18,24,1):
    col_temp = data_all.loc[:,i].str.extract('([a-zA-Z]*)([\d]*)')
    temp = pd.get_dummies(col_temp.loc[:,0])
    temp.to_csv('./DataSet2/data_all_After_90%off_10%mean_ym_string_Col'+str(i)+'half.csv',  header=False, index=False)
    col_temp.loc[:,1].to_csv('./DataSet2/data_all_After_90%off_10%mean_ym_string_Col'+str(i)+'half2.csv',  header=False, index=False)

# 25-26列
col_temp = data_all.loc[:,25].str.extract('([a-zA-Z]*)([\d]*)([a-zA-Z]*)([\d]*)')
pd.get_dummies(col_temp.loc[:,0]).to_csv('./DataSet2/data_all_After_90%off_10%mean_ym_string_Col25quarter1.csv',  header=False, index=False)
col_temp.loc[:,1].to_csv('./DataSet2/data_all_After_90%off_10%mean_ym_string_Col25quarter2.csv',  header=False, index=False)
pd.get_dummies(col_temp.loc[:,2]).to_csv('./DataSet2/data_all_After_90%off_10%mean_ym_string_Col25quarter3.csv',  header=False, index=False)
col_temp.loc[:,3].to_csv('./DataSet2/data_all_After_90%off_10%mean_ym_string_Col25quarter4.csv',  header=False, index=False)

col_temp = data_all.loc[:,26].str.extract('([a-zA-Z]*)([\d]*)-([a-zA-Z]*)-([\d]*)')
pd.get_dummies(col_temp.loc[:,0]).to_csv('./DataSet2/data_all_After_90%off_10%mean_ym_string_Col26quarter1.csv',  header=False, index=False)
col_temp.loc[:,1].to_csv('./DataSet2/data_all_After_90%off_10%mean_ym_string_Col26quarter2.csv',  header=False, index=False)
pd.get_dummies(col_temp.loc[:,2]).to_csv('./DataSet2/data_all_After_90%off_10%mean_ym_string_Col26quarter3.csv',  header=False, index=False)
col_temp.loc[:,3].to_csv('./DataSet2/data_all_After_90%off_10%mean_ym_string_Col26quarter4.csv',  header=False, index=False)

# 将所有字符直接factorize
temp = pd.DataFrame()
for i in data_all.columns:
    t = pd.DataFrame(pd.factorize(data_all[i])[0])
    temp = pd.concat([temp, t], axis=1)
temp.to_csv('./DataSet2/data_all_After_90%off_10%mean_ym_string_Factorize.csv',  header=False, index=False)

# 各个部分进行归一化操作，这里用min-max
min_max_scaler = preprocessing.MinMaxScaler()
min_max_data = min_max_scaler.fit_transform(data_all.values)
data_all = pd.DataFrame(data=min_max_data)

# 还原成train data和test data
train_label = pd.read_csv('./DataSet/train_label.csv',header=None)
train_count = len(train_label)
train_data = data_all.iloc[:train_count, :]
test_data = data_all.iloc[train_count:, :]
train_data.to_csv('./DataSet2/Last_Train.csv',header=False, index=False)
test_data.to_csv('./DataSet2/Last_Test.csv',header=False, index=False)
train_label.to_csv('./DataSet2/Last_Train_Label.csv',header=False, index=False)
