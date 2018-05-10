# Main Progarm
import argparse
import pandas as pd
import numpy as np
import sys,warnings,time,os
from sklearn.model_selection import StratifiedKFold,KFold,GridSearchCV
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from xgboost import plot_importance
from matplotlib import pyplot
import lightgbm as lgb
from lightgbm import LGBMClassifier,LGBMRegressor
from Class.Preprocess import Preprocess
from Class.LightGBM import LightGBM
from Class.Stacking import Stacking 
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--method', type=str, default="LightGBM",help="Choose method to process. LightGBM, Stacking, Preprocess are available.")
parser.add_argument('--data_path', type=str, default="DataSet",help="The data path involved train.csv and test.csv")

args = parser.parse_args()
if args.method == "LightGBM":
	ligb = LightGBM(args.data_path)
	ligb.train()
	ligb.predict()
elif args.method == "Stacking":
	stacking = Stacking(args.data_path)
	stacking.predict()
elif args.method == "Preprocess":
	Prepro = Preprocess(args.data_path)
	Prepro.Auto()
else:
	print ('Receive a Wrong Input, Please Check again!')

