from sklearn.model_selection import train_test_split
from sklearn import svm
import matplotlib as plt
import pandas as pd
from sklearn import preprocessing
from sklearn import model_selection,metrics
from imblearn.over_sampling import SMOTE
import numpy as np
from sklearn.model_selection import GridSearchCV
import sklearn
from sklearn.tree import DecisionTreeRegressor
from imblearn.under_sampling import RandomUnderSampler
MMS=preprocessing.MinMaxScaler()
data=MMS.fit_transform(data)

data1=pd.read_excel("E:\比赛\DC新网杯\data\\trainxy\group1.xlsx",header=None)
data2=pd.read_excel("E:\比赛\DC新网杯\data\\trainxy\group2.xlsx",header=None)
data3=pd.read_excel("E:\比赛\DC新网杯\data\\trainxy\group3.xlsx",header=None)
rus=RandomUnderSampler({0:500})
data_resd1,rank_resed1=rus.fit_sample(data1.iloc[:,4:],data1.iloc[:,3])
data_resd2,rank_resed2=rus.fit_sample(data2.iloc[:,4:],data2.iloc[:,3])
data_resd3,rank_resed3=rus.fit_sample(data3.iloc[:,4:],data3.iloc[:,3])

tree=DecisionTreeRegressor()