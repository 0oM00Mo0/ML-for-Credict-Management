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
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.under_sampling import RandomUnderSampler
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from keras.models import Sequential
from keras.layers import Dense
from xgboost import  XGBClassifier
from sklearn.model_selection import learning_curve
from sklearn.ensemble import AdaBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import Imputer

#f1=open("E:\比赛\DC新网杯\\train_xy.csv")
#读取训练数据
data1=pd.read_excel("E:\比赛\DC新网杯\separate\\train1.xlsx",na_values=-99)
data2=pd.read_excel("E:\比赛\DC新网杯\separate\\train2.xlsx",na_values=-99)
data3=pd.read_excel("E:\比赛\DC新网杯\separate\\train3.xlsx",na_values=-99)
test1=pd.read_excel("E:\比赛\DC新网杯\separate\\test1.xlsx",na_values=-99)
test2=pd.read_excel("E:\比赛\DC新网杯\separate\\test2.xlsx",na_values=-99)
test3=pd.read_excel("E:\比赛\DC新网杯\separate\\test3.xlsx",na_values=-99)
Test1=test1.drop(['cust_id','cust_group'],axis=1)
Test2=test2.drop(['cust_id','cust_group'],axis=1)
Test3=test3.drop(['cust_id','cust_group'],axis=1)
print(np.shape(data1))
print(np.shape(test1))

x1=data1.iloc[:,3:]#删去标签和分组
x2=data2.iloc[:,3:]
x3=data3.iloc[:,3:]

#填充缺失值

'''
imputer=Imputer(strategy="median")
x1=imputer.fit_transform(x1)
imputer=Imputer(strategy="median")
Test1=imputer.fit_transform(Test1)
'''
#分离标签
y1=data1['y']
y2=data2['y']
y3=data3['y']
#欠采样
#rus=RandomUnderSampler({0:10000})
#x,y=rus.fit_sample(x,y)
#分离训练集，测试集
x_train1,x_test1,y_train1,y_test1 = train_test_split(x1,y1,test_size=0.3,random_state=1)
x_train2,x_test2,y_train2,y_test2 = train_test_split(x2,y2,test_size=0.3,random_state=1)
x_train3,x_test3,y_train3,y_test3 = train_test_split(x3,y3,test_size=0.3,random_state=1)


#f2=open("E:\比赛\DC新网杯\\test_all.csv")
#读取待预测数据



#筛选变量
'''
tree=XGBClassifier()
tree.fit(x_train,y_train)
model = SelectFromModel(tree, prefit=True)
x_train = model.transform(x_train)
x_test = model.transform(x_test)
Test1=model.transform(Test1)
print(np.shape(x_train))
'''
'''
#网格搜索
if __name__ == '__main__':
    parameters = {'learning_rate':[0.06,0.07]}
    lgbm=XGBClassifier(booster="gbtree",subsample=0.7,n_estimators=140,max_depth=4,gamma=0.01,reg_lambda=0.08,reg_alpha=0.08,min_child_weight=1,learning_rate=0.06)
    clf=GridSearchCV(lgbm,parameters,cv=2,scoring='roc_auc',n_jobs=-1)#
    clf.fit(x_train,y_train)
    print("best:",clf.best_score_)
    print("using:",clf.best_params_)
'''
#定义，训练分类器
s1=XGBClassifier(booster="gbtree",subsample=0.7,n_estimators=140,max_depth=4,gamma=0.01,reg_lambda=0.08,reg_alpha=0.08,min_child_weight=20,learning_rate=0.06)
s2=XGBClassifier(booster="gbtree",subsample=0.7,n_estimators=140,max_depth=4,gamma=0.01,reg_lambda=0.08,reg_alpha=0.08,min_child_weight=1,learning_rate=0.06)
s3=XGBClassifier(booster="gbtree",subsample=0.7,n_estimators=140,max_depth=4,gamma=0.01,reg_lambda=0.08,reg_alpha=0.08,min_child_weight=20,learning_rate=0.06)

#AdaBoostClassifier(algorithm="SAMME.R",learning_rate=0.04,n_estimators=300)
#XGBClassifier(booster="gbtree",subsample=0.7,n_estimators=140,max_depth=4,gamma=0.01,reg_lambda=0.08,reg_alpha=0.08,min_child_weight=1,learning_rate=0.06)
#GradientBoostingClassifier(warm_start=False,max_depth=2,min_samples_split=10,loss="exponential",n_estimators=100,subsample=0.7)
#LGBMClassifier(n_estimators=150,amx_depth=8,num_leaves=32,min_split_gain=[0.02],learning_rate=0.02,reg_alpha=0.04,min_child_weight=40,subsample=0.7)
s1.fit(x_train1,y_train1)
s2.fit(x_train2,y_train2)
s3.fit(x_train3,y_train3)
#显示模型效果
auc1 = metrics.roc_auc_score(y_test1,s1.predict_proba(x_test1)[:,1:])
auc2 = metrics.roc_auc_score(y_test2,s2.predict_proba(x_test2)[:,1:])
auc3 = metrics.roc_auc_score(y_test3,s3.predict_proba(x_test3)[:,1:])
print(auc1,auc2,auc3)
fs1=metrics.f1_score(y_test1,s1.predict(x_test1))
fs2=metrics.f1_score(y_test2,s2.predict(x_test2))
fs3=metrics.f1_score(y_test3,s3.predict(x_test3))
print(fs1,fs2,fs3)

#保存预测结果
res1=pd.DataFrame({'cust_id':test1['cust_id'],'pred_prob':s1.predict_proba(Test1)[:,1]})
res2=pd.DataFrame({'cust_id':test2['cust_id'],'pred_prob':s2.predict_proba(Test2)[:,1]})
res3=pd.DataFrame({'cust_id':test3['cust_id'],'pred_prob':s3.predict_proba(Test3)[:,1]})

with open("E:\比赛\DC新网杯\data\\result\\res01.csv",'w') as f1:
    res1.to_csv(f1,index=False)
with open("E:\比赛\DC新网杯\data\\result\\res01.csv",'a') as f2:
    res2.to_csv(f2,index=False,header=False)
with open("E:\比赛\DC新网杯\data\\result\\res01.csv",'a') as f3:
    res3.to_csv(f3,index=False,header=False)

'''
clf = LGBMClassifier(n_jobs=-1,
                     n_estimators=200,
                     learning_rate=0.01,
                     num_leaves=34,
                     colsample_bytree=0.9,
                     subsample=0.9,
                     max_depth=8,
                     reg_alpha=0.04,
                     reg_lambda=0.07,
                     min_split_gain=0.02,
                     min_child_weight=40,
                    )'''