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
from sklearn.ensemble import AdaBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import Imputer
from sklearn.model_selection import StratifiedKFold
from sklearn.semi_supervised import label_propagation

def connect(a,b):
    c=np.row_stack((a,b))
    return c

f1=open("E:\比赛\DC新网杯\\test_all.csv")
f2=open("E:\比赛\DC新网杯\\train_xy.csv")
#读取训练数据
data=pd.read_excel("E:\比赛\DC新网杯\\train_xy1.xlsx",na_values=-99)
#"E:\比赛\DC新网杯\\train_xy.xlsx"
print(np.shape(data))
#"E:\比赛\DC新网杯\\test_all1.xlsx"
test=pd.read_excel("E:\比赛\DC新网杯\\test_all1.xlsx",na_values=-99)
#"E:\比赛\DC新网杯\\test_all.xlsx"
print(np.shape(test))
Test1=test.drop(['cust_id','cust_group'],axis=1)
x=data.iloc[:,3:]#删去标签和分组
y=data['y']
Test1_y=-np.ones((10000,1))


y=np.array([y]).T
print(np.shape(y))
print(np.shape(Test1_y))
#填充缺失值

imputer=Imputer(strategy="median")
x=imputer.fit_transform(x)
print(np.shape(Test1))

Test1=imputer.fit_transform(Test1)
print(np.shape(Test1))


#半监督学习
data_semi=connect(x,Test1)
y_semi=connect(y,Test1_y)
y_semi=y_semi.ravel()
semi_model=label_propagation.LabelSpreading(gamma=0.25,max_iter=2,alpha=0.5)
semi_model.fit(data_semi,y_semi)
y_semi=semi_model.transduction_
print(y_semi.sum())
'''
Test1_y=semi_model.predict(Test1)
Test1_y=np.array([Test1_y]).T
y=np.array([y]).T
print(np.shape(y))
print(Test1_y.sum())
print(np.shape(Test1_y))

data_semi=connect(x,Test1)
y_semi=connect(y,Test1_y)
'''

#归一化
'''
MMS1=preprocessing.MinMaxScaler()
MMS2=preprocessing.MinMaxScaler()
x=MMS1.fit_transform(x)
Test1=MMS2.fit_transform(Test1)
'''
#分离标签

#欠采样
'''
rus=RandomUnderSampler({0:14000})
x,y=rus.fit_sample(x,y)
#分离训练集，测试集
'''
print(np.shape(x))


#读取待预测数据


print(np.shape(Test1))

#筛选变量
'''
tree=XGBClassifier(missing=-99)
tree.fit(x,y)
model = SelectFromModel(tree, prefit=True,threshold="1*mean")
x = model.transform(x)

Test1=model.transform(Test1)
print(np.shape(x))
x=pd.DataFrame(x)

Test1=pd.DataFrame(Test1)

'''
print(np.shape(data_semi),np.shape(y_semi))
x_train,x_test,y_train,y_test = train_test_split(data_semi,y_semi,test_size=0.4,random_state=1)
'''
#网格搜索
if __name__ == '__main__':
    parameters = {'learning_rate':[0.055,0.065,0.07,0.075,0.08,0.085]}
    lgbm=XGBClassifier(n_estimators=90,gamma=0.03,max_depth=2,colsample_bylevel=0.8,colsample_bytree=0.5,subsample=0.7,min_child_weight=3,scale_pos_weight=1,booster="gbtree",reg_alpha=0.011,reg_lambda=0.02,learning_rate=0.08)
    clf=GridSearchCV(lgbm,parameters,cv=4 ,scoring='roc_auc',n_jobs=-1)#
    clf.fit(x,y)
    print("best:",clf.best_score_)
    print("using:",clf.best_params_)
'''
#定义，训练分类器

s1=XGBClassifier(booster="gbtree",subsample=0.7,n_estimators=140,max_depth=4,gamma=0.01,reg_lambda=0.08,reg_alpha=0.08,min_child_weight=1,learning_rate=0.06)
#RandomForestClassifier(max_features=57,n_estimators=500)
#XGBClassifier(booster="gbtree",subsample=0.7,n_estimators=140,max_depth=4,gamma=0.01,reg_lambda=0.08,reg_alpha=0.08,min_child_weight=1,learning_rate=0.06)
#LGBMClassifier(n_estimators=150,max_depth=2,num_leaves=32,min_split_gain=[0.02],learning_rate=0.02,reg_alpha=0.04,min_child_weight=40,subsample=0.7)
#GradientBoostingClassifier(warm_start=False,max_depth=2,min_samples_split=10,loss="exponential",n_estimators=100,subsample=0.7)不用了
#AdaBoostClassifier(algorithm="SAMME.R",learning_rate=0.04,n_estimators=300)不用了
s1.fit(x_train,y_train)
#s1.fit(x_train,y_train)
#显示模型效果
auc1 = metrics.roc_auc_score(y_test,s1.predict_proba(x_test)[:,1:])
print(auc1)
fs1=metrics.f1_score(y_test,s1.predict(x_test))
print(fs1)

#保存预测结果
res1=pd.DataFrame({'cust_id':test['cust_id'],'pred_prob':s1.predict_proba(Test1)[:,1]})
with open("E:\比赛\DC新网杯\data\\result\\res.csv",'w') as f1:
    res1.to_csv(f1,index=False)

