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
f1=open("E:\比赛\DC新网杯\\test_all.csv")
f2=open("E:\比赛\DC新网杯\\train_xy.csv")
#读取训练数据
data=pd.read_excel("E:\比赛\DC新网杯\\train_xy.xlsx",na_values=-99)
print(np.shape(data))
#"E:\比赛\DC新网杯\\test_all1.xlsx"
test=pd.read_excel("E:\比赛\DC新网杯\\test_all.xlsx",na_values=-99)
print(np.shape(test))
Test1=test.drop(['cust_id','cust_group'],axis=1)
x=data.iloc[:,3:]#删去标签和分组
y=data['y']

#填充缺失值

imputer=Imputer(strategy="median")
x=imputer.fit_transform(x)
print(np.shape(Test1))

Test1=imputer.fit_transform(Test1)
print(np.shape(Test1))

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

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.01,random_state=1)

#训练GBDT
clf_feature=LGBMClassifier(n_estimators=150,max_depth=4,num_leaves=32,min_split_gain=[0.02],learning_rate=0.02,reg_alpha=0.04,min_child_weight=40,subsample=0.7)
clf_feature.fit(x_train,y_train)

#得到新特征
x_train=clf_feature.predict(x_train,pred_leaf=True)
print(np.shape(x_train))
x_test=clf_feature.predict(x_test,pred_leaf=True)
Test1=clf_feature.predict(Test1,pred_leaf=True)
print(x_train[0])
#独热编码
ohe=preprocessing.OneHotEncoder()
x_test=ohe.fit_transform(x_test).toarray()
x_train=ohe.fit_transform(x_train).toarray()
Test1=ohe.fit_transform(Test1).toarray()
print(np.shape(x_train))

'''
if __name__ == '__main__':
    parameters = {'penalty':["l1"],"C":[0.8,0.9,1],"class_weight":[{0:10},{0:5},{0:2},{0:1}]}
    lr = LogisticRegression(penalty="l2")
    clf=GridSearchCV(lr,parameters,cv=3 ,scoring='roc_auc',n_jobs=-1)#
    clf.fit(x,y)
    print("best:",clf.best_score_)
    print("using:",clf.best_params_)
'''
lr=XGBClassifier(booster="gbtree",subsample=0.7,n_estimators=140,max_depth=4,gamma=0.01,reg_lambda=0.08,reg_alpha=0.08,min_child_weight=1,learning_rate=0.06)
lr.fit(x_train,y_train)






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




#s1.fit(x_train,y_train)
#显示模型效果
auc1 = metrics.roc_auc_score(y_test,lr.predict_proba(x_test)[:,1:])
print(auc1)
fs1=metrics.f1_score(y_test,lr.predict(x_test))
print(fs1)

#保存预测结果
res1=pd.DataFrame({'cust_id':test['cust_id'],'pred_prob':lr.predict_proba(Test1)[:,1]})
with open("E:\比赛\DC新网杯\data\\result\\res.csv",'w') as f1:
    res1.to_csv(f1,index=False)
