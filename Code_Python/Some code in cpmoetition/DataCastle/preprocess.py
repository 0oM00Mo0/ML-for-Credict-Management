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
#f1=open("E:\比赛\DC新网杯\\train_xy.csv")
#读取训练数据
test=pd.read_csv(f1,na_values=-99)
data=pd.read_csv(f2,na_values=-99)
Test1=test.drop(['cust_id','cust_group'],axis=1)
x=data.iloc[:,3:]#删去标签和分组
#填充缺失值
'''
imputer=Imputer(strategy="mean")
x=imputer.fit_transform(x)
imputer=Imputer(strategy="mean")
Test1=imputer.fit_transform(Test1)
'''
#分离标签
y=data['y']
#欠采样
#rus=RandomUnderSampler({0:10000})
#x,y=rus.fit_sample(x,y)
#分离训练集，测试集

print(np.shape(x))

#f2=open("E:\比赛\DC新网杯\\test_all.csv")
#读取待预测数据


print(np.shape(Test1))

#筛选变量
tree=[]
for i in [0,1,2]:
    tree.append(LGBMClassifier())
    tree[i].fit(x,y)


print(tree[0].feature_importances_+tree[1].feature_importances_+tree[2].feature_importances_)



x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.005,random_state=1)
'''
#网格搜索
if __name__ == '__main__':
    parameters = {'n_estimators':[300,250],'learning_rate':[0.05,0.06]}
    lgbm=GradientBoostingClassifier(warm_start=False,min_samples_split=10,subsample=0.7,loss="exponential",learning_rate=0.05,max_depth=2,n_estimators=300)
    clf=GridSearchCV(lgbm,parameters,cv=4 ,scoring='roc_auc',n_jobs=-1)#
    clf.fit(x,y)
    print("best:",clf.best_score_)
    print("using:",clf.best_params_)
'''
#定义，训练分类器

s1=XGBClassifier(booster="gbtree",subsample=0.7,n_estimators=140,max_depth=4,gamma=0.01,reg_lambda=0.08,reg_alpha=0.08,min_child_weight=1,learning_rate=0.06,missing=-99)

#AdaBoostClassifier(algorithm="SAMME.R",learning_rate=0.04,n_estimators=300)
#XGBClassifier(booster="gbtree",subsample=0.7,n_estimators=140,max_depth=4,gamma=0.01,reg_lambda=0.08,reg_alpha=0.08,min_child_weight=1,learning_rate=0.06)
#GradientBoostingClassifier(warm_start=False,max_depth=2,min_samples_split=10,loss="exponential",n_estimators=100,subsample=0.7)
#LGBMClassifier(n_estimators=150,max_depth=8,num_leaves=32,min_split_gain=[0.02],learning_rate=0.02,reg_alpha=0.04,min_child_weight=40,subsample=0.7)
s1.fit(x_train,y_train)
#显示模型效果
auc1 = metrics.roc_auc_score(y_test,s1.predict_proba(x_test)[:,1:])
print(auc1)
fs1=metrics.f1_score(y_test,s1.predict(x_test))
print(fs1)

#保存预测结果
res1=pd.DataFrame({'cust_id':test['cust_id'],'pred_prob':s1.predict_proba(Test1)[:,1]})
with open("E:\比赛\DC新网杯\data\\result\\res.csv",'w') as f1:
    res1.to_csv(f1,index=False)


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