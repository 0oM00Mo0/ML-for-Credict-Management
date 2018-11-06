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
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from keras.models import Sequential
from keras.layers import Dense
from xgboost import  XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import Imputer
from sklearn.model_selection import StratifiedKFold
from  mlxtend.classifier import StackingCVClassifier
from  mlxtend.classifier import StackingClassifier
f1=open("E:\比赛\DC新网杯\\shuju1.csv")
f2=open("E:\比赛\DC新网杯\\testdata.csv")
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
imputer=Imputer(strategy="mean")
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


#欠采样
#rus=RandomUnderSampler({0:10000})
#x,y=rus.fit_sample(x,y)
#分离训练集，测试集

print(np.shape(x))

#f2=open("E:\比赛\DC新网杯\\test_all.csv")
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

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.005,random_state=1)
x_train=np.array(x_train)
x_test=np.array(x_test)
y_train=np.array(y_train)
y_test=np.array(y_test)
print(y_train)
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

s1=XGBClassifier(n_estimators=200,gamma=0.03,max_depth=2,colsample_bylevel=0.8,colsample_bytree=0.8,subsample=0.9,min_child_weight=3,scale_pos_weight=1,booster="gbtree",reg_alpha=0.011,reg_lambda=0.02,learning_rate=0.08)
s2=LGBMClassifier(n_estimators=150,max_depth=2,num_leaves=32,min_split_gain=[0.02],learning_rate=0.02,reg_alpha=0.04,min_child_weight=40,subsample=0.9)
s3=RandomForestClassifier(max_depth=10)
s4=GradientBoostingClassifier(warm_start=False,max_depth=2,min_samples_split=10,loss="exponential",n_estimators=100,subsample=0.9)
s5=AdaBoostClassifier(algorithm="SAMME.R",learning_rate=0.04,n_estimators=300)
s6=XGBClassifier(n_estimators=200,gamma=0.03,max_depth=5,colsample_bylevel=0.8,colsample_bytree=0.8,subsample=0.9,min_child_weight=1,booster="dart",reg_alpha=0.011,reg_lambda=0.02,learning_rate=0.08)
s8=AdaBoostClassifier(algorithm="SAMME",learning_rate=0.04,n_estimators=300)
s9=LGBMClassifier(boosting_type="goss",n_estimators=200,max_depth=8,num_leaves=34,min_split_gain=0.02,learning_rate=0.02,reg_alpha=0.07,min_child_weight=40,subsample=0.9)
S6=XGBClassifier(n_estimators=200,gamma=0.03,max_depth=2,colsample_bylevel=0.8,colsample_bytree=0.8,subsample=0.7,min_child_weight=3,scale_pos_weight=1,booster="gbtree",reg_alpha=0.011,reg_lambda=0.02,learning_rate=0.08)

S7=LogisticRegression()
S8=svm.SVC(probability=True)
stacking=StackingClassifier(classifiers=[s1,s2,s3,s4,s5],meta_classifier=S6,verbose=2,use_probas=True,use_features_in_secondary=True)

#RandomForestClassifier(max_features=57,n_estimators=500)
#XGBClassifier(booster="gbtree",subsample=0.7,n_estimators=140,max_depth=4,gamma=0.01,reg_lambda=0.08,reg_alpha=0.08,min_child_weight=1,learning_rate=0.06)
#LGBMClassifier(n_estimators=150,max_depth=2,num_leaves=32,min_split_gain=[0.02],learning_rate=0.02,reg_alpha=0.04,min_child_weight=40,subsample=0.7)
#GradientBoostingClassifier(warm_start=False,max_depth=2,min_samples_split=10,loss="exponential",n_estimators=100,subsample=0.7)不用了
#AdaBoostClassifier(algorithm="SAMME.R",learning_rate=0.04,n_estimators=300)不用了
stacking.fit(x_train,y_train)
#显示模型效果
auc1 = metrics.roc_auc_score(y_test,stacking.predict_proba(x_test)[:,1:])
print(auc1)
fs1=metrics.f1_score(y_test,stacking.predict(x_test))
print(fs1)

#保存预测结果
res1=pd.DataFrame({'cust_id':test['cust_id'],'pred_prob':stacking.predict_proba(Test1)[:,1]})
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