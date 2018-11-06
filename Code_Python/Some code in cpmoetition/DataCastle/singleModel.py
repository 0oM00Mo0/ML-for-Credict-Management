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
def createModel(n):
    model=Sequential()
    model.add(Dense(6,input_dim=n,activation="relu"))#s输入层为4，隐藏层为3
    model.add(Dense(12,activation='sigmoid'))
    model.add(Dense(1,activation="sigmoid"))#输出层为1
    model.compile(loss="binary_crossentropy",optimizer="adam",metrics=['accuracy'])#编译模型
    return model

data1=pd.read_excel("E:\比赛\DC新网杯\data\\trainxy\group1.xlsx",header=None)
data2=pd.read_excel("E:\比赛\DC新网杯\data\\trainxy\group2.xlsx",header=None)
data3=pd.read_excel("E:\比赛\DC新网杯\data\\trainxy\group3.xlsx",header=None)
'''
MMS=preprocessing.MinMaxScaler()
data1=MMS.fit_transform(data1.iloc[:,3:])
data2=MMS.fit_transform(data2.iloc[:,3:])
data3=MMS.fit_transform(data3.iloc[:,3:])
'''
rus=RandomUnderSampler({0:2000})
data_resd1,rank_resed1=rus.fit_sample(data1.iloc[:,4:],data1.iloc[:,3])
data_resd2,rank_resed2=rus.fit_sample(data2.iloc[:,4:],data2.iloc[:,3])
data_resd3,rank_resed3=rus.fit_sample(data3.iloc[:,4:],data3.iloc[:,3])

tree1=RandomForestClassifier()
tree2=RandomForestClassifier()
tree3=RandomForestClassifier()
tree1.fit(data_resd1,rank_resed1)
tree2.fit(data_resd2,rank_resed2)
tree3.fit(data_resd3,rank_resed3)
model1 = SelectFromModel(tree1, prefit=True)
model2 = SelectFromModel(tree2, prefit=True)
model3 = SelectFromModel(tree3, prefit=True)
data_resd1 = model1.transform(data_resd1)
data_resd2 = model2.transform(data_resd2)
data_resd3 = model3.transform(data_resd3)
data_resd1,Xtset1,rank_resed1,Ytest1=train_test_split(data_resd1,rank_resed1,test_size=0.5,random_state=2)
data_resd2,Xtset2,rank_resed2,Ytest2=train_test_split(data_resd2,rank_resed2,test_size=0.5,random_state=2)
data_resd3,Xtset3,rank_resed3,Ytest3=train_test_split(data_resd3,rank_resed3,test_size=0.5,random_state=2)
print(np.shape(data_resd1))
'''
s1=svm.SVC(C=2,kernel="rbf",gamma=1,class_weight="balanced",probability=True)
s2=svm.SVC(C=2,kernel="rbf",gamma=1,class_weight="balanced",probability=True)
s3=svm.SVC(C=2,kernel="rbf",gamma=1,class_weight="balanced",probability=True)#balance解决数据不平衡问题'''
s1=RandomForestClassifier(min_samples_leaf=10)
s2=RandomForestClassifier(min_samples_leaf=10)
s3=RandomForestClassifier(min_samples_leaf=10)
s1.fit(data_resd1,rank_resed1)
s2.fit(data_resd2,rank_resed2)
s3.fit(data_resd3,rank_resed3)

test1=pd.read_excel("E:\比赛\DC新网杯\data\\test\\1.xlsx",header=None)
test2=pd.read_excel("E:\比赛\DC新网杯\data\\test\\group2.xlsx",header=None)
test3=pd.read_excel("E:\比赛\DC新网杯\data\\test\\group3.xlsx",header=None)
test1=model1.transform(test1.iloc[:,3:])
test2=model2.transform(test2.iloc[:,3:])
test3=model3.transform(test3.iloc[:,3:])
'''
test1=MMS.fit_transform(test1)
test2=MMS.fit_transform(test2)
test3=MMS.fit_transform(test3)
'''
auc1 = metrics.roc_auc_score(Ytest1,s1.predict_proba(Xtset1)[:,1])
auc2 = metrics.roc_auc_score(Ytest2,s2.predict_proba(Xtset2)[:,1])
auc3 = metrics.roc_auc_score(Ytest3,s3.predict_proba(Xtset3)[:,1])
print(auc1,auc2,auc3)

res1=pd.DataFrame((s1.predict_proba(test1)))
print(np.shape(res1))
res2=pd.DataFrame((s2.predict_proba(test2)))
res3=pd.DataFrame((s3.predict_proba(test3)))
#print(res1,res2,res3)
fs1=metrics.f1_score(Ytest1,s1.predict(Xtset1))
fs2=metrics.f1_score(Ytest2,s2.predict(Xtset2))
fs3=metrics.f1_score(Ytest3,s3.predict(Xtset3))
print(fs1,fs2,fs3)
'''
with open("E:\比赛\DC新网杯\data\\result\\res.csv",'w') as f1:
    res1.to_csv(f1,index=False,header=False)
with open("E:\比赛\DC新网杯\data\\result\\res.csv",'a') as f2:
    res2.to_csv(f2,index=False,header=False)
with open("E:\比赛\DC新网杯\data\\result\\res.csv",'a') as f3:
    res3.to_csv(f3,index=False,header=False)
'''