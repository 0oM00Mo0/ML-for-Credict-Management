from sklearn.model_selection import train_test_split
import pandas as pd
#读取数据，不再赘述
data=pd.read_excel()
#方式1 先分离训练集和测试集，再分离x和y
train,test=train_test_split(data,test_size=0.5)
x_train=train.drop["Y"]
y_train=train["Y"]
x_test=test.drop["Y"]
y_test=test["Y"]
#方式2 先分离x和y，再分离测试集和验证集
x=data.drop(["Y"])
y=data["Y"]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.4,random_state=1)


'''
parameters:
test_size (float,int or None,default=0.25) 测试集的比例或个数
train_size  训练集的比例或个数
random_state 随机数种子
shuffle (boolean,default=True) 划分之前是否要打乱数据
stratify (array-like or none,default=None) 以数组为标签进行分层抽样

'''



