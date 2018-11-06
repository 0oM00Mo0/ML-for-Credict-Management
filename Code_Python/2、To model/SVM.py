#This is a simple exampl to show how to use SVM in aklearn
#These are tools we need
from sklearn.svm import SVC
import pandas as pd
data=pd.read("Type the path and name of your data")
Y=data["Y"]#get the label of your data
X=data.drop(["Y","ID"])#get your independent variable,so we delet Y and ID
SVM=SVC()
#SVR(),回归预测的SVM模型
'''
Parameters:
C= (float,default=1.0) SVM模型对于误分类样本的惩罚度（不懂看书）
kernel= (string,default="rbf") SVM的核函数，包括"rbf"(高斯核函数),"poly"(多项式核函数),"linear"(线性核函数),
"sigmoid"(sigmoid核函数)
degree= (int,default=3),kernel="poly"时，指定多项式的次数
gamma= (float,default="auto"),kernel="rbf"时，指定 1/(N*std(X))
probability= (boolean,default=False),等于True时，进行概率计算。之后可以使用predict_proba()函数输出样本分到每
个类的概率
class_weight= (dict or "balanced"),当样本不均衡时，使用此方法调整每个类中样本的权重。如{"1":5,"0":1},表示分类
为1的样本权重为5，分类为0的样本权重为1
Attributes：
support_ :支持向量的目录
support_vectors_:支持向量
n_support_:支持向量的个数
intercept_:偏移量b
'''
SVM.fit(X,Y)#代入数据让模型来学习
Y_pred=SVM.predict(X)#输出SVM对样本的预测结果

