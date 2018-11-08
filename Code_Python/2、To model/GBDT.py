from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
#读入数据
data=pd.read_excel("")
x=data.drop(["Y"])
y=data["Y"]

GBDT=GradientBoostingClassifier()
'''
parameters:
loss (deviance, exponential,default=deviance) 损失函数 等于“exponential”时。模型退化为Adaboost
learning_rate (float,default=0.1) 学习速率
n_estimators (int,default=100) 模型基学习器的个数
subsample (float,default=1.0,<=1) 欠采样的比例
criterion (string,default="friedman_mse") 树分裂时的准则
min_samples_split (int,default=2) 指定在下一步分裂中，分裂时结点中包含的最小样本数。如果样本数小于这个数，
将不再分裂子结点
min_samples_leaf (int，default=1) 叶结点的中包含的最小样本个数
min_weight_fraction_leaf （int，default=0） 叶结点中最小的权重和
max_depth （int，default=3） 每个基学习器中最大的深度
random_state （int） 随机数种子
max_features （int，default=None） 每个学习器中可使用的最大变量个数

Attributes：
n_estimators_  ：模型学习器的个数
oob_improvement_  ：模型损失函数在 out-of-bags 样本上的提升
feature_importances_ ：模型对变量重要程度的打分。分数越高，代表变量越重要
'''
GBDT.fit(x,y)
score=GBDT.score(x,y)#对模型的准确率进行打分
print(score)

