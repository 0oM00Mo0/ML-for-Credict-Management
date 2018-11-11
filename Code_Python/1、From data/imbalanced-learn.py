#使用imbalanceed-learn库来进行 欠采样、过采样
#但相当一部分模型都可以通过 class_weight 参数来调整每种分类下样本的权重。
from imblearn.over_sampling import SMOTE,RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

#随机欠采样：在样本中，如果某一分类的样本明显多于另一类，则可以使用欠采样来减少多数类样本的个数
rus=RandomUnderSampler({0:4000,1:3000})#“0”,“1”为多数样本，4000为采样个数，可以根据实际情况调整。参数中没有提及的分类则不会改动
x,y=rus.fit_resample(x,y)
#这时x，y就被替换为欠采样后的数据
#如果欠采样的比例过小，则可能造成欠拟合


#过采样
#1.随机过采样：某分类的样本明显偏少，则可以使用它对少数样本进行过采样。本质上是对少数类样本的随机重复使用
ros=RandomOverSampler({2:3000,3:4000})#同上
x,y=ros.fit_resample(x,y)

#2.少数类合成过采样 Synthetic Minority Over-Sampling Technique 我看过用它写的一篇论文。写得还行
smote=SMOTE(kind='regular',k_neighbors=5,ratio={2:3000,3:4000})
'''
parameters:
kind: ('regular', 'borderline1', 'borderline2' or  'svm',default='regular') 样本合成的方式，不懂就百度SMOTE，有真相
k_neighbors: (int,default=5) 样本合成时使用的邻近样本数
ratio: 样本合成的数量或比例
svm_estimator： kind=svm时才需要设置，传入一个sklearn的模型就行
'''
#如果过采样比例过大，则可能造成“过拟合”，即把少数类中的偶然情况当作一般规律


