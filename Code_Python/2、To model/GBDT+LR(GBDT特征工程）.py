from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
import numpy as np
import pandas as pd
'''
利用GBDT做特征工程，简单来讲就是不使用GBDT的分类（或预测）结果，而是利用GBDT的特点，把每个样本在每棵数上的
位置作为样本的特征，从而实现利用GBDT模型自动提取样本的特征
比如，现有一个由10棵树组成的GBDT模型，每棵树有5个叶结点。那么，带入数据计算后，输出每个样本在每棵树上的结点
位置，得到一个 10*n （n为样本个数）的矩阵。再对这个矩阵进行独热编码，直接带入另一个模型进行计算，或者与原数
据结合，带入另一个模型计算
但是结合我上一次比赛的经验来看，第二次带入的模型是逻辑回归时，这方法的效果比单独的逻辑要好，但比GBDT要差。
'''
data=pd.read_excel("")
X=data.drpo(["Y"])
Y=data["Y"]
clf_feature=LGBMClassifier()
#有人会说，这不是lightgbm吗，但是熟悉lightgbm的人就会知道，它的默认boosting_type="GBDT",这时候它就是一个GBDT模型。
#又有人会问了，那你为啥不用sklearn库里的GBDT呢？ 这是因为我在翻阅他的代码时，并没有发现输出样本结点位置的选项。
#如果有人知道，那么请告诉我一下
clf_feature.fit(X,Y)#训练模型

#得到新特征
X_new=clf_feature.predict(X,pred_leaf=True)
#此时得到的时原始特征，还需要独热编码才能带入python模型使用

#独热编码
ohe=preprocessing.OneHotEncoder()
X_new=ohe.fit_transform(X_new).toarray()

#新特征与原特征结合（这一步不要也可以）
X=np.column_stack((X,X_new))
#这里使用了numpy的column_stack方法，但是pandas也有相应的添加列的方法，但只能一列一列加，太麻烦了

#将新数据带入新模型进行训练，这里使用逻辑回归
LR=LogisticRegression()
LR.fit(X,Y)
