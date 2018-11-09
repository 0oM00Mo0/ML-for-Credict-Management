from sklearn.preprocessing import Imputer
import pandas as pd
#我们先假装读入了一个带有缺失值的数据
data=pandas.read_excel("")
#方法1：
imputer=Imputer(strategy="median")
data=imputer.fit_transform(data)
#齐活！
'''
parameters:
missing_values: (integer or "NaN",default="NaN") 用来表示缺失值的值。
strategy：(string,default="mean") mean:平均数， median：中位数，most_frequent:众数
'''

#方法2：
#data在被创建时就是DataFrame类型，因此我们可以直接使用pandas中自带的方法进行缺失值处理
data.dropna()
#删除有缺失值的样本
data.fillna(0)
#用0填入缺失的位置
data.fillna(method="ffill")
'''
method=  ffill 用缺失值前面的有效值从前往后填充
         bfill 用缺失值后面的有效值来从后往前填充
'''

