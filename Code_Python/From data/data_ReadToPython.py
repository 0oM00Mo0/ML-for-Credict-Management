#Method1
import pandas as pd
data=pd.read_csv("Type the path and name of your data")#对于 .csv文件
data=pd.read_excel("Type the path and name of your data")#对于 .xlsx文件
# 更多用法请参阅pandas文档
'''
Parameters:
sep= (str,default=",")   数据的分隔符，默认为逗号（因为csv是以逗号分隔的数据文件）
header= (int or list of ints) 表示数据开始的行数。若header=1，则表示数据是从第二行开始的，第一行是数据的列名称
names= (array-like,default=None) 指定每一列的列名。
index_col= (int or sequence) 用作索引的列编号或列名。如果给定一个序列，则表示有多列作为索引
na_values= 表示缺失值的值，例如，na_values=-99,则表示以-99为缺失值
'''
#更多参数可以参阅pandas文档