import pandas as pd
#假设我们有一个DataFrame数据df，一个numpy的矩阵数据num，两个list：lable和Y

#对于DataFrame的数据，可以直接写入文件
df.to_excel("Path of file")

#其他数据要转换成DataFrame的形式
num_df=pd.DataFrame(num)
num_df.to_excel("")

#list同上。如果想把两个list写入同一个文件：
list=pd.DataFrame({"Label":label,"Y":Y})
list.to_excel("")
#写入后label和y数据对应的列名就会变成“Label”和“Y”

