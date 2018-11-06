from random import *
import numpy as np
def dataRead():
    a=open("D:\\mldata\\shuju.txt")
    data=[]
    rank=[]
    for line in a:
        l=line.split("\t")
        data.append(l[0:len(l)-1])
        rank.append(l[len(l)-1].replace("\n",""))
    for i in range(len(data)):
        for j in range(len(data[i])):
            data[i][j] = float(data[i][j])
        rank[i] = float(rank[i])
    return data,rank
def dataCheck():
    error=[]
    for i in range(len(rank)):
        xi=data[i]
        xi=np.array(xi)
        yi=rank[i]
        error.append(1-yi*(np.dot(w,xi.T)+b))
    return error
def dataTrain(w,b):
    maxe=np.amax(error)
    if maxe>0:
        i=np.where(error==np.max(error))
        i=choice(i[0])
        xi = data[i]
        xi = np.array(xi)
        yi = rank[i]
        w=(1-n)*w+n*C*yi*xi
        b=b+n*C*yi
    return w,b
data=[]
rank=[]
data,rank=dataRead()
data=np.array(data)
rank=np.array(rank)
print(rank)
w=np.empty((1,len(data[1])))
b=0
n=0.5
C=0.5
j=0
enum=0
xunhuannum=0
error=np.empty((1,len(rank)))
dataRead()
for i in range(100):
    xunhuannum = xunhuannum+1
    error=dataCheck()
    if max(error)<=0:
        break
    else:
        w,b=dataTrain(w,b)
for i in range(len(error)):
    if error[i]>0:
        enum = enum+1
print("w=",w,"    b=",b)
print("训练集有{0}个样本，经过{1}次循环，分类正确率为{2}".format(len(data),xunhuannum,1-enum/len(data)))
print(data[:,2])