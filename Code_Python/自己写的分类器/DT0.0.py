import numpy
import string
from operator import itemgetter
import math
def dataRead():
    a=open("D://mldata//DTdata1.txt")
    data=[]
    label=[]
    n=0
    for line in a:
        if n==0:
            l=line.split("\t")
            l[-1]=l[-1].replace("\n","")
            label.extend(l)
            n=n+1
        else:
            l=line.split("\t")
            l[-1]=l[-1].replace("\n","")
            data.append(l[0:len(l)])
    return data,label
#读入数据的函数
def H(list):
    h=0.0
    listlen=len(list)
    temp={ }
    for i in list:
        temp[i]=temp.get(i,0)+1
    for key in temp:
        prob=(temp[key])/listlen
        h=h-prob*math.log(prob,2)
    return h
#求一个数列的熵的函数
def dataSplit(data,row,value):
    s_data=[]
    for line in data:
        if line[row]==value:
            newdata=line[0:row]
            newdata.extend(line[row+1:len(line)])
            s_data.append(newdata)
    return s_data
#拆分数据,新数据集中不含用于分类的指标
def choseBestSplit(data):
    rank = [x[-1] for x in data]
    Hy=H(rank)
    bestGain=0
    split=0
    for i in range(len(data[0])-1):
        HA=0
        numlist=[elm[i] for elm in data]
        numset=set(numlist)#找出特征的不同取值
        for n in numset:
            newdata=dataSplit(data,i,n)
            prob=len(newdata)/len(data)
            ndata=[x[-1] for x in newdata]
            HA=HA-prob*H(ndata)
        infoGain=Hy-HA
        if infoGain>bestGain:
            bestGain=infoGain
            split=i
    return split
#选择最合适分类的指标
def major(rank):
    count={}
    for i in rank:
        count[i]=count.get(i,0)+1
    sortcount=sorted(count.iteritems(),key=itemgetter(1),reverse=True)
    return sortcount[0][0]
#少数服从多数

def creatTree(data,label):
    rank=[x[-1] for x in data]
    if rank.count(rank[0])==len(rank):
        return rank[0]
    if len(data[0])==1:
        return major(rank)
    bestFeature=choseBestSplit(data)
    bestFlabel=label[bestFeature]
    DT={bestFlabel:{}}
    del(label[bestFeature])
    bestline=[i[bestFeature] for i in data]
    uniqvals=set(bestline)
    for i in uniqvals:
        nlabels=label[:]
        DT[bestFlabel][i]=creatTree(dataSplit(data,bestFeature,i),nlabels)
    return DT
#生成决策树
data=[]
label=[]

data,label=dataRead()
a=creatTree(data,label)
print(a)



