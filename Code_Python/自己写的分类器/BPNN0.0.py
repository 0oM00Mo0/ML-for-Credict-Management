import numpy as np
import random
def tan(x):
    return np.tanh(x)
def d_tan(x):
    return 1 -  np.tanh(x) * np.tanh(x)
# sigmod函数
def logistic(x):
    return 1 / (1 + np.exp(-x))
# sigmod函数的导数
def d_logistic(x):
    return logistic(x) * (1 - logistic(x))
#定义两种常见激活函数及其导函数
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
#定义读入数据的函数
#输入层
class inLayer():
    Input=[]
    output=[]
    numin=0
    numout=0
    delta=[]
    w=[]
    def __init__(self,numin,numout,active="tan"):#初始化
        self.numin=numin
        self.numout=numout
        self.w=  np.random.rand(numin, numout)
        if active=='tan':
            self.activation=tan#定义激活函数
            self.dactivation=d_tan#激活函数的导函数
        if active=='logistic':
            self.activation=logistic
            self.dactivation=d_logistic

    def inPut(self,a):#输入层传入数据
        self.Input=a
        self.output=self.activation(self.Input)

    def update(self,n):
        n.delta=np.array(n.delta)
        #print(np.shape(n.delta))
        self.w = np.array(self.w)
        #print(np.shape( self.w))
        self.output=np.array([self.output])
        #print(np.shape(self.output))
        self.w = self.w - n.n * np.matmul(self.output.T , n.delta)
        #print(np.shape(self.w))
#中间层
class midLayer():
    Input = []
    output = []
    w = []
    numin = 0
    numout = 0
    delta = []
    def __init__(self, numin, numout, n, active="tan"):  # 初始化
        self.numin = numin
        self.numout = numout
        self.w = np.random.rand(numin, numout)
        self.n = n
        if active == 'tan':
            self.activation = tan  # 定义激活函数
            self.dactivation = d_tan  # 激活函数的导函数
        if active == 'logistic':
            self.activation = logistic
            self.dactivation = d_logistic

    def inPut(self, a):  # 输入层传入数据
        self.Input=[]
        tw = np.array(a.w)

        self.Input = np.matmul(a.output ,tw)

    def outPut(self):
        self.output=[]
        self.output = self.activation(self.Input)


    def Delta(self, a):  # 根据上一层梯度计算这一层梯度
        self.delta=[]
        self.w = np.array(self.w)
        a.delta=np.array(a.delta)

        self.Input=np.array([self.Input])

        self.delta=np.matmul(a.delta,self.w.T)*self.dactivation(self.Input)

    def update(self,n):
        self.w=np.array(self.w)
        n.delta=np.array([n.delta])
        self.output=np.array([self.output])
       # print(np.shape(self.w))
        #print(np.shape(self.output))
        self.w = self.w-np.dot(self.n,np.matmul(self.output.T,n.delta))
#输出层
class outLayer():
    Input = []
    output = []
    numin = 0
    numout = 0
    delta = []

    def __init__(self,  n, active="tan"):  # 初始化
        self.n = n
        if active == 'tan':
            self.activation = tan  # 定义激活函数
            self.dactivation = d_tan  # 激活函数的导函数
        if active == 'logistic':
            self.activation = logistic
            self.dactivation = d_logistic

    def inPut(self, a):  # 输入层传入数据
        tw=np.array(a.w)
        self.Input = np.matmul(a.output, tw)

    def outPut(self):
        self.output = self.activation(self.Input)

    def Delta(self,y):  # 根据上一层梯度计算这一层梯度
        self.delta = self.activation(self.Input)-y
data=[]
rank=[]
data,rank=dataRead()
data=np.array(data)
rank=np.array(rank)
l1=4
l2=3
l3=1
lin=inLayer(l1,l2)
lmid=midLayer(l2,l3,1)
lout=outLayer(1)
mis=0
count=0
for i in range(1000):
    a=random.randint(0,len(data)-1)
    lin.inPut(data[a])
    lmid.inPut(lin)
    lmid.outPut()
    lout.inPut(lmid)
    lout.outPut()
    lout.Delta(rank[a])
    lmid.update(lout)
    lmid.Delta(lout)
    lin.update(lmid)
    count=count+1
def pre(a):
    lin.inPut(a)
    lmid.inPut(lin)
    lout.inPut(lmid)
    lout.outPut()
    if lout.output<0:
        return -1
    else:
        return 1
for i in range(len(data)):
    if pre(data[i])==rank[i]:
        continue
    else:
        mis=mis+1

print("经过{0}次循环，训练集的正确率为{1}".format(count,1-mis/len(data)))