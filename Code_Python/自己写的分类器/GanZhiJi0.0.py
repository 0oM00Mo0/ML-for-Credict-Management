from random import *
import numpy
class gzj():
    def __init__(self,position,value,w,b,n):
        self.pos=position
        self.val=value
        self.w=w
        self.b=b
        self.error=[]
        self.n=n
        #初始化函数，得到样本标志，类型，w，b，学习率
    def check(self):
        self.error=[]
        for i in range(len(self.pos)):
            if self.val[i]*(self.w[0]*self.pos[i][0]+self.w[1]*self.pos[i][1]+self.b)<=1:
                self.error.append(i)
#检查是否存在误分类，并将误分类的变量编号存在error中
    def update(self):
        i=choice(self.error)
        self.w[0]= self.w[0] + self.n * self.val[i] * self.pos[i][0]
        self.w[1] = self.w[1] + self.n * self.val[i] * self.pos[i][1]
        self.b = self.b + self.n * self.val[i]
#更新w和b的值
    def print(self):
        print(self.w,self.b)
#显示w和b

value=[1,1,1,1,1,-1,-1,-1,-1,-1]
position=[[1,1],[1,2],[1.2,0.5],[0.5,3],[2,0.03],[2,1],[2,2],[1.5,3],[3,0.5],[1,10]]
w=[0,0]
b=0
n=0.5
gzj=gzj(position,value,w,b,n)
i=0
for j in range(1000):
    gzj.check()
    i=i+1
    if len(gzj.error)>0:
        gzj.update()
    else:
       break



gzj.print()
print(gzj.error)
print("误分类数=",len(gzj.error),"变量个数=",len(gzj.pos),"循环次数=",i)

