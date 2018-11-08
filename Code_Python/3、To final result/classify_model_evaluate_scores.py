import pandas as pd
from sklearn.tree import DecisionTreeClassifier
data=pd.read("Type the path and name of your data")
Y=data["Y"]#get the label of your data
X=data.drop(["Y","ID"])#get your independent variable,so we delet Y and ID
clf=DecisionTreeClassifier()
clf.fit(X,Y)
Y_pred=clf.predict(X)
#准确率=(TP+TN)/(P+N)
score=clf.score(X,Y)
#查准率和召回率
'''
查准率（Precision）             P=TP/(TP+FP)
召回率（Recall,就是敏感度）     R=TP/(TP+FN)=TP/P
F1_score=2*PR/(P+R)
'''
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
P=precision_score(Y,Y_pred)
R=recall_score(Y,Y_pred)
F1=f1_score(Y,Y_pred)
#ROC曲线和AUC
#ROC曲线作图部分参考了强力网友的代码，因为自己还不太会用Matplotlib （黑脸）
Y_proba=clf.predict_proba(X)#注意此处为预测分类的概率
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import matplotlib as plt
#auc
auc=roc_auc_score(Y,Y_proba[:,1])#第一列是预测为0的概率，第二列是预测为1的概率，我们取第二列
#绘制ROC曲线
#先获得每个cut_off下对应的fpr，tpr
fpr,tpr,cut_off=roc_curve(Y,Y_proba[:,1])
#cut_off：区分0，1分类的临界值  fpr：对应的cut_off值下的累计坏样本率  tpr：累计好样本率
#fpr=TP/P,也就是敏感度，召回率  tpr=1-TN/N，也就是1-特异度
#绘图
plt.figure()#创建图像
lw = 2
plt.figure(figsize=(10,10))
plt.plot(fpr, tpr, color='darkorange',lw=lw, label='ROC curve (area = %0.2f)' )#以fpr为纵轴，tpr为横轴画图
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
#规定坐标轴范围
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')#横轴标题
plt.ylabel('True Positive Rate')#纵轴标题
plt.title('Receiver operating characteristic example')#图表标题
plt.legend(loc="lower right")
plt.show()#显示图表

