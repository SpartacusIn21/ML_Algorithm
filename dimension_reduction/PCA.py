#通过PCA进行降维处理，我们就可以同时获取SVM和决策树的优点：一方面，得到了和决策树一样简单的分类器，同时分类间隔和SVM一样好。
#前面我曾提到的第一个主成分就是从数据差异性最大（即方差最大）的方向提取出来，第二个主成分则来自于数据差异性次大的方向，并且该方向与第一个主成分方向正交。通过数据集的协方差矩阵及其特征值分析，我们就可以求得这些主成分的值。一旦得到了协方差矩阵的特征向量，我们就可以保留最大的N个值。这些特征向量也给出了N个最重要特征的真实结构。
#将数据转换成N个主成分的伪代码如下：
#(1)去除平均值
#(2)计算协方差矩阵
#(3)计算协方差矩阵的特征值和特征向量
#(4)将特征值从大到小排序
#(5)保留最上面的N个特征向量
#(6)将数据转换到上述N个特征向量构建的新空间中
#以上文字和下代码都来自《机器学习实战》一书

from numpy import *

def loadDataSet(fileName, delim='\t'):
	fr = open(fileName)
	#strip去除字符串前后空格
	stringArr = [line.strip().split(delim) for line in fr.readlines()]
	#将line中元素强制转换为float类型, map(function,iterable,...)
	dataArr = [map(float,line) for line in stringArr]
	return mat(dataArr)

#pca算法
def pca(dataMat, topNfeat=9999999):
	meanVals = mean(dataMat, axis=0)
	meanRemoved = dataMat - meanVals
	covMat = cov(meanRemoved, rowvar=0)
	eigVals,eigVects = linalg.eig(mat(covMat))
	eigValInd = argsort(eigVals)
	eigValInd = eigValInd[:-(topNfeat+1):-1]
	redEigVects = eigVects[:,eigValInd]
	lowDDataMat = meanRemoved * redEigVects
	reconMat = (lowDDataMat * redEigVects.T) + meanVals
	return lowDDataMat,reconMat
