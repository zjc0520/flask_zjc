"""计算信息熵"""
from math import log

dataSet=[
            [1,1,'yes'],
            [1,1,'yes'],
            [1,0,'no'],
            [0,1,'no'],
            [0,1,'no']]
lables=['no surfacing','flippers']


def calcShannoEnt(dataSet):

    numEntries=len(dataSet)
    labelCounts={}
    for featVec in dataSet:
       currentLable=featVec[-1]
       # 取最后一列作为数组的分类标签
       if currentLable not in labelCounts.keys():
           labelCounts[currentLable]=0
       else:
           labelCounts[currentLable]=1
    shannonEnt=0.0
    for key in  labelCounts:
        prob=float(labelCounts[key])/numEntries
        # 求某一项的分类（P(X)）
        shannonEnt-=prob*log(prob,2)
        # 求熵∑[i=1,n]P(xi)log₂P(xi)
    # return  shannonEnt
    print(shannonEnt)
# dataSet=[
#         [1,1,'yes'],
#         [1,1,'yes'],
#         [1,0,'no'],
#         [0,1,'no'],
#         [0,1,'no']]
# lables=['no surfacing','flippers']
#创建测试数
calcShannoEnt(dataSet)

#按照给定特征划分数据集
#输入参数：待划分的数据集、划分数据集的特征、需要返回的特征的值
def splitDataSet(dataSet,axis,value):
    retDataSet=[]
    for featVec in dataSet:
        if featVec[axis]==value:
            reducedFeatVac=featVec[:axis]
            reducedFeatVac.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVac)
    return retDataSet
testdata,testlables=dataSet,lables
print(testdata)
print(splitDataSet(testdata,0,1))
print(splitDataSet(testdata,0,0))

# a=[1,2,3]
# b=[4,5,6]
# a.append(b)
# a.extend(b)
