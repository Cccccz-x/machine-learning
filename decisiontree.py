import numpy as np
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import graphviz
import pandas as pd

name = ['no', 'age', 'job', 'house', 'debet', 'y']
data = pd.read_csv("decisiontree_data.csv", header=None, sep=',', names=name)
data = data.iloc[1:, 1:]
print(data)


def Ent(dataset):
    n = dataset.shape[0]
    cls = dataset.iloc[:, -1].value_counts()
    p = cls / n
    return np.sum(-p * np.log2(p))


def bestsplit(dataset):
    baseEnt = Ent(dataset)
    bestGain = 0
    axis = -1
    for i in range(dataset.shape[1] - 1):
        cls = dataset.iloc[:, i].value_counts().index
        ents = 0

        for j in cls:
            childset = dataset[dataset.iloc[:, i] == j]
            ent = Ent(childset)
            ents += ent * (childset.shape[0] / dataset.shape[0])

        Gain = baseEnt - ents
        if Gain > bestGain:
            bestGain = Gain
            axis = i
    return axis


def split(dataset, axis, value):
    col = dataset.columns[axis]
    newdataset = dataset.loc[dataset[col] == value, :].drop(col, axis=1)
    return newdataset

leaf = 0
layer = 0

def decisiontree(dataset):
    global leaf
    global layer

    featlist = list(dataset.columns)
    y = dataset.iloc[:, -1].value_counts()

    if y[0] == dataset.shape[0] or dataset.shape[1] == 1:
        leaf += 1
        if dataset.shape[1] < data.shape[1] - layer + 1:
            layer = data.shape[1] - dataset.shape[1] + 1
        return y.index[0]

    axis = bestsplit(dataset)
    bestfeat = featlist[axis]
    tree = {bestfeat: {}}
    del featlist[axis]
    valuelist = set(dataset.iloc[:, axis])

    for value in valuelist:
        tree[bestfeat][value] = decisiontree(split(dataset, axis, value))

    return tree


myTree = decisiontree(data)
print(myTree,leaf,layer)

# 画决策树
# 特征
Xtrain = (data.iloc[:, :-1])
# 标签
Ytrain = data.iloc[:, -1]

# 绘制树模型
clf = DecisionTreeClassifier()
clf = clf.fit(Xtrain, Ytrain)
tree.export_graphviz(clf)
dot_data = tree.export_graphviz(clf, out_file=None)
graphviz.Source(dot_data)

# 给图形增加标签和颜色
feat_name = ['age', 'job', 'house', 'debet']
dot_data = tree.export_graphviz(clf, out_file=None,
                                feature_names=feat_name,
                                class_names=['1', '0'],
                                filled=True, rounded=True,
                                special_characters=True)
graphviz.Source(dot_data)

# 利用render方法生成图形
graph = graphviz.Source(dot_data)
graph.render("tree")
graph.view()
