import numpy as np
import pandas as pd

path = "AdaBoost_data.csv"
name = ["no","h","w","s"]
data = pd.read_csv(path, header=None, sep=' ', names=name).iloc[:,1:]
m = len(data)

w = np.array([1/m for i in range(m)])
T = 10

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


#
# for t in range(T):
