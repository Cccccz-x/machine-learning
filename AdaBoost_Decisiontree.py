import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = "AdaBoost_data.csv"
data = np.array(pd.read_csv(path, header=None, sep=' '))[:,1:]
data[data=='男'] = 1
data[data=='女'] = -1

#build tree
def bt(dataset,D):
    m = len(dataset)
    n = len(dataset[0])
    besterr = np.inf
    tree = {}
    stepnum = 20
    for i in range(n-1):
        Min = min(dataset[:,i])
        Max = max(dataset[:,i])
        step = (Max-Min)/stepnum
        for j in range(m):
            thresh = Min+j*step
            for inequal in ['lt','gt']:
                err = calerr(dataset, i, thresh, inequal, D)
                if err < besterr:
                    besterr = err
                    tree["feature"] = i
                    tree["thresh"] = thresh
                    tree["inequal"] = inequal
                    tree["err"] = err
    return tree

def calerr(dataset, feature, thresh, inequal, D):
    D = D.flatten()
    errcnt = 0
    i = 0
    if inequal == 'lt':
        for data in dataset:
            if (data[feature] <= thresh and data[-1] == -1) or \
               (data[feature] > thresh and data[-1] == 1):
                errcnt += 1*D[i]
            i += 1
    else:
        for data in dataset:
            if (data[feature] >= thresh and data[-1] == -1) or \
               (data[feature] < thresh and data[-1] == 1):
                errcnt += 1*D[i]
            i += 1
    return errcnt

def calacc(dataset, G):
    cnt = 0
    for data in dataset:
        pre = adapredict(data,G)
        if pre == data[-1]:
            cnt += 1
    return cnt / len(dataset)

def adapredict(dataset, G):
    score = 0
    for key in G.keys():
        pre = predict(dataset, G[key]["tree"])
        score += G[key]["alpha"] * pre
    flag = 0
    if score > 0:
        flag = 1
    else:
        flag = -1
    return flag

def predict(dataset, tree):
    if tree["inequal"] == 'lt':
        if dataset[tree["feature"]] <= tree["thresh"]:
            return 1
        else:
            return -1
    else:
        if dataset[tree["feature"]] >= tree["thresh"]:
            return 1
        else:
            return -1

def AdaBoost(dataset, T):
    m = len(dataset)
    n = len(dataset[0])
    D = np.array([1 / m for i in range(m)])
    clslabel = dataset[:,-1].reshape(1,-1)
    G = {}
    for t in range(T):
        tree = bt(dataset,D)
        err = tree["err"]
        alpha = 0.5*np.log((1-err)/err)
        pre = np.zeros((1,m))
        for i in range(m):
            pre[0][i] = predict(dataset[i], tree)
        a = np.exp(np.array(-alpha * clslabel * pre,dtype=np.float64))
        D = D * a / D.dot(a.T)
        G[t] = {}
        G[t]["alpha"] = alpha
        G[t]["tree"] = tree
    return G

def plot(dataset, clf):
    x1,x2 = [],[]
    y1,y2 = [],[]
    datas = dataset
    labels = dataset[:,-1]
    for data, label in zip(datas, labels):
        if label > 0:
            x1.append(data[0])
            y1.append(data[1])
        else:
            x2.append(data[0])
            y2.append(data[1])
    x = [0,200]
    y = [0,100]
    for key in clf.keys():
        z = [clf[key]["tree"]["thresh"]]*2
        if clf[key]["tree"]["feature"] == 0:
            plt.plot(z,y)
        else:
            plt.plot(x,z)
    plt.scatter(x1, y1, marker='+', label='Male', color='b')
    plt.scatter(x2, y2, marker='o', label='Female', color='r')
    plt.xlabel('Height')
    plt.ylabel('Weight')
    plt.xlim(150,190)
    plt.ylim(40,70)
    plt.legend(loc='upper left')

def out(G):
    for i in G.keys():
        print(f"第{i+1}个弱分类器的结果")
        if G[i]["tree"]["feature"] == 0:
            if G[i]["tree"]["inequal"] == 'lt':
                print("身高<=",G[i]["tree"]["thresh"],"为男性")
                print("身高>", G[i]["tree"]["thresh"], "为女性")
            else:
                print("身高>=", G[i]["tree"]["thresh"], "为男性")
                print("身高<", G[i]["tree"]["thresh"], "为女性")
        else:
            if G[i]["tree"]["inequal"] == 'lt':
                print("体重<=",G[i]["tree"]["thresh"],"为男性")
                print("体重>",G[i]["tree"]["thresh"],"为女性")
            else:
                print("体重>=",G[i]["tree"]["thresh"],"为男性")
                print("体重<",G[i]["tree"]["thresh"],"为女性")


plt.figure(figsize=(15,10),dpi=90)
for t in range(1,12):
    G = AdaBoost(data, t)
    # print('集成学习器（字典）：',f"G{t}={G}")
    print(f'第{t}个集成学习器的准确率=', calacc(data, G))
    plt.subplot(3,4,t)
    plot(data, G)
    if t == 11:
        out(G)


plt.tight_layout()
plt.show()