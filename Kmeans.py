import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path="K-means.data"
data = np.array(pd.read_csv(path, header=None, sep=','))

n = len(data)
m = len(data[0])


# 因为数据本身就是随机的，不妨取前k个作为中心点
def kmeans(k, u0):
    dist = [[np.inf for i in range(k)]for j in range(n)]
    dataset = np.array(list(data)) #避免引用产生错误

    cls = np.empty(n) #cls is lamda
    Cls = [[]for i in range(k)] #Cls is class

    for i in range(n):
        for j in range(k):
            dist[i][j] = np.sqrt(np.sum(np.square(dataset[i]-u0[j])))
        cls[i] = np.argmin(dist[i])
        Cls[int(cls[i])].append(i)

    u = np.array(list(u0))
    for i in range(k):
        for j in range(m):
            b = dataset[Cls[i],j]
            a = np.mean(b)
            u[i][j] = a
    if (u == u0).all():
        return u, Cls
    else:
        u, Cls = kmeans(k, u)
    return u, Cls

def sse(k, Cls):
    sse = 0
    for i in range(k):
        for j in range(len(Cls[i])):
            sse += np.sum(np.square(data[Cls[i][j]] - u[i]))
    return sse

SSE = np.empty(10)
for k in range(1,10):
    u, Cls = kmeans(k,np.array(list(data[:k,:])))
    SSE[k] = sse(k, Cls)


fig, ax = plt.subplots(figsize=(10, 7))
ax.set_ylabel('SSE')
ax.set_xlabel('k')
_ = ax.plot(range(1, 10), SSE[1:], 'r')
plt.show()

u, Cls = kmeans(4, np.array(list(data[:4, :])))
fig1, ax1 = plt.subplots(figsize=(10, 7))
ax1.set_ylabel('x1')
ax1.set_xlabel('x2')
_ = ax1.scatter(data[Cls[1],0], data[Cls[1],1], color='r')
_ = ax1.scatter(data[Cls[0],0], data[Cls[0],1], color='b')
_ = ax1.scatter(data[Cls[2],0], data[Cls[2],1], color='g')
_ = ax1.scatter(data[Cls[3],0], data[Cls[3],1], color='y')

plt.show()