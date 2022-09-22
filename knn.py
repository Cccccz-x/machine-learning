import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

path="knn_data.csv"
data = np.array(pd.read_csv(path, header=None, sep=' '))
print(data)
n = len(data)


dist = [[np.inf for i in range(n)] for j in range(n)]
for i in range(n):
    dist[i][i] = 0
    for j in range(i+1,n):
        dist[i][j] = np.sqrt(np.sum(np.square(data[i,:4]-data[j,:4])))
        dist[j][i] = dist[i][j]

# k=10， 采用5折交叉验证法
# 将数据集分成5组,每组4个数据
# 因为两类数据量均为10，故不需要对类别添加权重，可对距离添加权重
group = list()
j = 0
while j<n/2:
    group.append([int(j),int(j+1),int(n/2+j),int(n/2+j+1)])
    j+=2

# 寻找最近
right = 0
false = 0
close = np.empty(10)
for i in range(len(group)):
    for k in group[i]:
        s = 0
        v = 0
        close = np.argsort(np.delete(dist[k],group[i]))[:10] #相距最近的前十个索引,除去group[i]

        for m in close:
            if data[m,-1] == 'setosa':
                s+=1
            else:
                v+=1
        if s>v:
            print('the prediction of ',k,' is setosa')
            if data[k,-1] == 'setosa':
                print('is right')
                right +=1
            else:
                print('is false')
                false +=1
        else:
            print('the prediction of ',k,' is versicolor')
            if data[k,-1] == 'versicolor':
                print('is right')
                right += 1
            else:
                print('is false')
                false += 1


#plot
fig1=plt.figure()
ax1=Axes3D(fig1,auto_add_to_figure=False)
fig1.add_axes(ax1)
ax1.set_xlabel('x1')
ax1.set_ylabel('x2')
ax1.set_zlabel('x3')
ax1.scatter3D(data[data[:,-1]=='setosa',0],data[data[:,-1]=='setosa',1],data[data[:,-1]=='setosa',2],color='b')
ax1.scatter3D(data[data[:,-1]=='versicolor',0],data[data[:,-1]=='versicolor',1],data[data[:,-1]=='versicolor',2],color='r')

center = [data[0,0],data[0,1],data[0,2]]
radius = dist[0][np.argsort(np.delete(dist[0],group[0]))[9]]
# data
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x = radius * np.outer(np.cos(u), np.sin(v)) + center[0]
y = radius * np.outer(np.sin(u), np.sin(v)) + center[1]
z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + center[2]
# surface plot
ax1.plot_surface(x, y, z, rstride=4, cstride=4, color='b',alpha=0.3)
plt.show()

print('right:',right)
print('false:',false)
print('accuracy',right/n)
