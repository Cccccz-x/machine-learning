import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#get data
path_train="iris_train.data"
path_test="iris_test.data"
train_set = pd.read_csv(path_train, header=None, sep=',')
test_set = pd.read_csv(path_test, header=None, sep=',')
train_set=np.array(train_set)
test_set=np.array(test_set)

epochs=10000
learning_rate=0.0001

def cost(w, x, y):
    return np.sum(np.log(1 + np.exp(w.dot(x)))) - y.dot((w.dot(x)).T)
def argmin(w,x,y):
    history_cost=np.zeros(epochs)
    for i in range(epochs):
        w=w-learning_rate*((np.exp(w.dot(x))/(1+np.exp(w.dot(x)))).dot(x.T)-y.dot(x.T))
        history_cost[i]=cost(w,x,y)
        if i%1000==0:
            print(w,cost(w,x,y))
    return w,history_cost

global x_train,y_train,history_cost
w=np.random.rand(3,5)

for i in range(3):
    #initialize data
    # 使用OvO,first:s=1,ve=0;second:s=1,vi=0;third:ve=1,vi=0
    # setosa，versicolor，virginica

    if i==0:
        x_train=train_set[:40,:4]
        train_set0=np.array(list(train_set))
        y_train=train_set0[:40,4]
        y_train[y_train=='Iris-setosa']=1
        y_train[y_train=='Iris-versicolor']=0
    if i==1:
        x_train = np.r_[train_set[:20, :4],train_set[40:,:4]]
        train_set0=np.array(list(train_set))
        y_train = np.r_[train_set0[:20, 4],train_set0[40:,4]]
        y_train[y_train == 'Iris-setosa'] = 1
        y_train[y_train == 'Iris-virginica'] = 0
    if i==2:
        x_train = train_set[20:, :4]
        train_set0=np.array(list(train_set))
        y_train = train_set0[20:, 4]
        y_train[y_train == 'Iris-versicolor'] = 1
        y_train[y_train == 'Iris-virginica'] = 0
    x_train=np.c_[x_train,np.ones(len(x_train))]
    x_train=x_train.astype(np.float64)
    y_train=y_train.astype(np.float64)
    w[i],history_cost=argmin(w[i],x_train.T,y_train.T)
    #print(w[i],'\n')

#test
test_set[test_set=='Iris-setosa']=0
test_set[test_set=='Iris-versicolor']=1
test_set[test_set=='Iris-virginica']=2
count=0
test_set=test_set.astype(np.float64)
for i in range(len(test_set)):
    y=[0,0,0]
#ovo一共3个模型 这里不妨记setosa为0，versicolor为1，virginica为2
    if w[0].dot(np.r_[test_set[i,:4],1])>0:
        y[0]+=1
    else:
        y[1]+=1
    if w[1].T.dot(np.r_[test_set[i,:4],1])>0:
        y[0]+=1
    else:
        y[2]+=1
    if w[2].T.dot(np.r_[test_set[i,:4],1])>0:
        y[1]+=1
    else:
        y[2]+=1

    if y.index(max(y))==test_set[i,4]:
        count+=1

print(w)
print(count,len(test_set))
print(count/len(test_set))
fig, ax = plt.subplots(figsize=(10, 7))
ax.set_ylabel('cost')
ax.set_xlabel('epochs')
_ = ax.plot(range(epochs), history_cost, 'r')
plt.show()