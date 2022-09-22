import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

path_train="abalone_train.data"
path_test="abalone_test.data"
train_set = np.array(pd.read_csv(path_train, header=None, sep=','))
test_set = np.array(pd.read_csv(path_test, header=None, sep=','))
y_train = train_set[:,0]
x_train = np.array(train_set[:,1:],dtype=np.float64)
y_test = test_set[:,0]
x_test = np.array(test_set[:,1:],dtype=np.float64)
print(train_set)


# 归一化
x_train[:,-1]/=10
x_test[:,-1]/=10


train_size=train_set.shape[0]
test_size = test_set.shape[0]

for i in range(train_size):
    if y_train[i] == 'M':
        y_train[i] = [1,0,0]
    else:
        if y_train[i] == 'F':
            y_train[i] = [0,1,0]
        else:
            y_train[i] = [0,0,1]

for i in range(test_size):
    if y_test[i] == 'M':
        y_test[i] = [1,0,0]
    else:
        if y_test[i] == 'F':
            y_test[i] = [0,1,0]
        else:
            y_test[i] = [0,0,1]


learning_rate=0.01
epochs=50
#define the number of neuron
#there is 1 input layer, 1 output layer, 1 hidden layer
input_size=8
output_size=3
hide1_size=10

#initialize data
#x=input  y=output  b=hidden
#wi = i layer connection weight
# x2=np.random.rand(hide1_size,1)

#+1 means include the threshold
w1=0.1*np.random.rand(input_size+1,hide1_size)
w2=0.1*np.random.rand(hide1_size+1,output_size)

extra=np.ones((1,1))

def sigmoid(x):
    return 1/(1+np.exp(-x))


def cost(w1,w2,x1,y):
    x2=sigmoid(w1.T.dot(np.r_[x1,extra]))
    y_pre=sigmoid(w2.T.dot(np.r_[x2,extra]))
    return (1/2)*np.sum(np.square(y_pre-y))

def argmin(w1,w2):
    history_cost=np.zeros(epochs)
    for k in range(epochs):
        cos=np.zeros(train_size)
        for i in range(train_size):

            x1=np.array(x_train[i]).reshape(input_size,1)
            y=np.array(y_train[i]).reshape(output_size,1)

            x1e=np.r_[x1,extra]
            x2=sigmoid(w1.T.dot(x1e))
            x2e=np.r_[x2,extra]
            y_pre=sigmoid(w2.T.dot(x2e))
            dy=(y_pre-y).T
            diff_y_pre=y_pre.T.dot(1-y_pre)

            t=np.delete(w2,hide1_size,axis=0).T #此变量定义只为了截图方便
            w2-=learning_rate*2*(x2e.dot(diff_y_pre)).dot(dy)
            w1-=learning_rate*2*(diff_y_pre*(x2.T.dot(1-x2))*((x1e.dot(dy)).dot(t)))
            cos[i]=cost(w1,w2,x1,y)
        history_cost[k]=np.sum(cos)
        if k%10==0:
            print(history_cost[k])
    return w1,w2,history_cost

w1,w2,history_cost = argmin(w1,w2)

# test
false=0
for i in range(test_size):

    x1 = np.array(x_test[i]).reshape(input_size, 1)
    y = np.array(y_test[i]).reshape(output_size, 1)

    x1e = np.r_[x1, extra]
    x2 = sigmoid(w1.T.dot(x1e))
    x2e = np.r_[x2, extra]
    y_pre = sigmoid(w2.T.dot(x2e))
    m = 0
    n = 0
    index1 = 0
    index2 = 0

    for j in range(output_size):
        if m<y_pre[j]:
            m = y_pre[j]
            index1 = j
        if n<y[j]:
            n = y[j]
            index2 = j
    if index1 != index2:
        false+=1
        print(y_pre.reshape(1,3),y.reshape(1,3))
print('错误个数：',false)
print('准确率：',(test_size-false)/test_size)


fig, ax = plt.subplots(figsize=(10, 7))
ax.set_ylabel('cost')
ax.set_xlabel('epochs')
_ = ax.plot(range(epochs), history_cost, 'r')
plt.show()