import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#initialize data
learning_rate=0.001
epochs=10000
size=1000
sep=700
x1=np.random.normal(0,1,size)
x2=np.random.normal(0,2,size)
y=x1+x2+0.1*np.random.rand(size)
for i in range(size):
    if y[i]>0:
        y[i]=1
    if y[i]<=0:
        y[i]=0
w=np.ones(3)
#training set
x1_train=x1[:sep]
x2_train=x2[:sep]
y_train=np.array(y[:sep])
#test set
x1_test=x1[sep:]
x2_test=x2[sep:]
y_test=np.array(y[sep:])
x_train=np.array([x1_train,x2_train,np.ones(sep)])
x_test=np.array([x1_test,x2_test,np.ones(size-sep)])
#print(x_train)
def cost(w,x,y):
    return np.sum(np.log(1+np.exp(w.dot(x))))-y.dot((w.dot(x)).T)

def argmin(w,x,y):
    print(x.dtype,y.dtype,w.dtype)
    print(x,'\n',y,'\n',w)
    history_cost=np.zeros(epochs)
    for i in range(epochs):
        w=w-learning_rate*((np.exp(w.dot(x))/(1+np.exp(w.dot(x)))).dot(x.T)-y.dot(x.T))
        history_cost[i]=cost(w,x,y)
        if i%1000==0:
            print(w,cost(w,x,y))

    return w,history_cost
(w,history_cost)=argmin(w,x_train,y_train)
def accuracy(w,x,y):
    n=0
    for i in range(size-sep):
        if w.dot(x[:,i])>0:
            if y[i]==1:
                n=n+1
        else:
            if y[i]==0:
                n=n+1
    return n/(size-sep)


print(accuracy(w,x_test,y_test))
print(accuracy(np.array([1,1,0]),x_test,y_test))

fig, ax = plt.subplots(figsize=(10, 7))
ax.set_ylabel('cost')
ax.set_xlabel('epochs')
_ = ax.plot(range(epochs), history_cost, 'r')



#plot
fig1=plt.figure()
ax1=Axes3D(fig1,auto_add_to_figure=False)
fig1.add_axes(ax1)
ax1.set_xlabel('x1')
ax1.set_ylabel('x2')
ax1.set_zlabel('y')
ax1.scatter3D(x1_train,x2_train,y_train,'r')
ax1.scatter3D(x1_test,x2_test,y_test,'b')
plt.show()