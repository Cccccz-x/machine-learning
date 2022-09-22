import numpy as np
import matplotlib.pyplot as plt

w=1.7
b=1.8
x=np.random.rand(100,1)
y=w*x+b+np.random.rand(100,1)-0.5
plt.scatter(x,y)
epochs=5000
m=len(x)
plt.plot(x,w*x+b,'y')

w1=(x.T.dot(y)-(1/m)*np.sum(x)*np.sum(y))/(x.T.dot(x)-1/m*np.square(np.sum(x)))
b1=(1/m)*np.sum(y-w*x)
print(w1,b1)
plt.plot(x,w1*x+b1,'g')


def cost(x,y,w,b):
    m=len(x)
    y_pre=w*x+b
    c=(1/(2*m))*np.sum(np.square(y_pre-y))
    return c

def gradient_descent(x,y,w,b,learning_rate,epochs):
    m=len(x)
    history_cost=np.zeros(epochs)
    for i in range(epochs):
        y_pre=w*x+b
        w=w-learning_rate*(1/m)*(x.T.dot(y_pre-y))
        b=b-learning_rate*(1/m)*np.sum(y_pre-y)
        history_cost[i]=cost(x,y,w,b)
    return history_cost,w,b
ww=1
bb=1
learning_rate=5
history_cost,w2,b2=gradient_descent(x,y,ww,bb,learning_rate,epochs)
print(w2,b2)
plt.plot(x,w2*x+b2,'b')

fig, ax = plt.subplots(figsize=(10, 7))

ax.set_ylabel('cost')
ax.set_xlabel('epochs')
_ = ax.plot(range(epochs), history_cost, 'r')
plt.show()