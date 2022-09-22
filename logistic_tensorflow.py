import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_spilt

#initialization
mu=0
sigma=1
num=100
x11=np.random.normal(mu,sigma,num)
x1=x11.reshape([num,1])
x22=np.random.normal(mu,sigma,num)
x2=x22.reshape([num,1])
x=np.hstack((x1,x2))
y1=[]
y22=[]
for i in range(num):
    y1.append(x1[i]+x2[i])
    if y1[i]>0:
        y22.append(1)
    else:
        y22.append(0)
y11=np.array(y22)
y=y11.reshape([num,1])
data=np.hstack((x,y))
print("data set",data)

#sepration
X_train,X_test,y_train,y_test=train_test_spilt(x,y,test_size=0.3)
X=tf.placeholder(tf.float32,shape=[None,2])
Y=tf.placeholder(tf.float32,shape=[None,1])
W=tf.Variable(tf.random_normal([2,1]),name='weight')
b=tf.Variable(tf.random_normal([1]),name='bias')
hypothesis=tf.sigmoid(tf.matmul(X,W)+b)
predicted=tf.cast(hypothesis>0.5,dtype=tf.float32)
accuracy=tf.reduce_mean(Y*tf.log(hypothesis)+(1-Y)*tf.log(1-hypothesis))
cost=-tf.reduce_mean(Y*tf.log(hypothesis)+(1-Y)*tf.log(1-hypothesis))
train=tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)
total=[]
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(5000)
        total_loss=0
        cost_val,_=sess.run([cost,train],feed_dict={X:X_train,Y:y_train})
        total_loss+=cost_val
        total.append(total_loss/num)
        if step % 200==0:
            print(step,cost_val)

    h,c,a=sess.run([hypothesis,predicted,accuracy],feed_dict={X:X_test,Y:y_test})
    print("\nHypothesis:",h,"\nCorrect(Y):",c,"\nAccuracy:",a)

M2=np.array(c)
N2=np.array(y_test)
M2=M2.astype(int)
M=M2.flatten()
N=N2.flatten()
print('predict',M)
print('reality',N)
plt.plot(X_test,M2,'bo',label='Predict')
plt.plot(X_test,N2,'ro',label='Real')
plt.legend()
plt.show()
plt.plot(total,label='loss')
plt.show()
