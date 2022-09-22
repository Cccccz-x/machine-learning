import numpy as np
import xlrd
import matplotlib.pyplot as plt

learning_rate=0.00055
epochs=1000000
file="水泥数据集.xlsx"
data=xlrd.open_workbook(file)
sheet=data.sheet_by_index(0)

w=10*np.ones(sheet.ncols-1)#行向量
x=[[sheet.cell_value(r,c) for c in range(1,5)] for r in range(1,sheet.nrows)]
y=[sheet.cell_value(r,5) for r in range(1,sheet.nrows)]
m=len(x)
x=np.array(np.column_stack((x,np.ones(m))))#b in w and x

def cost(y,x,w):
    m=len(x)
    y_pre = w.dot(x.T)
    c=1/(2*m)*np.sum(np.square(y-y_pre))

    return c

def gradient_descent(y,x,w,learning_rate,epochs):
    m=len(x)
    history_cost=np.zeros(epochs)
    for i in range(epochs):
        y_pre=w.dot(x.T)
        w=w-learning_rate*(1/m)*(x.T.dot(y_pre-y))
        history_cost[i]=cost(y,x,w)
        print(history_cost[i])
    return w,history_cost

w,history_cost=gradient_descent(y,x,w,learning_rate,epochs)
print(history_cost[epochs-1],'\n',w)

fig, ax = plt.subplots(figsize=(10, 7))
ax.set_ylabel('cost')
ax.set_xlabel('epochs')
_ = ax.plot(range(epochs), history_cost, 'r')
plt.show()

#0.0001 10000000
#1.9874021806069808
#[2.12621683 1.08636359 0.69015312 0.42068779 6.49720018]

#0.00005 10000000
#2.001791254857906
#[2.15380026 1.1139989  0.71836626 0.44777408 3.81575142]

#0.0005 10000000
#1.9101385895080805
#[ 1.94646099  0.90626962  0.50629356  0.24417172 23.97167744]

#0.00001 1000000
#641.469913287894
#[103.96030356   0.39695249   3.2346345 ]