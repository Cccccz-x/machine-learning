import numpy as np
import matplotlib.pyplot as plt

x = 2 * np.random.rand(100, 1)
y = 5 * x - 6 + np.random.randn(100, 1)


def cost(x, y, theta):
    m = len(y)
    prediction = x.dot(theta)
    c = (1 / (2 * m)) * np.sum(np.square(prediction - y))
    return c


def gradient_descent(x, y, theta, learning_rate=0.01, epochs=5000):
    m = len(y)
    history_cost = np.zeros(epochs)
    history_theta = np.zeros((epochs, 2))
    for i in range(epochs):
        prediction = x.dot(theta)

        theta = theta - (1 / m) * learning_rate * (x.T.dot((prediction - y)))
        history_theta[i, :] = theta.T
        history_cost[i] = cost(x, y, theta)
    return theta, history_theta, history_cost


learning_rate = 0.01
epochs = 5000
theta = np.random.randn(2, 1)

x_b = np.c_[np.ones((len(x), 1)), x]
theta, history_theta, history_cost = gradient_descent(x_b, y, theta, learning_rate, epochs)
# print('Gradient Descent:\n')
# print('Theta0: {:0.3f},\nTheta1: {0.3f}'.format(theta[0][0],theta[1][0]))
# print('MSE: {:0.3f}'.format(history_cost[-1]))

fig, ax = plt.subplots(figsize=(12, 8))

ax.set_ylabel('cost')
ax.set_xlabel('epochs')
_ = ax.plot(range(epochs), history_cost, 'r')
plt.show()
