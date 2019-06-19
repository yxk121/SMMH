import sys, os
sys.path.append(os.pardir)
import numpy as np
from common.gradient import numerical_gradient
from common.functions import softmax, cross_entropy_error


class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)#形状为2*3

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)

        return loss


net = simpleNet()
print("W:"+str(net.W))
x = np.array([0.6, 0.9])
p = net.predict(x)
print("p:"+str(p))
t = np.array([0, 0, 1])
y = net.loss(x, t)
print("loss:"+str(y))


def f(W):
    return net.loss(x, t)


dW = numerical_gradient(f, net.W) #观察损失函数的梯度
print("dW:"+str(dW))

