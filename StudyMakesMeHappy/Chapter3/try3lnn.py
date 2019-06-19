#   3lnn means 3 layers neural networks

import numpy as np


def sigmoid(x):
    return 1/(1 + np.exp(-x))


def identity_function(x):
    return x


def softmax(x):
    c = np.max(x)#c越大exp_a越小，就越不会溢出
    exp_a = np.exp(x-c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a#分子分母同时加减某一个值，结果不变(溢出对策)

    return y     #此函数特点：y所有值之和为1


def init_network():
    network = {} # 字典
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]]) #(2,3)   输入节点有两个，即x.shape=(1,2)，所以可以点乘，注意是x乘W
    network['b1'] = np.array([0.1, 0.2, 0.3]) #(1,3)   这俩是输入层到第一层的权重和偏置
     #第一层的结果经过激活函数计算后再进行到第二层的计算，第一层结果是(1,3),由于第二层的神经元有两个，所以到第二层的权重为(3,2)
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]]) #点乘后变为(1,2)，因此偏置为(1,2)
    network['b2'] = np.array([0.1, 0.2])
    #输出层为两个神经元，因此权重为(2,2)，点乘结果为(1,2)，所以偏置为(1,2)
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])

    return network


def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = identity_function(a3)#将输入值直接输出 恒等函数

    return y


network = init_network()
x = np.array([1.0, 0.5])
y = forward(network, x)
print(y)
