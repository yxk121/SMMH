#计算神经网络精度，批处理
#预测结果和标签对比，计算精度

import sys
import os
sys.path.append(os.pardir) # 为了导入父目录中的文件而进行的设定
import numpy as np
from dataset.mnist import load_mnist #dataset是从深度学习入门代码里考来的
import pickle
from common.functions import sigmoid, softmax


def get_data():
    (x_train, t_train), (x_test, t_test) = \
        load_mnist(normalize=True, flatten=True, one_hot_label=False)
 # 上面这句的反斜杠换行用的
    return x_test, t_test


def init_network():
    with open("sample_weight.pkl", 'rb') as f:  #打开pickle文件中的学习到的权重
        network = pickle.load(f)

    return network


def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y


x, t = get_data()
network = init_network()

batch_size = 100   #批处理
accuracy_cnt = 0


for i in range(0, len(x), batch_size): #len返回对象（字符、列表、元组等）长度或项目个数
 # range(start, end, step)
    x_batch = x[i:i+batch_size] #左闭右开，取每一批的第i行
    y_batch = predict(network, x_batch)
    #y = predict(network, x[i])
    #p = np.argmax(y) #获得最大值的索引,最大值即为预测结果（谁的概率最大他就是谁）
    p = np.argmax(y_batch, axis=1)#沿第一维方向,一行一行的来
    # if p == t[i]:
    #     accuracy_cnt += 1
    accuracy_cnt += np.sum(p == t[i:i+batch_size])#p和t有多少个相等的元素，如果p中保留的预测结果和t中一样就累加

print("Accuracy:" + str(float(accuracy_cnt)/len(x)))
