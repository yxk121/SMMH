#p114

import numpy as np
from dataset.mnist import load_mnist
from Chapter4.try2lnn import TwoLayerNet

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

train_loss_list = []#存放损失函数的值

#超参数
iters_num = 10000 #梯度算法的更新次数
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

for i in range(iters_num):#为啥要循环?更新iters_num次
    #获取mini-batch
    batch_mask = np.random.choice(train_size, batch_size)#取0-train_size之间的batch_size个数字
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    #计算梯度
    grad = network.numerical_gradient(x_batch, t_batch)
    #grad = network.gradient(x_batch, t_batch)

    #更新参数
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    #记录学习过程
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
