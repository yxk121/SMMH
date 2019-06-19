import numpy as np
from dataset.mnist import load_mnist
from Chapter4.try2lnn import TwoLayerNet

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
train_loss_list = []

iters_num = 10000
batch_size = 100
train_size = t_train.shape[0]
learn_rate = 0.1
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    grad = network.numerical_gradient(x_batch, t_batch)

    for key in ('W1', 'W2', 'b1', 'b2'):
        network.params[key] -= learn_rate * grad[key]

    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)