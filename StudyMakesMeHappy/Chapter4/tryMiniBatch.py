import os, sys
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = \
    load_mnist(normalize=True, one_hot_label=True)

train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)#取0-train_size之间的batch_size个数字
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]#相当于选了mask那么多行


print(train_size)
print(batch_mask)
print(batch_mask.shape)
print(x_batch.shape)
print(t_batch.shape)
print(x_train.shape[1])
