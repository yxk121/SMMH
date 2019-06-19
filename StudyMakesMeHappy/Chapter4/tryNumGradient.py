import numpy as np


def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x)

        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val

    return grad


'''
x[idx],当单独使用x时，x代表的是什么？整个数组还是某一个值？x[idx]代表输入的若干自变量中的一个值

f如何输入？就这样输入

'''


def gradient_descent(f, init_x, lr=0.01, step_num=100): #找梯度最小，损失函数值最小的点
    x = init_x

    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad

    return x


def function1(x):
    return x[0]**2 + x[1]**2


y1 = numerical_gradient(function1, np.array([3.0, 4.0]))#x处输入了两个值，是二元方程
print(y1)

y2 = gradient_descent(function1, np.array([3.0, 4.0]), lr=0.1, step_num=100)
print(y2)
