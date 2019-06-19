import numpy as np


def AND(x1, x2):#与门
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    y = np.sum(x * w) + b
    if y > 0:
        return 1
    else:
        return 0


def NAND(x1, x2):#与非门
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    y = np.sum(w * x) + b
    if y > 0:
        return 1
    else:
        return 0


def OR(x1, x2):#或门
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.3
    y = np.sum(x * w)+b
    if y > 0:
        return 1
    else:
        return 0


def XOR(x1, x2):#异或门
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y


print(XOR(1, 1))
print(XOR(1, 0))
print(XOR(0, 1))
print(XOR(0, 0))