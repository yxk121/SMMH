import numpy as np

x = np.array([[1.0, 2.0, 3.0, 4.0], [7.0, 8.0, 9.0, 10.0]])
y = np.array([[4.0, 5.0, 10.0, 11.0]])
print(x.shape)
print(y.shape)
print(x)
print(y)
print(x*y)

'''
numpy数组这个东西似乎只能进行广播，如上面这个程序，x(2,4)和x相乘的y要么是(1,4)要么是(2,4)
好像并不是，详见Chapter3：tryNumpyCal
点乘和叉乘是不一样的哦，此处为叉乘，只能实现广播。Chapter3那个是点乘，和矩阵方法一样
'''

for row in x:
    print(row)#读每一行

x = x.flatten()#将x变成一维数组
print(x)
