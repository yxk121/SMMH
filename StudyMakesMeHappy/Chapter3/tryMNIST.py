import sys, os
sys.path.append(os.pardir) # 为了导入父目录中的文件而进行的设定
import numpy as np
from dataset.mnist import load_mnist #dataset是从深度学习入门代码里考来的
from PIL import Image  #PIL在终端装的


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img)) #此函数把numpy数组的图像数据转化成PIL用的数据对象
    pil_img.show()


(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False) # flatten：是否展开成一维数组输出，normalize：是否正规化0.0~1.0
# 返回值：(训练图像, 训练标签), (测试图像, 测试标签)
img = x_train[0] #x_train是个numpy数组，此处表示其第0行，一行就是一个待处理的图
label = t_train[0]
print(label)


print(img.shape)
img = img.reshape(28, 28) #变形!
print(img.shape)


img_show(img)
