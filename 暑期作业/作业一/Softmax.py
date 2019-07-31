#
# Date 2019.7.12
# Author Zhao Guo
# Softmax Classifier
#

import numpy as np
import matplotlib.pyplot as plt
import struct


class Softmax():
    """
    此类用于实现Softmax分类器，用于进行多分类，使用的数据集是MNIST
    """

    def __init__(self, W):
        self.W = W

    def loss(self, X_train, y_train, reg):
        """
        此函数用于计算损失值和梯度，采用的是梯度下降算法
        :param X_train:训练集图片
        :param y_train:训练集标签
        :param reg:正则项系数
        返回值为损失值和梯度值
        """

        loss = 0.0
        dW = np.zeros_like(self.W)

        num_train = X_train.shape[0]
        scores = X_train.dot(self.W)
        scores_shift = scores - np.max(scores, axis = 1).reshape((-1,1))
        softmax_output = np.exp(scores_shift) / np.sum(np.exp(scores_shift), axis = 1).reshape((-1,1))
        
        # 计算损失值，采用似然函数作为损失函数
        loss -= np.sum(np.log(softmax_output[range(num_train), list(y_train)]))
        loss /= num_train
        loss += reg * np.sum(self.W * self.W)

        # 计算梯度，采用反向传播算法
        dS = softmax_output.copy()
        dS[range(num_train), list(y_train)] -= 1
        dW = (X_train.T).dot(dS)
        dW = dW / num_train + 2 * reg * self.W

        return loss, dW

    def train(self, X_train, y_train, batch_size = 200, num_iters = 100, reg = 0.005, learning_rate = 0.001):
        """
        此函数用于对softmax模型进行训练，进行若干次迭代后，获得最佳的w,b
        :param batch_size:每次进行训练的样本大小
        :param num_iters:进行训练的迭代次数
        :param reg:正则项系数
        :param learning_rate:学习率
        """
        
        num_train = X_train.shape[0]
        loss_history = []

        for i in range(num_iters):
            indices = np.random.choice(num_train, batch_size)
            X_batch = X_train[indices]
            y_batch = y_train[indices]

            loss, dW = self.loss(X_batch, y_batch, reg)
            
            loss_history.append(loss)
            self.W -= learning_rate * dW

        return loss_history

    def predict(self, X_test):
        """
        用于测试softmax模型在测试集上的效果
        :param X_test:这是测试集
        """

        y_pred = np.zeros(X_test.shape[0])
        scores = X_test.dot(self.W)
        y_pred = np.argmax(scores, axis = 1)

        return y_pred

    def Show(self, loss_history):
        """
        将损失值进行可视化，用于判断在训练过程中的的情况
        :param loss_history: 损失值列表
        """

        plt.figure(figsize=(10, 6))
        plt.subplot(1,1,1)

        plt.plot(loss_history)
        plt.title("Loss History")
        plt.xlabel("Train Numbers")
        plt.ylabel("Loss")

        plt.savefig("Loss.jpg")
        #plt.show()


def load_data():
    """
    加载数据
    """

    X_train_path = "MNIST/train-images-idx3-ubyte"
    y_train_path = "MNIST/train-labels-idx1-ubyte"
    X_test_path = "MNIST/test-images-idx3-ubyte"
    y_test_path = "MNIST/test-labels-idx1-ubyte"

    X_train = image_decode(X_train_path)
    X_train = np.reshape(X_train, (X_train.shape[0], -1))
    y_train = label_decode(y_train_path)

    X_test = image_decode(X_test_path)
    X_test = np.reshape(X_test, (X_test.shape[0], -1))
    y_test = label_decode(y_test_path)

    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

    # 对特征进行缩放,进行max-min归一化
    X_train_max, X_train_min = np.max(X_train), np.min(X_train)
    X_train = (X_train - X_train_min) / (X_train_max - X_train_min)
    X_test_max, X_test_min = np.max(X_test), np.min(X_test)
    X_test = (X_test - X_test_min) / (X_test_max - X_test_min)

    # 将数据集和测试集的图片部分进行扩充
    X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
    X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
    
    return X_train, y_train, X_test, y_test


def image_decode(file):
    """
    解析二进制图像文件，然后放入训练集中
    : param file: 文件路径
    """
    
    f = open(file, 'rb').read()   # 打开文件并读入
    offset = 0                    # 文件偏移量

    fmt_header = '>iiii'          # '>' 表示大端方式，i 表示32为整数

    # 将byte 类型数据解析为整数形式，带入fmt_header格式字符串，返回一个元组类型
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, f, offset)  

    image_size = num_cols * num_rows   # 手写数字图片大小

    offset += struct.calcsize(fmt_header)  # 更改偏移量
    fmt_header = '>' + str(image_size) + 'B'   # 更改字符串格式,一个像素是一个字节

    images = np.empty((num_images, num_rows, num_cols))  # 设定返回图片格式

    for i in range(num_images):
        if (i + 1) % 10000 == 0:
            print('图片已解析 %d ' % (i + 1) + '张')
        # 解析一张图片
        images[i] = np.array(struct.unpack_from(fmt_header, f, offset)).reshape((num_rows, num_cols))
        # 更改偏移量
        offset += struct.calcsize(fmt_header)

    return images


def label_decode(file):
    """
    解析标签值
    : param file : 标签二进制文件
    """

    f = open(file, 'rb').read()

    offset = 0
    fmt_header = '>ii'          # 注意这里的偏移量与图片不一样，是两个int32

    magic_number, num_images = struct.unpack_from(fmt_header, f, offset)   # 解析byte
    offset += struct.calcsize(fmt_header)
    fmt_image = '>B'            # 一个标签是一个字节
    labels = np.empty(num_images)

    for i in range(num_images):
        if (i + 1) % 10000 == 0:
            print('标签已解析 %d ' % (i + 1) + '张')
        labels[i] = struct.unpack_from(fmt_image, f, offset)[0]   # 解析标签，注意返回是元组类型
        offset += struct.calcsize(fmt_image)                      # 更改偏移量

    return labels


def main():

    X_train, y_train, X_test, y_test = load_data()

    W = np.random.randn(X_train.shape[1], 10)
    softmax = Softmax(W)

    # 调整参数，让softmax在测试集上的精确度达到最优的值
    loss_history = softmax.train(X_train, y_train, batch_size = 200, num_iters = 10000,   \
                                reg = 0.08, learning_rate = 0.05)
    y_train_pred =  softmax.predict(X_train)
    y_test_pred = softmax.predict(X_test)
    print("Train Accuracy: %.4lf" % np.mean(y_train_pred == y_train))
    print("Test Accuracy: %.4lf" % np.mean(y_test_pred == y_test))
    softmax.Show(loss_history)

main()
