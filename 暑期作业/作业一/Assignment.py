# 
# Assingment 1
# Author Zhao Guo
# Date 2019.07.10
#

import numpy as np
import struct
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import cv2


def load_data(style="pixel"):
    """
    加载数据
    :param style: 用于表示读取的数据是像素值还是直方图
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


    # 特征采用灰度直方图
    if style == "hist":
        num_train = X_train.shape[0]
        num_test = X_test.shape[0]

        X_train_output = np.empty((num_train, 256))
        for i in range(num_train):
            # 统计各个像素值的个数，返回一个array
            # minlength 指定bin的个数，图片像素为float类型，需要将其转化成int类型
            X_train_output[i] = np.bincount(X_train[i].astype(int), minlength = 256)

        X_test_output = np.empty((num_test, 256))
        for i in range(num_test):
            X_test_output[i] = np.bincount(X_test[i].astype(int), minlength = 256)

        return X_train_output, y_train, X_test_output, y_test
    
    # 采用像素值
    else:
        # 对特征进行缩放,进行max-min归一化
        X_train_max, X_train_min = np.max(X_train), np.min(X_train)
        X_train = (X_train - X_train_min) / (X_train_max - X_train_min)

        X_test_max, X_test_min = np.max(X_test), np.min(X_test)
        X_test = (X_test - X_test_min) / (X_test_max - X_test_min)

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


def Logistic_Regression(X_train, y_train, X_test, y_test):
    """
    这个函数是采用逻辑回归模型来训练
    :param X_train: 训练图片
    :param y_train: 训练图片标签
    :param X_test: 测试图片
    :param y_test: 测试图片标签
    """

    # 建立一个逻辑回归模型
    # verbose 表示是否打印训练记录
    # solver 表示采用的优化方法
    # multi_class 表示采用多分类逻辑回归
    lr = LogisticRegression(verbose=1, solver="sag", multi_class="multinomial")
    # 开始训练
    lr.fit(X_train, y_train)
    # 对测试集进行预测
    predict = lr.predict(X_test)
    # 打印测试精度和报告
    print("accuracy scores: %.4lf" % accuracy_score(predict, y_test))
    print("classificatiom report for classifier %s:\n%s\n" % (lr, classification_report(y_test, predict)))


def SVM(X_train, y_train, X_test, y_test):

    # 建立一个支持向量机模型
    svm = SVC(gamma= "scale")
    svm.fit(X_train, y_train)
    predict = svm.predict(X_test)
    print("accuracy scores: %.4lf" % accuracy_score(predict, y_test))
    print("classificatiom report for classifier %s:\n%s\n" % (svm, classification_report(y_test, predict)))


def Decision_Tree(X_train, y_train, X_test, y_test):
    
    # 建立一个决策树模型
    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    predict = dt.predict(X_test)
    print("accuracy scores : %.4f" % accuracy_score(predict, y_test))
    print("classification report for classifier %s:\n%s\n" % (dt, classification_report(y_test, predict)))


def main():

    X_train, y_train, X_test, y_test = load_data(style = "hist")

    # 采用像素精确度达到92.55%
    # 采用直方图精度达到34.17%
    #Logistic_Regression(X_train, y_train, X_test, y_test) 

    # 采用像素精确度达到92.71%
    # 采用直方图精度达到21.03%
    SVM(X_train, y_train, X_test, y_test)

    # 采用像素精确度达到87.80%
    # 采用直方图精度达到26.03%
    #Decision_Tree(X_train, y_train, X_test, y_test)

main()
  