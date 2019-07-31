import numpy as np
import struct


def load_data():
    """
    加载数据
    """

    X_train_path = "MNIST/train-images-idx3-ubyte"
    y_train_path = "MNIST/train-labels-idx1-ubyte"
    X_test_path = "MNIST/test-images-idx3-ubyte"
    y_test_path = "MNIST/test-labels-idx1-ubyte"

    X_train = image_decode(X_train_path)
    #X_train = np.reshape(X_train, (X_train.shape[0], -1))
    y_train = label_decode(y_train_path)

    X_test = image_decode(X_test_path)
    #X_test = np.reshape(X_test, (X_test.shape[0], -1))
    y_test = label_decode(y_test_path)

    # 将float转化为uint8类型
    
    """
    X_train = X_train.astype(np.uint8)
    X_test = X_test.astype(np.uint8)
    y_train = y_train.astype(np.uint8)
    y_test = y_test.astype(np.uint8)
    """
    
    # 对特征进行缩放,进行max-min归一化
    #X_train_max, X_train_min = np.max(X_train), np.min(X_train)
    #X_train = (X_train - X_train_min) / (X_train_max - X_train_min)
    #X_test_max, X_test_min = np.max(X_test), np.min(X_test)
    #X_test = (X_test - X_test_min) / (X_test_max - X_test_min)

    # 将数据集和测试集的图片部分进行扩充
    #X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
    #X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
    
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
