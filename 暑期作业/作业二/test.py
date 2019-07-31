import numpy as np
import cv2
from data import load_data
from NB import *
from PCA import *
from Kmeans import *
import matplotlib.pyplot as plt 
from paper import *


def _Paper(X_train, y_train, X_test, y_test):

    paper = Paper()
    # 切块
    images_train_1 = paper.patch(X_train, patch_per_num=20, patch_size=12, type_class=0)
    images_train_2 = paper.patch(X_train, patch_size=12, stride=2, type_class=1)
    images_test = paper.patch(X_test, y_test, patch_size=12, stride=2, type_class=1)
    # 归一化
    pre_train_1 = paper.preprocessing(images_train_1)
    pre_train_2 = paper.preprocessing(images_train_2)
    pre_test = paper.preprocessing(images_test)
    print(pre_train_1.shape)
    print(pre_train_2.shape)
    print(pre_test.shape)
    
    # 降维
    #pca_train = paper.pca(pre_train, dim=25)
    #pca_test = paper.pca(pre_test, dim=25)
    # 聚类加池化
    images_train, images_test = paper.K_means(pre_train_1, pre_train_2, pre_test, k=50, small_blocks=81, large_blocks=9)
    print(images_train.shape)
    print(images_test.shape)
    # 映射

    paper.SVM(images_train, y_train, images_test, y_test)
    

def main():
    """
    分别对贝叶斯和kmeans进行预测，发现在贝叶斯上能达到84%的精确度，而在kmeans上只能达到11%精度。。。
    可能是我的算法有问题。。。
    """

    X_train, y_train, X_test, y_test = load_data()
    _Paper(X_train[:10000], y_train[:10000], X_test[:5000], y_test[:5000])

    """
    ret, dst1 = cv2.threshold(X_train, 50, 1, cv2.THRESH_BINARY)   # 对训练集进行二值操作，阀值设为50
    ret, dst2 = cv2.threshold(X_test, 50, 1, cv2.THRESH_BINARY)    # 对测试集进行二值操作，阀值设为50
    nb = NB()
    y_rate, temp_matrix = nb.train(dst1, y_train)
    predict = nb.predict(dst2, y_rate, temp_matrix)
    # 打印精度
    print("Test Accuracy: %.4lf" % np.mean(predict == y_test))
    """
    

    """
    pca = PCA(X_test[:10])
    lower_test = pca.reduction(dim = 100)
    print(lower_test.shape)

    for i in range(len(lower_test)):
        plt.subplot(2,5,i + 1)
        plt.imshow(np.reshape(lower_test[i], (10,10)))
    plt.show()
    """

    """
    kmeans = K_means(10)
    centers, labels = kmeans.train(X_train, y_train, num_iters = 20)
    print(labels)
    
    
    for i in range(10):
        plt.subplot(1,10,i + 1)
        plt.imshow(np.reshape(centers[i], (28, 28)))
        plt.axis("off")
    plt.show()
    

    predict = kmeans.predict(X_test, centers, labels)
    print("Test Accuracy: %.4lf" % np.mean(y_test == predict))

    """

    """
    for i in range(10):
        plt.subplot(1, 10, i + 1)
        plt.imshow(np.reshape(pca_output[i], (5, 5)))
        plt.axis("off")
    plt.show()
    """

main()