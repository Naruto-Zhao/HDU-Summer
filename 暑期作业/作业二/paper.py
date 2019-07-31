import numpy as np
import random
from PCA import *
from Kmeans import *
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.cluster import KMeans


class Paper():
    """
    按照论文的要求来写
    """

    def patch(self, X, patch_per_num = 5, patch_size = 10, stride = 2, type_class = 0):
        """
        对图像切块，切块是随意的，个数也是随意的
        :param patch_per_num: 每个图片进行切块的个数
        :param patch_size: 切块的尺寸
        """

        # 切块的图片
        images = []
        num = X.shape[0]

        # 第一次切
        if type_class == 0:
            for i in range(num):
                # 将每个图片的切块个数限制在一个范围
                blocks = random.randrange(5, patch_per_num)
                for j in range(blocks):
                    # 初始化起点，注意边界情况
                    row = random.randrange(0, 28 - patch_size + 1)
                    col = random.randrange(0, 28 - patch_size + 1)

                    # 切块
                    patch = X[i][row:row+patch_size, col:col + patch_size]
                    patch = patch.reshape(patch_size * patch_size)
                    images.append(patch)
        # 第二次切
        else:
            for i in range(num):
                # 将每个图片的切块个数限制在一个范围
                blocks = (28 - patch_size) // stride + 1
                for row in range(blocks):
                    # 初始化起点，注意边界情况
                    for col in range(blocks): 
                        # 切块
                        patch = X[i][row*stride:row*stride+patch_size, col*stride: col*stride+patch_size]
                        patch = patch.reshape(patch_size * patch_size)
                        images.append(patch)

        return np.array(images)

    def preprocessing(self, images):
        """
        数据预处理，对数据进行归一化操作
        """

        output = (images - np.mean(images, axis = 0)) / np.std(images, axis = 0)

        return output

    def pca(self, X, dim = 25):
        """
        进行PCA降维
        :param X: 图片
        :param dim: 将维后图片维度
        """

        pca = PCA(X)
        output = pca.reduction(dim = 25)

        return output

    def K_means(self, X1, X2, X, k=10, small_blocks = 81, large_blocks=9):
        """
        对切块后的数据进行聚类
        :param X: 图片
        :param y: 标签
        :param k: 聚类中心个数
        """

        kmeans = KMeans(n_clusters=k, random_state=0).fit(X1)
        predict_train = kmeans.predict(X2)
        predict_test = kmeans.predict(X)
        count1 = X2.shape[0] // small_blocks
        count2 = X.shape[0] // small_blocks
        # 处理一张图片
        images_train = []
        images_test = []

        # 训练集进行池化
        for i in range(count1):
            per_image = np.zeros((k, 3, 3))
            one_predict = predict_train[small_blocks*i:small_blocks*(i+1)]
            one_hot = one_predict.reshape((9, 9))
            for row in range(3):
                for col in range(3):
                    patch = one_hot[row*3:(row+1)*3, col*3:(col+1)*3]
                    for l in range(k):
                        if l in patch:
                            per_image[l][row][col] = 1

            image = per_image.reshape(k * 3 * 3)
            images_train.append(image)

        # 测试集进行池化
        for i in range(count2):
            per_image = np.zeros((k, 3, 3))
            one_predict = predict_test[small_blocks*i:small_blocks*(i+1)]
            one_hot = one_predict.reshape((9, 9))
            for row in range(3):
                for col in range(3):
                    patch = one_hot[row*3:(row+1)*3, col*3:(col+1)*3]
                    for l in range(k):
                        if l in patch:
                            per_image[l][row][col] = 1

            image = per_image.reshape(k * 3 * 3)
            images_test.append(image)

        return np.array(images_train), np.array(images_test)


    def SVM(self, X_train, y_train, X_test, y_test):
        """
        建立一个支持向量机模型
        """

        svm = SVC(gamma= "scale")
        svm.fit(X_train, y_train)
        predict = svm.predict(X_test)
        print("accuracy scores: %.4lf" % accuracy_score(predict, y_test))
        print("classificatiom report for classifier %s:\n%s\n" % (svm, classification_report(y_test, predict)))
