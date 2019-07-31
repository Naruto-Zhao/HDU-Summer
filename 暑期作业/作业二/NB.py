import numpy as np


class NB():

    def __init__(self):
        """
        初始化函数
        """

        # prediction 用于保存测试集每个图片的预测结果
        self.class_num = 10

    def train(self, X_train, y_train):
        """
        对训练集进行训练
        """
        
        num_train = X_train.shape[0]
        dimension = X_train.shape[1]
        y_rate = []
        temp_matrix = np.zeros((self.class_num, dimension, 2))
        
        # 计算每个类别的概率
        for j in range(self.class_num):

            class_j_index = np.where(y_train == j)[0]   # 保存类别j的图片序号
            j_rate = 1.0 * len(class_j_index) / num_train  # 计算p(y == j)
            y_rate.append(j_rate)                       # 保存p(y == j)
            class_j_x = X_train[class_j_index]          # 保存属于类别j的图片

            # 对每个维度进行计算
            for dim in range(dimension):

                # 计算p(x|y==j)
                temp = (class_j_x[:, dim] == 0)       # 计算像素值为0的概率
                temp_matrix[j][dim][0] = 1.0 * len(temp[temp == 1]) / len(class_j_x)
                temp = (class_j_x[:, dim] == 1)       # 计算像素值为1的概率
                temp_matrix[j][dim][1] = 1.0 * len(temp[temp == 1]) / len(class_j_x)
        
        return y_rate, temp_matrix


    def predict(self, X_test, y_rate, temp_matrix):
        """
        对测试集进行测试，返回预测值
        :param X_test: 测试集图片
        :param y_rate: p(y == j)的概率
        :param temp_matrix: 在训练集上的结果
        """

        num_test = X_test.shape[0]
        dimesion = X_test.shape[1]
        predict = []

        # 计算每个测试集图片的预测值
        for i in range(num_test):
            value = []

            # 计算每个类别的概率
            for cl in range(self.class_num):
                rate = 1.0
                for dim in range(dimesion):
                    rate *= temp_matrix[cl][dim][X_test[i][dim]]
                rate *= y_rate[cl]
                value.append(rate)
                
            # 返回最大概率的类别
            predict.append(np.argmax(value))

        return predict       