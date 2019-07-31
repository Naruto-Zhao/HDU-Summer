import numpy as np


class PCA():

    def __init__(self, X):
        
        self.X = X

    def reduction(self, dim = 10):
        """
        对数据进行降维
        """

        old_mean = np.mean(self.X, axis = 0)
        self.X -= old_mean                        # 对数据进行中心化 
        conv = np.cov(self.X, rowvar=0)           # 计算协方差矩阵
        eigValue, eigVect = np.linalg.eig(conv)   # 进行特征值分解

        eigValue_indice = np.argsort(-eigValue)[:dim]   # 取最大的dim个特征值
        eigVect_lower = eigVect[:, eigValue_indice]     # 取相应的特征向量
        lowerValue = self.X.dot(eigVect_lower)          # 进行降维操作

        return lowerValue

