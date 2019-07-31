import numpy as np

class K_means():
    """
    建立一个k-means分类器，实现在mnist上的多分类
    """

    def __init__(self, K):
        """
        k为聚类中心的个数
        """

        self.K = K
    
    def train(self, X_train, y_train, num_iters):
        """
        对训练集进行训练
        :param X_train: 训练集图片
        :param y_test: 训练集标签
        :param num_iters: 训练次数 
        """

        num_train = X_train.shape[0]

        # 随机选择k个聚类中心
        K_center = np.random.choice(num_train, self.K, replace = False)
        centers = X_train[K_center]

        clusters = []
        
        # 进行多次训练，要么达到最大训练次数退出，要么聚类中心不再变化退出
        for num in range(num_iters):
            clusters = []
            flag = 0
            for i in range(self.K):
                clusters.insert(1,[])

            # 分别计算一个样例到各个聚类中心的距离，找到距离最小的那个聚类中心，将该样例归为该聚类的一员
            for i in range(num_train):
                dist = np.sum((centers - X_train[i])**2, axis = 1)
                index = np.argmin(dist)
                clusters[index].append(i)
            clusters = np.array(clusters)

            # 更新聚类中心
            for i in range(self.K):
                if len(clusters[i]) == 0:
                    continue
                train = X_train[clusters[i]]
                mean_value = np.mean(train, axis = 0)
                if (mean_value == centers[i]).all():
                    pass
                else:
                    centers[i] = mean_value
                    flag = 1

            # 聚类中心不再变化，退出
            if flag == 0:
                print(num)
                break

        """
        labels = []
        for i in range(self.K):
            if len(clusters[i]) == 0:
                labels.append(0)
                continue
            label = np.argmax(np.bincount(y_train[clusters[i]]))
            labels.append(label)
        """

        return centers

    def predict(self, X_test, centers, labels):
        """
        对测试集进行预测
        :param X_test: 测试集图片
        :param centers: 聚类中心
        :param centers_class: 聚类中心的类别
        """

        num_test = X_test.shape[0]
        predict = []

        for i in range(num_test):
            dist = np.sum((centers - X_test[i])**2, axis =1)  # 计算测试样例到每个聚类中心的距离
            index = np.argmin(dist)                           # 选择距离最小的那个聚类
            predict.append(labels[index])                     # 将样例的类别授予聚类中心的类别

        return predict