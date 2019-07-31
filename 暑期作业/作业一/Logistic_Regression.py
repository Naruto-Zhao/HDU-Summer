import numpy as np
import matplotlib.pyplot as plt


class Logistic_Regression():
    """
    本类用于实现逻辑回归，使用的梯度下降的优化方法
    """

    def __init__(self, W):

        self.W = W          # 初始化W值

    def one_calc(self, X_train, y_train, reg):
        """
        执行一次训练
        :param X_train: 训练集
        :param y_train: 标签
        """
        output = 1.0 / (1 + np.exp((-1) * (X_train.dot(self.W))))                 # 前向计算一次
        loss = (-1) * np.sum(y_train * np.log(output) + (1 - y_train) * np.log(1 - output))   # 计算似然函数的值
        loss = loss / X_train.shape[0] + reg * np.sum(self.W * self.W)            # 加上正则项
        
        # 计算W的梯度,使用反向传播算法
        dW = (-1) * np.sum((y_train - output).T * X_train.T, axis = 1).reshape((-1, 1))
        dW += 2 * reg * self.W   

        return loss, dW

    def train(self, X_train, y_train, num_iters = 1000, alpha = 0.01, reg = 0.01):
        """
        进行多次训练，不断优化w,b值
        :param X_train: 训练数据
        :param y_train: 标签
        :param num_iters: 训练迭代次数
        :param alpha: 学习率
        :param reg: 正则项系数        
        """

        loss_history = []
        for i in range(num_iters):

            loss, dW = self.one_calc(X_train, y_train, reg)
            # 不断更新W的值
            self.W -= alpha * dW 

            loss_history.append(loss)

        return loss_history

    def predict(self, X):
        
        output = 1.0 / (1 + np.exp((-1) * (X.dot(self.W))))   # 前向计算一次
        output[output >= 0.5] = 1
        output[output < 0.5] = 0

        return output.reshape(output.shape[0])

    def Show(self, loss):
        """
        此函数用于将数据和训练模型进行可视化

        :param loss: 损失值历史
        """

        plt.figure(figsize=(10,8))
        plt.subplot(1,1,1)
        plt.plot(loss)
        plt.title("Loss History")
        plt.xlabel("Train Numbers")
        plt.ylabel("Loss")

        plt.show()

def colicTest():            #对马这个数据进行处理
    '''
    读取数据
    :return: 列表形式
    '''

    frTrain = open('horse/horseColicTraining.txt')
    frTest = open('horse/horseColicTest.txt')
    trainingSet = []
    trainingLabels = []
    testSet = []
    testLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    for line in frTest.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        testSet.append(lineArr)
        testLabels.append(float(currLine[21]))

    X_train, y_train, X_test, y_test = np.array(trainingSet), np.array(trainingLabels), np.array(testSet), np.array(testLabels)
    
    # 对数据进行归一化，否则数据会溢出
    X_train = (X_train - np.mean(X_train, axis = 0)) / np.std(X_train, axis = 0)
    X_test = (X_test - np.mean(X_test, axis = 0)) / np.std(X_test, axis = 0)

    X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
    X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
    
    return X_train, y_train, X_test, y_test
    

def main():
    """
    这里的数据集为死马活马
    """

    X_train, y_train, X_test, y_test = colicTest()

    W = np.random.randn(X_train.shape[1],1)

    classifier = Logistic_Regression(W)
    loss =  classifier.train(X_train, y_train.reshape((-1,1)), num_iters=1000, alpha=0.0001, reg = 0.0001)
    
    X_train_predict = classifier.predict(X_train)
    X_test_predict = classifier.predict(X_test)

    # 分别输出模型在训练集和测试集中的精度
    print("Train accuracy: %.4lf" % np.mean(X_train_predict == y_train))
    print("Test accuracy: %.4lf" % np.mean(X_test_predict == y_test))
    classifier.Show(loss)

main()
