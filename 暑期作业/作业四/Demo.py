import torch
import numpy as np
import matplotlib.pyplot as plt
from data import load_data
import torch.utils.data as Data
import torch.nn.functional as F 

class TwoLayerNet(torch.nn.Module):
    """
    Set up a two_layer Network by a subclass extenting the superclass torch.nn.Module
    Conv-->BN-->Relu-->Conv-->BN-->Relu-->Maxpool-->Linear-->Relu-->Linear-->Softmax
    """
    def __init__(self, D_in, H, C_class):
        """
        Initialize the Net
        任何需要进行学习的参数层倒要放到这个里面
        """
        super(TwoLayerNet, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, 
                            out_channels=5, 
                            kernel_size=5, 
                            padding=2),
            torch.nn.BatchNorm2d(5, affine=True),    # 加上这个精度提升不少
            torch.nn.ReLU(inplace=True)
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=5, 
                            out_channels=5, 
                            kernel_size=3, 
                            padding=1),
            torch.nn.BatchNorm2d(5, affine=True),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2)
        )
        self.linear1 = torch.nn.Linear(D_in, H)
        self.relu = torch.nn.ReLU(inplace=True)
        self.linear2 = torch.nn.Linear(H, C_class)

    def forward(self, x):
        """
        Compute the forward pass
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.shape[0], -1)
        h_relu = self.relu(self.linear1(x))
        y_pred = self.linear2(h_relu)

        return y_pred


def data_load():
    """
    加载数据，将ndarray转化成tensor
    """
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()

    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).long()
    X_val = torch.from_numpy(X_val).float()
    y_val = torch.from_numpy(y_val).long()
    X_test = torch.from_numpy(X_test).float()
    y_test = torch.from_numpy(y_test).long()

    return X_train, y_train, X_val, y_val, X_test, y_test


def Accuracy(X_train, y_train, X_val, y_val, model):
    
    model.eval()
    N = X_train.shape[0]

    # 对一部分训练集和验证集进行预测，用于检验模型的性能
    indice = np.random.choice(N, size=10000)
    train = X_train[indice]
    # 记住，tensor.max返回的是一个(value, indice)组合，max里面指定哪个轴
    _, train_out = model(train).max(1)
    num_correct = (train_out == y_train[indice]).sum()
    # 在进行个数统计时，也需要具体到某个维度
    num_samples = train_out.size(0)
    train_acc = float(num_correct) / num_samples
    print("Got %d/%d train examples" % (num_correct, num_samples))
    print("Train Accuracy:%.4f" % (train_acc * 100))

    _, val_out = model(X_val).max(1)
    num_correct = (val_out == y_val).sum()
    num_samples = val_out.size(0)
    val_acc = float(num_correct) / num_samples
    print("Got %d/%d val examples" % (num_correct, num_samples))
    print("Val Accuracy:%.4f" % (val_acc * 100))

    return train_acc, val_acc


def train_predict(X_train, y_train, X_val, y_val, X_test, y_test, optimizer, model):
    """
    对模型进行训练，并对测试集进行预测
    """

    loss_history = []
    train_accuracy_history = []
    val_accuracy_history = []

    batch_size = 500
    # 采用mini batch形式,每1个epoch清洗一下
    torch_dataset = Data.TensorDataset(X_train, y_train)
    loader = Data.DataLoader(
        dataset=torch_dataset,    # 数据集
        batch_size=batch_size,    # batch_size大小
        shuffle=True,             # 每个epoch是否进行重新排列
        num_workers=2             # set multi-work num read data
    )

    # 建立多次训练
    for epoch in range(10):
        for step, (batch_x, batch_y) in enumerate(loader):
            # 进入训练状态
            model.train()
            y_pred = model(batch_x)

            # 注意，交叉熵损失函数的input=(N,C)，是float类型，而target=(N),是long类型的
            # 交叉熵自带softmax函数
            loss = F.cross_entropy(y_pred, batch_y)
            # 计算损失值,返回的是一个tensor,使用item反回其真实值
            loss_history.append(loss.item())

            # 将梯度缓冲区清空，否则这个缓冲区就会一致占用，这样内存占用就会越来越多
            optimizer.zero_grad()
            # 计算梯度
            loss.backward()
            # 参数更新
            optimizer.step()

            if step % 25 == 0:
                # 进入测试状态
                train_acc, val_acc = Accuracy(X_train, y_train, X_val, y_val, model)
                train_accuracy_history.append(train_acc)
                val_accuracy_history.append(val_acc)
    
    # 首先进入测试状态，再对测试集进行预测
    model.eval()
    _, test_out = model(X_test).max(1)
    num_correct = (test_out == y_test).sum()
    num_samples = test_out.size(0)
    acc = float(num_correct) / num_samples
    print("-------------------------------------------")
    print("Got %d/%d test examples" % (num_correct, num_samples))
    print("Test Accuracy:%.4f" % (acc * 100))

    return loss_history, train_accuracy_history, val_accuracy_history


def show(loss_history, train_accuracy_history, val_accuracy_history):
    """ 
    对结果进行可视化
    """
    plt.figure(figsize=(6,8))
    plt.subplots_adjust(hspace=0.5)

    plt.subplot(2,1,1)
    plt.plot(loss_history)
    plt.xlabel('Iterations')
    plt.ylabel('loss')
    plt.title("Loss")

    plt.subplot(2,1,2)
    plt.plot(train_accuracy_history, label = "Train")
    plt.plot(val_accuracy_history, label ='Val')
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.title('Accuracy')
    plt.legend(ncol=2, loc='lower right')
    plt.show()


def main():
    X_train, y_train, X_val, y_val, X_test, y_test = data_load()
    
    D_in, H, C_class = 14 * 14 * 5, 20, 10
    # Set up a model
    model = TwoLayerNet(D_in, H, C_class)
    print(model)
    # 采用的优化算法，这里使用的是优化算法Adam
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # 保存训练记录
    loss_history, train_accuracy_history, val_accuracy_history = \
    train_predict(X_train, y_train, X_val, y_val, X_test, y_test, optimizer, model)
    # 可视化
    show(loss_history, train_accuracy_history, val_accuracy_history)


main()