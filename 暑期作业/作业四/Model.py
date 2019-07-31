import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.datasets as dset
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torchvision.models as models
import torchvision.transforms as T
import torch.nn.functional as F


device = torch.device('cpu')     # 使用cpu进行训练
dtype = torch.float32            # 使用数据类型float32

def load_data():
    """
    此函数用于加载MNIST数据
    """
    transform = T.Compose([
        T.Resize(224),                            # 对图片大小进行修改
        T.ToTensor(),                             # 将PIL转化成tensor模式
        T.Normalize((0.1307,),(0.3081,))          # 对数据进行标准化
    ])

    NUM_TRAIN = 1000   # 这里只放1000张图片是因为我的电脑只能跑这么多，还是电脑太垃圾
    batch_size = 10    # batch_size也只能放这么多了

    mnist_train = dset.MNIST("./datasets", train=True, download = True, transform=transform)
    loader_train = DataLoader(mnist_train, batch_size=batch_size, 
                        sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN)))
    mnist_val = dset.MNIST("./datasets", train=True, download = True, transform=transform)
    loader_val = DataLoader(mnist_val, batch_size=batch_size, 
                        sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN, 1100)))
    mnist_test = dset.MNIST("./datasets", train=False, download = True, transform=transform)
    loader_test = DataLoader(mnist_test, batch_size=batch_size, 
                        sampler=sampler.SubsetRandomSampler(range(100)))
    print("Download is completed!")

    return loader_train, loader_val, loader_test


class ResNet_New(nn.Module):
    
    def __init__(self, in_channel, C_class):
        """
        Initialize the Model
        """
        super(ResNet_New, self).__init__()
        self.model = models.resnet152(pretrained=True)
        # 只对模型的第一层和最后一层进行修改，这样就可以保存预训练模型的大部分参数，可以加快训练的速度
        # 修改的主要内容是将三通道改为单通道，类别由1000改为了10
        self.model.conv1 = nn.Conv2d(in_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(512 * 4, C_class)

    def forward(self, x):
        """
        此函数用于计算前向传播
        """
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.model.fc(x)
        
        return x
 
def Accuracy(data_loader, model):
    """
    此函数用于计算模型在验证集或者测试集的性能
    """
    num_correct = 0
    num_samples = 0
    # 将模型模式转化成测试状态，因为模型中有使用的dropout或者batch norm时，其训练和测试的计算是不一样，
    # 所以需要将模型的状态从训练转化成测试状态
    model.eval()
    for x, y in data_loader:
        x = x.to(device=device, dtype=dtype)         # 需要将数据转成成float32类型
        y = y.to(device=device, dtype=torch.int64)   # 这里需要将数据转化成int64
        # 记住，tensor.max返回的是一个(value, indice)组合，max里面指定哪个轴
        _, out = model(x).max(1)
        num_correct += (out == y).sum()
        # 在进行个数统计时，也需要具体到某个维度
        num_samples += out.size(0)
    acc = float(num_correct) / num_samples
    print("Got %d / %d" % (num_correct, num_samples))

    return acc

def trian_predict(loader_train, loader_val, loader_test, optimizer, model):
    """
    对模型进行训练和测试
    """
    loss_history = []
    accuracy_history = []

    # 建立多次训练
    for epoch in range(1):
        for step, (batch_x, batch_y) in enumerate(loader_train):
            # 进入训练状态
            model.train()

            batch_x = batch_x.to(device=device, dtype=dtype)         # 这里需要将数据转化成float类型
            batch_y = batch_y.to(device=device, dtype=torch.long)    # 这里要将数据转化成long类型
            y_pred = model(batch_x)

            # 注意，交叉熵损失函数的input=(N,C)，是float类型，而target=(N),是long类型的
            # 计算损失函数,使用交叉熵损失函数,交叉熵函数自带softmax激活函数
            loss = F.cross_entropy(y_pred, batch_y)
            # 计算损失值,返回的是一个tensor,使用item反回其真实值
            loss_history.append(loss.item())

            # 将梯度缓冲区清空，否则这个缓冲区就会一致占用，这样内存占用就会越来越多
            optimizer.zero_grad()
            # 计算梯度
            loss.backward()
            # 参数更新
            optimizer.step()

            #if step % 50 == 0:
            acc = Accuracy(loader_val, model)
            print("Val Accuracy:%.4f" % (acc * 100))
            accuracy_history.append(acc)

    # 用于测试模型在测试集上的性能    
    print("--------------------------------")  
    acc = Accuracy(loader_test, model)
    print("Test Accuracy:%.4f" % (acc * 100))
    return loss_history, accuracy_history


def show(loss_history, accuracy_history):
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
    plt.plot(accuracy_history)
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.title('Accuracy')
    plt.show()


def main():
    # 加载数据
    loader_train, loader_val, loader_test = load_data()
    # 建立一个ResNet预训练模型
    model = ResNet_New(1, 10)
    print(model)
    # 采用的优化算法，这里使用的是优化算法Adam
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)    
    # 对模型进行训练和预测
    loss_history, accuracy_history = trian_predict(loader_train, loader_val, loader_test, optimizer, model)
    # 进行可视化
    show(loss_history, accuracy_history)

main()   