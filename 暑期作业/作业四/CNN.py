import numpy as np
import torch
import matplotlib.pyplot as plt


dtype = torch.float            # 设置数据类型
device = torch.device('cpu')   # 使用cpu计算

N, D_in, H, D_out = 64, 1000, 100, 10

x = torch.randn(N, D_in, device = device, dtype = dtype)
y = torch.randn(N, D_out, device=device, dtype = dtype)

# 这里需要将w1,w2设置成可以进行梯度计算的，即requires_grad设置成True
w1 = torch.randn(D_in, H, device = device, dtype = dtype, requires_grad=True)
w2 = torch.randn(H, D_out, device=device, dtype = dtype, requires_grad=True)

loss_history = []
learning_rate = 1e-07

for t in range(500):
    y_pred = x.mm(w1).clamp(min=0).mm(w2)    # 相当于矩阵运算

    loss = (y_pred - y).pow(2).sum()
    loss_history.append(loss)
    print(t, loss)

    """
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.t().mm(grad_y_pred)   # t相当于转置，计算w2的导数
    grad_h_relu = grad_y_pred.mm(w2.t())   # 计算h_relu的导数
    grad_h = grad_h_relu.clone()           # 对grad_h_relu进行复制
    grad_h[h<0] = 0                        # 计算h的导数
    grad_w1 = x.t().mm(grad_h)             # 计算w1的导数
    """

    # 上面的代码全可以采用下面的代码来代替
    # Pytorch可以采用这种方式来进行梯度计算
    loss.backward()

    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad

        # 需要手动将梯度设置为0
        w1.grad.zero_()
        w2.grad.zero_()

plt.figure(figsize=(6,8))
plt.plot(loss_history)
plt.show()