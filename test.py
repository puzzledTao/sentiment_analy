import torch.nn.functional as F
import torch
from torch import nn
# a = torch.rand(10, 3)
# b = torch.rand(2,3)


# c = F.linear(a, b)

# print(c)

# a = torch.tensor([0,1,0])
# c = F.one_hot(a, num_classes=2)
# print(c)

# # label = torch.tensor([2])  # 2显示的是索引
# # num_class = 5
# # label2one_hot = F.one_hot(label, num_classes=num_class)

# # print(label2one_hot)、
import math

# class Net(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv = nn.Conv2d(in_channels=1,out_channels=1,kernel_size=2,stride=1,padding=0)   
#     def forward(self,x):
#         out = self.conv(x)
#         return out
 
# net = Net()

# def rule(epoch):
#     lamda = math.pow(0.95, epoch)
#     return lamda
# optimizer = torch.optim.SGD([{'params': net.parameters(), 'initial_lr': 1.41e-05}], lr = 0.1)
# scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = rule)

# for i in range(20):
#     print("lr of epoch", i, "=>", scheduler.get_lr())
#     optimizer.step()
#     scheduler.step()

a = torch.tensor([123, 2334])
b = torch.stack(a)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._model.to(device) 
print(b)

