#全连接网络
import torch.nn as nn
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class Net(nn.Module):
    def __init__(self,input_num,hidden_num1,hidden_num2,hidden_num3,output_num):
        super(Net,self).__init__()
        self.fc1 = nn.Linear(input_num,hidden_num1)
        self.fc2 = nn.Linear(hidden_num1,hidden_num2)
        self.fc3 = nn.Linear(hidden_num2,hidden_num3)
        self.fc4 = nn.Linear(hidden_num3,output_num)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        y = self.fc4(x)
        return y
