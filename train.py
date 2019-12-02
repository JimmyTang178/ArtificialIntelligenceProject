import numpy as np
from model import Net
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn as nn
from dataset import Mydata

#device = torch.device("cuda" if torch)
train_path = "./dataset/train_data.txt"
test_path = "./dataset/test_data.txt"
train_path_41 = "./dataset/train_data_41.txt"
test_path_41 = "./dataset/test_data_41.txt"
train_path_4 ="./dataset/train_data_4.txt"
test_path_4 = "./dataset/test_data_4.txt"
train_path_41_bin = "./dataset/train_data_41_bin.txt"
test_path_41_bin =  "./dataset/test_data_41_bin.txt"
train_path_10_bin = "./dataset/train_data_10_bin.txt"
test_path_10_bin = "./dataset/test_data_10_bin.txt"
save_path = "./model.pt"
loss_path = "./loss.log"
save_path_41 = "./model_41.pt"
loss_path_41 = "./loss_41.log"
save_path_4 = "./model_4.pt"
loss_path_4 = "./loss_4.log"
save_path_41_bin = "./model_41_bin.pt"
save_path_10_bin = "./model_10_bin.pt"

epochs = 20
lr = 0.001
input_num = 10#用于10维特征分类
#input_num = 41#用于未经选择的41维特征分类
#input_num = 5
hidden_num1 = 50
hidden_num2 = 100
hidden_num3 = 60
#output_num = 3
output_num = 2#用于2分类
mydata = Mydata(train_path_10_bin)#加载训练集
train_loader = DataLoader(mydata,batch_size=64,shuffle=True)
testdata = Mydata(test_path_10_bin)#加载测试集
test_loader = DataLoader(testdata,batch_size=64,shuffle=True)
model = Net(input_num,hidden_num1,hidden_num2,hidden_num3,output_num)#构建网络
#model.to(device)
criterion = nn.CrossEntropyLoss()#损失函数
optimizer = optim.Adam(model.parameters(),lr=lr)#优化器
train_loss = []
if __name__ == "__main__":
    for epch in range(epochs):#进行训练，如已有训练模型则注释该段代码
        for i,data in enumerate(train_loader):
            (features,labels) = data
            #print("labels:",labels)

            features = np.array(features).astype(float).T
            labels = np.array(labels).astype(float)
            labels = labels.astype(np.int)
            features = torch.Tensor(features)
            labels = torch.LongTensor(labels)
            features = Variable(features)
            labels = Variable(labels)

            #print(type(features))
            #print("Label种类：",len(labels))
            # print("产生的特征",features)
            # print("对应的标签",labels)
            output = model(features)
            #print(output.shape,labels.shape)
            loss = criterion(output,labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i+1)%100 == 0:
                print('Epoch [{}/{}],loss:{:.4f}'.format(epch+1,epochs,loss.item()))
                train_loss.append(loss.item())
    print("训练完毕")
    torch.save(model.state_dict(),save_path_10_bin)
    #torch.save(train_loss,loss_path_4)
    print("训练LOSS：",train_loss)
    print("训练最大Loss：",max(train_loss))
    #model.load_state_dict(torch.load(save_path))#加载预训练模型，注释上面的训练代码

