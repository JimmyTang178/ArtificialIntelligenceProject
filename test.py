import torch
from torch.utils.data import DataLoader
import numpy as np
from train import model
from dataset import Mydata

if __name__ == "__main__":
    #test_data_path = './dataset/test_data.txt'
    #model_path = './model.pt'
    #test_data_path = './dataset/test_data_41.txt'#41维数据集测试
    #model_path = './model_41.pt'
    test_data_path = './dataset/test_data_10_bin.txt'#5维数据集测试
    model_path = './model_10_bin.pt'
    test_data = Mydata(test_data_path)

    test_loader = DataLoader(test_data,batch_size=64,shuffle=True)
    model.load_state_dict(torch.load(model_path))
    with torch.no_grad():#进行测试
        correct = 0
        total = 0
        for features,labels in test_loader:
            #print(type(features))
            features = np.array(features).astype(float).T
            features = torch.Tensor(features)
            labels = np.array(labels).astype(float)
            labels = torch.LongTensor(labels)
            output = model(features)
            #print(output)
            _,predicted = torch.max(output,1)
            #print(labels)
            #print(labels.size())
            total+=labels.size(0)#获取总的样本数
            #print("label" ,labels.shape)
            #print("predicted ",predicted.shape)
            correct+=(predicted == labels).sum().item()
        print("2分类10维测试集准确率:{:.2f}".format(100*correct/total))