import torch
import torchvision
from torch.utils.data import DataLoader,Dataset
import numpy as np



class Mydata(Dataset):
    def __init__(self,txt,transform=None,target_transform=None):
        super(Mydata,self).__init__()
        f = open(txt,'r')
        feature = []
        for line in f:
            line = line.rstrip('\n')
            words = line.split(',')#用于10 维数据集
            #words = line.split(" ")#用于41维数据集
            feature.append(words)
        self.feature = feature
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        Feature= self.feature[index][0:10]#10维数据分类
        Label = self.feature[index][10]
        #Feature= self.feature[index][0:41]#41维数据分类
        #Label = self.feature[index][41]
        #Feature= self.feature[index][0:5]#5维数据分类
        #Label = self.feature[index][5]
        return  Feature,Label
    def __len__(self):
        return len(self.feature)

if __name__ =="__main__":
    path = "E:\\feature_extraction\\dataset\\train_data.txt"
    train_data = Mydata(path)
    print(train_data)

    train_loader = DataLoader(dataset=train_data,batch_size=2,shuffle=True)
    #for i in train_loader:
        #print(i)

