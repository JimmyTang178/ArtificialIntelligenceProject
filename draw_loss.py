#画出训练loss
import matplotlib.pyplot as plt
import torch
from matplotlib.pylab import *
mpl.rcParams["font.sans-serif"] = ["SimHei"]
mpl.rcParams["axes.unicode_minus"] = False
loss_path = './loss_4.log'
ls = []
l = torch.load(loss_path)
print(max(l))
for j in l:
    ls.append(j)
plt.plot(ls)
plt.ylim(0,50)
plt.title("训练loss",fontsize = 20)
plt.show()