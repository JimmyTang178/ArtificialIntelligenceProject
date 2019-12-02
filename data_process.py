#coding = utf-8
#选择信息增益最大的10维特征
import numpy as np
from InfoGain import choose_best_feature

data_path = "C:\\Users\\TJM\\OneDrive\\graduated\\研①\\人工智能算法与实践\\homework\\分享\\Test\\1-kddcup.data_10_percent_corrected"
test_path = "C:\\Users\\TJM\\OneDrive\\graduated\\研①\\人工智能算法与实践\\homework\\分享\\Test\\3-corrected.txt"
feature1 = []
feature2 = []
feature3 = []
label = []
def read_data(path):
    data = []
    with open(path) as f:
        for l in f.readlines():
            l = l.strip('.\n').split(',')
            if l[1] not in feature1:
                feature1.append(l[1])
            if l[2] not in feature2:
                feature2.append(l[2])
            if l[3] not in feature3:
                feature3.append(l[3])
            if l[41] not in label:
                label.append(l[41])
            data.append(l)
    return data

def split_data(data):
    data = np.array(data)
    features = data[:,0:41]
    labels = data[:,41]
    return features,labels

def quantify(feature,Label):
    feature1 = ['udp', 'tcp', 'icmp']
    feature2 = ['private', 'domain_u', 'http', 'smtp', 'ftp_data', 'ftp', 'eco_i', 'other', 'auth', 'ecr_i', 'IRC', 'X11', 'finger', 'time', 'domain', 'telnet', 'pop_3', 'ldap', 'login', 'name', 'ntp_u', 'http_443', 'sunrpc', 'printer', 'systat', 'tim_i', 'netstat', 'remote_job', 'link', 'urp_i', 'sql_net', 'bgp', 'pop_2', 'tftp_u', 'uucp', 'imap4', 'pm_dump', 'nnsp', 'courier', 'daytime', 'iso_tsap', 'echo', 'discard', 'ssh', 'whois', 'mtp', 'gopher', 'rje', 'ctf', 'supdup', 'hostnames', 'csnet_ns', 'uucp_path', 'nntp', 'netbios_ns', 'netbios_dgm', 'netbios_ssn', 'vmnet', 'Z39_50', 'exec', 'shell', 'efs', 'klogin', 'kshell', 'icmp']
    feature3 = ['SF', 'RSTR', 'S1', 'REJ', 'S3', 'RSTO', 'S0', 'S2', 'RSTOS0', 'SH', 'OTH']
    for j in range(len(feature)):
        if feature[j][2] not in feature1:
            feature2.append(feature[j][2])
        feature[j][1] = feature1.index(feature[j][1])+1
        feature[j][2] = feature2.index(feature[j][2])+1
        feature[j][3] = feature3.index(feature[j][3])+1
        if Label[j]=="normal":
            Label[j] = 1
        #elif Label[j] =="smurf":
            #Label[j] = 2
        else:
            Label[j] = 0
   # print(feature[99999])
    feature = feature.astype(float)
    Label = Label.astype(float)
    return feature,Label

if __name__ == "__main__":
    trainset = 1
    if trainset:#
        d = read_data(data_path)
        #print(d[0])
        print("lable种类：",label)
        features,labels = split_data(d)
        print(feature1)
        print(feature2)
        print(feature3)
        print(len(label))
        q_feature ,q_label= quantify(features,labels)
        new_data = np.array(np.column_stack((q_feature,q_label)))
        #np.savetxt("./dataset/train_data_41_bin.txt", new_data,fmt = '%f',delimiter=',')

        #np.savetxt("./dataset/train_data_41.txt",new_data)#保存10维训练数据     保存格式有点问题
        h= len(new_data)
        '''best_fearture_index,best_10_index = choose_best_feature(q_feature,q_label)#计算信息增益，取最大的前十个特征
        print("最佳特征：",best_fearture_index)
        print("最佳前10特征：",best_10_index)'''
        best_10_index = [5, 23, 3, 24, 36, 2, 33, 35, 34, 30]#计算得出的最佳特征维度
        best_5_index = [5, 23, 3, 24, 36]#取前5个特征来分类进行对比
        #new_data1 = np.zeros((h,11))
        #new_data1 = np.zeros((h, 6))
        new_data1 = np.zeros((h, 11))
        print(new_data.shape)
        print(new_data1.shape)
        best_10_index.append(42)
        best_5_index.append(42)
        '''for i in best_5_index:
            j=i-1
            new_data1[:,best_5_index.index(i)] = new_data[:,j]'''

        for i in best_10_index:
            j=i-1
            new_data1[:,best_10_index.index(i)] = new_data[:,j]
        np.savetxt("./dataset/train_data_10_bin.txt",new_data1,fmt='%f',delimiter=',')#保存10维训练数据'''
        #np.savetxt("./dataset/train_data_4.txt", new_data1, fmt='%f', delimiter=',')
    else:
        d = read_data(test_path)
        # print(d[0])
        features, labels = split_data(d)
        print(feature1)
        print(feature2)
        print(feature3)
        print(len(label))
        q_feature, q_label = quantify(features, labels)
        new_data = np.column_stack((q_feature, q_label))
        h = len(new_data)
        #np.savetxt("./dataset/test_data_41_bin.txt", new_data,fmt = "%f",delimiter=',')#保存41维的二分类测试机
        #best_fearture_index, best_10_index = choose_best_feature(q_feature, q_label)
        best_10_index = [5, 23, 3, 24, 36, 2, 33, 35, 34, 30]
        #print("最佳特征：", best_fearture_index)
        #print("最佳前10特征：", best_10_index)
        #new_data1 = np.zeros((h, 11))
        #np.savetxt("./dataset/test_data_41.txt", new_data)#保存41维测试数据
        best_5_index = [5, 23, 3, 24, 36]  # 取前5个特征来分类进行对比
        #new_data1 = np.zeros((h, 6))
        new_data1 = np.zeros((h, 11))
        print(new_data.shape)
        print(new_data1.shape)
        best_10_index.append(42)
        #best_5_index.append(42)
        '''for i in best_5_index:
            j = i - 1
            new_data1[:, best_5_index.index(i)] = new_data[:, j]'''
        #np.savetxt("./dataset/test_data_4.txt", new_data1, fmt='%f', delimiter=',')
        for i in best_10_index:
            j = i - 1
            new_data1[:, best_10_index.index(i)] = new_data[:, j]
        np.savetxt("./dataset/test_data_10_bin.txt", new_data1, fmt='%f', delimiter=',')#保存10维测试数据
