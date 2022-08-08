from timm.utils import random_seed
import os
import scipy.io as sio
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import scipy
from scipy import signal
import torch
from torch import nn
import STA_CRNN.CBAM as CBAM
from .CBAM import CBAMBlock
from torch.utils.data import DataLoader, Dataset
from .k_fold_CV import get_k_fold_data as kfold
from .k_fold_CV import evaluate_accuracy_gpu

import sys       # For data augmentation
PATH, PATH2 = '/home/alien/XUEYu/paper_code/enviroment1/ECG_GAN',\
              '/home/alien/XUEYu/paper_code/enviroment1/VAN' 

sys.path.append(PATH)
sys.path.append(PATH2)
from load_generator import LoadData
from trainer_frame_block import load_synthetized_data

# 1. Model part

# VGG Block
def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv1d(in_channels, out_channels,
                                kernel_size=3, padding=1))
        in_channels = out_channels
        layers.append(nn.BatchNorm1d(in_channels))
        layers.append(nn.ReLU())

    layers.append(nn.MaxPool1d(3))
    layers.append(nn.Dropout(0.2))
    
    return nn.Sequential(*layers)

def vgg(conv_arch, in_channels):
    conv_blks = []
    # 卷积层部分
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels
        conv_blks.append(CBAMBlock(channel = in_channels))
        # CBAMBlock.init_weights()

    # Global feature extraction
    # conv_blks.append()
    conv_blks.pop() # 排除最后一个attention 22/03/14
    return nn.Sequential(
        *conv_blks)

# 1.1. 封装一个交换维度的类
class Changeaxis(nn.Module):
    def __init__(self):
        super(Changeaxis, self).__init__()
    def forward(self, X):
        assert X.ndim == 3
        return X.permute(0,2,1)

# 1.2. 封装一个只输出GRU输出的类
class GRU_output(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, X):
        return X[0]

def try_gpu(i=0):  #@save
    """如果存在，则返回gpu(i)，否则返回cpu()"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

# Parameters
# X = torch.rand((64,12,15360))

# Block a big class (The first of four Test model)

def build_net(conv_arch):

    net_ECG = nn.Sequential(vgg(conv_arch,12), 
                        Changeaxis(),
                        nn.GRU(input_size = 256,hidden_size = 12,num_layers = 2,
                               dropout=0.2,batch_first=True, bidirectional=True),
                        GRU_output(),
                        Changeaxis(),
                        nn.MaxPool1d(63),
                        nn.Flatten(),
                        nn.Linear(24, 9)
                        )
    """
    
    del net_ECG
    net_ECG = nn.Sequential(
                            vgg(conv_arch,12), # shape(B, 256, 63)
                            nn.MaxPool1d(63),
                            nn.Flatten(),
                            nn.Linear(256, 9)
                           )
    """

    return net_ECG

# No need to add softmax

#2. Parameters initialization

def parameter_init(net_ECG):

    for i in net_ECG:
        if type(i) == nn.GRU:
            # print("nn.GRU")
            nn.init.orthogonal_(i.weight_hh_l0)
            nn.init.orthogonal_(i.weight_hh_l0_reverse)
            nn.init.orthogonal_(i.weight_hh_l1)
            nn.init.orthogonal_(i.weight_hh_l1_reverse)
            nn.init.orthogonal_(i.weight_ih_l0)
            nn.init.orthogonal_(i.weight_ih_l0_reverse)
            nn.init.orthogonal_(i.weight_ih_l1)
            nn.init.orthogonal_(i.weight_ih_l1_reverse)

            nn.init.zeros_(i.bias_hh_l0)
            nn.init.zeros_(i.bias_hh_l0_reverse)
            nn.init.zeros_(i.bias_hh_l1)
            nn.init.zeros_(i.bias_hh_l1_reverse)
            nn.init.zeros_(i.bias_ih_l0)
            nn.init.zeros_(i.bias_ih_l0_reverse)
            nn.init.zeros_(i.bias_ih_l1)
            nn.init.zeros_(i.bias_ih_l1_reverse)

        if type(i) == nn.Sequential:
            for j in i:
                if type(j) == CBAM.CBAMBlock:
                    # print("CBAM")
                    j.init_weights()
                if type(j) == nn.Sequential:
                    for k in j:
                        if type(k) == nn.Conv1d:
                            # print("Conv1d")
                            nn.init.xavier_uniform_(k.weight)
                            nn.init.zeros_(k.bias)
        if type(i) == nn.Linear:
            # print("nn.Linear")
            nn.init.xavier_uniform_(i.weight)
            nn.init.zeros_(i.bias)        

# 3. 数据的读取过程 以其中一个数据为例子
class TraindataSet(Dataset):
    def __init__(self,TrainX, TrainY):
        super(TraindataSet, self).__init__()
        self.train = TrainX
        self.label = TrainY
    
    def __getitem__(self,index):
        return self.train[index], self.label[index]

    def __len__(self):
        return len(self.train)

# 想实现的思路是不要把所有的数据一口气全部读进来，用什么读什么，但这样花费时间更长，还是一气儿全部读进来吧
# 弄一个读取数据的大类
class Read_data():
    def __init__(self, path):

        self.result = self.sort_data(path)
        self.labels = self.read_ylabel(path)
        self.path = path

    def output(self):
        return (self.read_data(self.result, self.path), self.labels)
    
    def sort_data(self, record_path):
        result = []
        for mat_item in os.listdir(record_path):
            if mat_item.endswith('.mat'):
                result.append(mat_item)
        result.sort()
        return result
    
    def read_ylabel(self, path):
        '''
        Output:
        y_single_label: The first label of the training data
        y_mlabel: Multiple labels of the training data

        '''
        dataframe = pd.read_csv(os.path.join(path, 'REFERENCE1.csv'))
        y_mlabel = dataframe[['First_label','Second_label','Third_label']].values
        # _single_label = dataframe['First_label'].values
        # y_single_label = y_single_label.reshape(-1,1)
        assert len(y_mlabel) == 6877

        return y_mlabel

    def read_data(self, result, record_path):
        # 一次性把所有数据都读进来了，很占用显存
        # 训练数据
        X = []
        for item in result :
            item_path = os.path.join(record_path, item)
            character = np.zeros((12, 15360), dtype = np.float32)
            array = sio.loadmat(item_path)['ECG'][0][0][2]
            array_resample = signal.resample_poly(array, 256, 500, axis=1) # 进行了FIR滤波
            character[:,-array_resample.shape[1]:] = array_resample[:,-array_resample.shape[1]: ] if array_resample.shape[1] <= 15360 else array_resample[:,-15360:]
            X.append(character)
        return X

# 4. 模型的训练过程(K折交叉验证)

def data_generate(k, i, Mydata, labels, samples):
    '''
    i: [0, k-1].
    labels: List. Labels for different samples.
    samples: List. The number of samples separately.
    '''
    X_train, y_train, X_test, y_test = kfold(k,i,*Mydata.output())
    y_train, y_test = y_train -1, y_test - 1

    for label, sample in zip(labels, samples):
        X_generated, y_generated = load_synthetized_data(label, sample) # (sample, 12, 5500) (sample, 3) [Tensor]*2
        X_compliment = torch.zeros(sample, 12, 9860) # 15360-550
        X_generated = torch.cat((X_compliment, X_generated), dim= -1)
        
        X_train = torch.cat((X_train, X_generated), dim= 0)
        y_train = torch.cat((y_train, y_generated), dim= 0)
    
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    return X_train, y_train, X_test, y_test

def train_experiment(net, num_gpus, batch_size, lr, epoches, k_num, fold_i, 
                     Mydata, labels, samples):
    # 单折训练 
    X_train, y_train, X_test, y_test = data_generate(k_num, fold_i, Mydata, labels, samples)

    Train_data = TraindataSet(X_train, y_train)
    Test_data = TraindataSet(X_test, y_test)
    train_iter, test_iter = DataLoader(Train_data, batch_size = batch_size, shuffle = True), \
                            DataLoader(Test_data, batch_size = batch_size, shuffle = True)
    
    print("Finishing dataloading")

    devices = [try_gpu(i) for i in range(num_gpus)]
    net = nn.DataParallel(net, device_ids=devices)
    optimizer = torch.optim.Adam(net.parameters(), lr = lr)
    loss = nn.CrossEntropyLoss()
    max_F1, count_patience, PATIENCE = 0, 0, 20  # Count the number of times that is not the maximum F1 score

    for epoch in range(epoches):
        print(f"Fold{fold_i + 1} Epoch {epoch+1}\n-------------------------------")
        net.train()
        for batch, (X, y) in enumerate(train_iter):
            optimizer.zero_grad()
            X, y = X.to(devices[0]), y[:,0].to(devices[0])
            # y = y.type(torch.int64)
            # print(type(y), y.dtype)
            l = loss(net(X), y)
            l.backward()
            optimizer.step()
            if batch % 10 == 0:
                losserror, current = l.item(), min(batch_size * (batch+1), y_train.shape[0])
                print(f"loss: {losserror:>7f}  [{current:>5d}/{y_train.shape[0]:>5d}]")
        
        test_accu, F1_score, confuse_mat= evaluate_accuracy_gpu(net, test_iter) # test_accu need to be superseded by average F1 score.
        print(f"Test Error: \n Accuracy: {(100*test_accu):>0.1f}% F1 score:{100 * F1_score:>4f}% \n")
        
        if F1_score >= max_F1:
            net_parameters = net.state_dict()
            confuse_mat_return = confuse_mat
            max_F1 = F1_score
            count_patience = 0
        else:
            count_patience += 1
            print(f'EarlyStopping counter: {count_patience} out of {PATIENCE}\n')
            if count_patience >= PATIENCE:
                break
        
    print("Finishing training")
    return net_parameters, confuse_mat_return, max_F1

if __name__ == '__main__':

    # 5. Train Model
    random_seed()

    # Parameters
    LR = 1e-3
    BATCH_SIZE = 128
    EPOCHES = 120
    LABELS = [8]
    SAMPLES = [150]
    path, path2 = '/home/alien/XUEYu/paper_code/enviroment1', \
                  '/home/alien/XUEYu/paper_code/Parameters/2018_torch'
    conv_arch = [(2,64),(2,128),(3,256),(3,256),(3,256)]

    # 5.1 Set up three models
    net_array = [build_net(conv_arch) for _ in range(4)]
    # print(net_array[0]) # print the structure of net

    # 5.2 Transfer model to the GPU
    for net_ECG in net_array:
        net_ECG = net_ECG.to(device = try_gpu(0))

    # 5.3 Parameters initialization
    for net_ECG in net_array:
        parameter_init(net_ECG)
    
    # 5.4 Reading Training data
    Mydata = Read_data(path) # 读取所有的训练数据和相应的标签

    # 5.5 Traing 3 models
    fn = 'Data_aug_5500' # folder name
    fpath = os.path.join(path2, fn)
    if not os.path.exists(fpath):
        os.makedirs(fpath)
    
    fcount = len(os.listdir(fpath)) // 2

    for k in range(1):
        net_kfold_params, confuse_mat, max_F1 = train_experiment(net_array[k], 2, BATCH_SIZE, 
                                                                LR, EPOCHES, 4, k, Mydata, LABELS, SAMPLES)
        print(max_F1)

        torch_name, np_name = f'Label{LABELS[0]}_Sample{SAMPLES[0]}_{k+1}fold Test_model_'+f'model_weights{fcount+1}.pth',\
                              f'Label{LABELS[0]}_Sample{SAMPLES[0]}_{k+1}fold Test_model_'+f'confuse_mat{fcount+1}.npy'
        torch.save(net_kfold_params, os.path.join(fpath, torch_name))
        np.save(os.path.join(fpath, np_name), confuse_mat)