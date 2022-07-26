import os
from torch.utils.data import Dataset
import pandas as pd
import scipy.io as sio
from scipy import signal
import numpy as np
# import argparse

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
        return (self.read_data(self.result, self.path),self.labels)
    
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
            character[:,-array_resample.shape[1]:] = array_resample if array_resample.shape[1] <= 15360 else array_resample[:,-15360:]
            X.append(character)
            
        return X

if __name__ == "__main__":
    pass