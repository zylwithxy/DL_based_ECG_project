import os
import pandas as pd
from scipy import signal
import numpy as np
import scipy.io as sio

import torch
from torch.utils.data import Dataset


class Read_data_augmented():
    def __init__(self, path):

        self.result = self.sort_data(path)
        self.labels = self.read_ylabel(path)
        self.path = path

    def output(self):
        # Output 2 augmentation results
        X_temp_aug1, X_temp_aug2 = self.read_data_aug1(self.result, self.path), \
                                    self.read_data_aug2(self.result, self.path)

        return self.list2tensor(X_temp_aug1, X_temp_aug2)
    
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
        y_mlabel: Multiple labels of the training data (6877, 3)

        '''
        dataframe = pd.read_csv(os.path.join(path, 'REFERENCE1.csv'))
        y_mlabel = dataframe[['First_label','Second_label','Third_label']].values
        # _single_label = dataframe['First_label'].values
        # y_single_label = y_single_label.reshape(-1,1)
        assert len(y_mlabel) == 6877

        return y_mlabel

    def read_data_aug1(self, result, record_path):
        # 一次性把所有数据都读进来了，很占用显存
        # 训练数据
        X_aug1 = []
        for item in result:
            item_path = os.path.join(record_path, item)
            character = np.zeros((12, 15360), dtype = np.float32)
            array = sio.loadmat(item_path)['ECG'][0][0][2]
            array_resample = signal.resample_poly(array, 256, 500, axis=1) # 进行了FIR滤波
            character[:,-array_resample.shape[1]:] = array_resample if array_resample.shape[1] <= 15360 \
                                                     else array_resample[:,-15360:]
            X_aug1.append(character)
        
        assert len(X_aug1) == 6877

        return X_aug1
    
    def read_data_aug2(self, result, record_path):
        """
        result: List. filename of .mat files (6877, ) 
        """

        X_aug2 = []
        for item in result:
            item_path = os.path.join(record_path, item)
            character = np.zeros((12, 15360), dtype = np.float32)
            sig = sio.loadmat(item_path)['ECG'][0][0][2]
            sos = signal.butter(8, 35, btype='low', output= 'sos', fs= 500) # Butterworth filtering
            filtered = signal.sosfilt(sos, sig) # shape (12, L)
            character[:,-filtered.shape[1]:] = filtered if filtered.shape[1] <= 15360 \
                                               else filtered[:,-15360:]
            X_aug2.append(character)
        
        assert len(X_aug2) == 6877

        return X_aug2
    
    def list2tensor(self, X_aug1, X_aug2):
        """
        X_aug1: List
        X_aug2: List
        """
        X_aug1_tensor, X_aug2_tensor = torch.tensor(np.array(X_aug1)), \
                                       torch.tensor(np.array(X_aug2))

        return (X_aug1_tensor, X_aug2_tensor)


class TraindataSet_Aug2(Dataset):

    def __init__(self, Train_aug1, Train_aug2):
        """
        Train_aug1: Tensor.
        Train_aug2: Tensor.
        """
        super().__init__()
        self.train_aug1 = Train_aug1
        self.train_aug2 = Train_aug2
    
    def __getitem__(self, index):

        return self.train_aug1[index], self.train_aug2[index]

    def __len__(self):
        return len(self.train_aug1)