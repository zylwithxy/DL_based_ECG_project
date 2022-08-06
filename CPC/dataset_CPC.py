import torch
from torch.utils.data import Dataset
import sys

# Self
PATH = '/home/alien/XUEYu/paper_code/enviroment1'
sys.path.append(PATH)

from STA_CRNN.k_fold_CV import get_k_fold_data as kfold

def data_generate(k, i, Mydata):
    '''
    i: [0, k-1].
    Mydata: instantiation of the class
    '''
    X_train, y_train, X_test, y_test = kfold(k,i,*Mydata.output())
    y_train, y_test = y_train -1, y_test - 1
    
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    return X_train, y_train, X_test, y_test


class TraindataSet(Dataset):
    def __init__(self,TrainX):
        super(TraindataSet, self).__init__()
        self.train = TrainX
    
    def __getitem__(self, index):
        return self.train[index]

    def __len__(self):
        return len(self.train)


class TraindataSet_total(Dataset): # For fine-tuning
    def __init__(self,TrainX, TrainY):
        super(TraindataSet_total, self).__init__()
        self.train = TrainX
        self.label = TrainY
    
    def __getitem__(self,index):
        return self.train[index], self.label[index]

    def __len__(self):
        return len(self.train)