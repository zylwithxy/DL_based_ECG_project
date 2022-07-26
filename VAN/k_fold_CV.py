import torch
import numpy as np
from torch import nn
import sys

def get_k_fold_data(k, i, X, y): 
    # 返回第i折交叉验证时所需要的训练和测试数据，分开放，X_train为训练数据，X_test为验证数据
    # i属于 0 -（k-1）, k表示第k折
    assert k > 1
    fold_size = len(X) // k  # 每份的个数:数据总条数/折数（向下取整）
    
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)  # slice(start,end,step)切片函数 得到测试集的索引
        
        X_part, y_part = X[idx], y[idx]			# 只对第一维切片即可
        if j == i: 									# 第i折作test
            X_test, y_test = torch.tensor(np.array(X_part)), torch.tensor(y_part,dtype = torch.int64)
            # X_test, y_test = torch.tensor(np.array(X_part)), torch.tensor(y_part)
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = np.concatenate([X_train, X_part], axis=0) # 其他剩余折进行拼接 也仅第一维
            y_train = np.concatenate([y_train, y_part], axis=0)
    
    X_train = torch.tensor(np.concatenate([X_train, X[k * fold_size:]], axis = 0))
    y_train = torch.tensor(np.concatenate([y_train, y[k * fold_size:]], axis = 0), dtype = torch.int64)
    
    return X_train, y_train, X_test, y_test 

# loss_func = nn.CrossEntropyLoss() 		# 声明Loss函数

class Accumulator:
    """For accumulating sums over `n` variables."""
    def __init__(self, n):
        """Defined in :numref:`sec_softmax_scratch`"""
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)] # 两个数的话，第一个是准确率， 第二个是样本个数


    def reset(self):
        self.data = [0.0] * len(self.data)


    def __getitem__(self, idx):
        return self.data[idx]
    
def confuse_mat(y_hat,y):
    '''
    Calculating the F1 score.
    '''
    confuse_matrix = np.zeros((9, 9), dtype=np.float)
    y_hat, y = y_hat.to(device = torch.device('cpu')), y.to(device = torch.device('cpu'))
    y_hat, y = y_hat.numpy(), y.numpy()

    for predict, true in zip(y_hat, y):
        if predict.astype(true.dtype) in true:
            confuse_matrix[int(predict)][int(predict)] += 1
        else:
            # confuse_matrix[true][predict] += 1
            confuse_matrix[int(true[0])][int(predict)] += 1
    return confuse_matrix

def cal_F1_score(confuse_matrix):

    F11 = 2 * confuse_matrix[0][0] / (np.sum(confuse_matrix[0, :]) + np.sum(confuse_matrix[:, 0]))
    F12 = 2 * confuse_matrix[1][1] / (np.sum(confuse_matrix[1, :]) + np.sum(confuse_matrix[:, 1]))
    F13 = 2 * confuse_matrix[2][2] / (np.sum(confuse_matrix[2, :]) + np.sum(confuse_matrix[:, 2]))
    F14 = 2 * confuse_matrix[3][3] / (np.sum(confuse_matrix[3, :]) + np.sum(confuse_matrix[:, 3]))
    F15 = 2 * confuse_matrix[4][4] / (np.sum(confuse_matrix[4, :]) + np.sum(confuse_matrix[:, 4]))
    F16 = 2 * confuse_matrix[5][5] / (np.sum(confuse_matrix[5, :]) + np.sum(confuse_matrix[:, 5]))
    F17 = 2 * confuse_matrix[6][6] / (np.sum(confuse_matrix[6, :]) + np.sum(confuse_matrix[:, 6]))
    F18 = 2 * confuse_matrix[7][7] / (np.sum(confuse_matrix[7, :]) + np.sum(confuse_matrix[:, 7]))
    F19 = 2 * confuse_matrix[8][8] / (np.sum(confuse_matrix[8, :]) + np.sum(confuse_matrix[:, 8]))

    F1 = (F11+F12+F13+F14+F15+F16+F17+F18+F19) / 9

    return float(F1)
    

def accuracy(y_hat, y):
    """Compute the number of correct predictions.

    Defined in :numref:`sec_softmax_scratch`"""

    reduce_sum = lambda x, *args, **kwargs: x.sum(*args, **kwargs)
    astype = lambda x, *args, **kwargs: x.type(*args, **kwargs)
    argmax = lambda x, *args, **kwargs: x.argmax(*args, **kwargs)

    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = argmax(y_hat, axis=1).reshape(-1,1) # 标记是 0-8, Multilabel
    cmp = astype(y_hat, y.dtype) == y
    cmp = torch.any(cmp, 1) # for multilabels, No need to keep dim

    return float(reduce_sum(astype(cmp, y.dtype))), confuse_mat(y_hat,y) # accuracy and F1 score

def evaluate_accuracy_gpu(net, data_iter, device=None):
    """Compute the accuracy for a model on a dataset using a GPU.

    Defined in :numref:`sec_lenet`"""
    if isinstance(net, nn.Module):
        net.eval()  # Set the model to evaluation mode
        if not device:
            device = next(iter(net.parameters())).device
    # No. of correct predictions, no. of predictions
    metric = Accumulator(2)
    confuse_mat = np.zeros((9,9))
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                # Required for BERT Fine-tuning (to be covered later)
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            y = y.squeeze(1)
            acc, temp_mat = accuracy(net(X), y)
            # metric.add(acc, y.numel())
            metric.add(acc, len(y))
            confuse_mat += temp_mat
    return metric[0] / metric[1], cal_F1_score(confuse_mat), confuse_mat