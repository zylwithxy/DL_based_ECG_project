import torch
import numpy as np
from torch import nn
import logging

logger = logging.getLogger("cdc") # Get the same logger from the main

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

    return float(reduce_sum(astype(cmp, y.dtype))), confuse_mat(y_hat,y) # accuracy and F1score


def validation(feature_extractor, classifier, data_iter, device):
    """
    net: model.
    data_iter: 
    """
    loss = nn.CrossEntropyLoss()
    feature_extractor.eval()
    classifier.eval()
   
    metric = Accumulator(2)
    confuse_mat = np.zeros((9,9))

    with torch.no_grad():
        for X, y in data_iter:      
            X, y = X.to(device), y.to(device)
            f_extracted = feature_extractor.encoder(X)
            # hidden = feature_extractor.init_hidden(len(y))
            # f_extracted, hidden_last = feature_extractor.predict(X, hidden)
            # pred = classifier(f_extracted, hidden_last) # (B, 9)
            pred = classifier(f_extracted) # (B, 9)
            l = loss(pred, y[:, 0])
            acc, temp_mat = accuracy(pred, y)
            metric.add(acc, len(y))
            confuse_mat += temp_mat

    test_accu, test_F1_score = metric[0] / metric[1], cal_F1_score(confuse_mat)
    logger.info(
        '===> Validation set: Average loss: {:.4f}\tAccuracy: {:.4f}\tF1 score:{:.4f}'.format( \
                l.item(), test_accu, test_F1_score) )

    return test_accu, test_F1_score, l.item(), confuse_mat