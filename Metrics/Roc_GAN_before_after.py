import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import os
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_auc_score

import sys
sys.path.append('..')

path_array = ['/home/alien/XUEYu/paper_code/Parameters/2018_torch/Data_aug_5500',
              '/home/alien/XUEYu/paper_code/Parameters/2018_VAN/VAN_3/1fold_test1_GAN_STE_600',
              '/home/alien/XUEYu/paper_code/Parameters/2018_torch',
              '/home/alien/XUEYu/paper_code/enviroment1/CPC',
              '/home/alien/XUEYu/paper_code/enviroment1/VAN'
             ]

for path in path_array:
    sys.path.append(path)

from STA_CRNN.cpsc2018_torch import build_net, Read_data
from dataset_CPC import data_generate

from model import VAN
from van_params import hyperparams_config
from k_fold_CV import get_k_fold_data as kfold
from dataset_frame_block_one import Read_data as Read_data_van


def load_model(PATH, filename):
    """Load STA-CRNN model"""
    devices = [torch.device(f"cuda:{i}") for i in range(2)]
    conv_arch = [(2,64),(2,128),(3,256),(3,256),(3,256)]
    model = build_net(conv_arch)
    model = model.to(devices[0])
    model = nn.DataParallel(model, device_ids=devices)
    temp = torch.load(os.path.join(PATH, filename))
    model.load_state_dict(temp)
    model.eval()

    return model, devices


def load_van_model(PATH, filename):
    """Load VAN model"""
    devices = [torch.device(f"cuda:{i}") for i in range(2)]
    model_params = hyperparams_config()
    model = VAN(**model_params)
    model = model.to(devices[0])
    model = nn.DataParallel(model, device_ids=devices)
    temp = torch.load(os.path.join(PATH, filename))
    model.load_state_dict(temp)
    model.eval()

    return model, devices


def load_before_after_model(PATHs, filenames):
    """
    PATHs: List. two model params path.
    filenames: List. two filenames.
    """
    assert len(PATHs) == len(filenames)
    models = [] # after GAN, before GAN 

    # STA-CRNN
    for path, filename in zip(PATHs[:2], filenames[:2]):
        model, _ = load_model(path, filename)
        models.append(model)
    
    # VAN
    for path, filename in zip(PATHs[2:], filenames[2:]):
        model, _ = load_van_model(path, filename)
        models.append(model)
    
    return models # len(models) == 4


def return_label(y_label, y_predict):
    '''
    y_label: shape(N, 3)
    y_predict: shape(N, 1)

    Return
    y_final: Modified label
    '''
    y_label, y_predict = y_label.to(device = torch.device('cpu')), y_predict.to(device = torch.device('cpu'))
    y_label, y_predict = y_label.numpy(), y_predict.numpy()

    y_final = []
    for label, predict in zip(y_label, y_predict):
        if predict.astype(label.dtype) in label:
            y_final.append(int(predict))
        else:
            y_final.append(int(label[0]))
    
    y_final = np.array(y_final)
    print(y_final.shape)

    return y_final


def plot_Roc(index, fprs, tprs, roc_aucs):
    '''
    index : 0-8
    '''
    plt.figure()
    lw = 2

    # figure_color, figure_color_res = f'C{index}', f'C{index+1}'
    figure_colors = [f'C{index}' for index in range(len(fprs))]
    labels = ['STA-CRNN with ACGAN', 'STA-CRNN without ACGAN',
            'VAN with ACGAN', 'VAN without ACGAN']
    typename = ['Normal', 'AF', 'I-AVB', 'LBBB', 'RBBB', 'PAC', 'PVC', 'STD', 'STE']
    PATH_fig = '/home/alien/XUEYu/Draft/ROC_GAN_before_after_STACRNN'
    
    if not os.path.exists(PATH_fig):
        os.makedirs(PATH_fig)

    for fpr, tpr, roc_auc, figure_color, label in zip(fprs, tprs, roc_aucs, figure_colors, labels):
        plt.plot(
            fpr[index],
            tpr[index],
            color= figure_color,
            lw=lw,
            label= label +" (area = %0.4f)" % roc_auc[index],
            )

    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC of {typename[index]}")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(PATH_fig, typename[index]+'.png'))
    plt.close()


def data_generate_van(k, i, Mydata):
    '''
    i: [0, k-1]
    labels: List. Labels for different samples.
    samples: List. The number of samples separately.
    '''
    assert i >= 0 and i < k
    # Mydata = Read_data(path) # 读取所有的训练数据和相应的标签
    X_train, y_train, X_test, y_test = kfold(k,i,*Mydata.output())
    y_train, y_test = y_train -1, y_test - 1
    
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    if X_train.dim() <= 3: # For 2D VAN
        X_train = X_train.unsqueeze(2)
        X_test = X_test.unsqueeze(2)

    return X_train, y_train, X_test, y_test # For the VAN model with additional 1 dimension

if __name__ == '__main__':
    
    devices = [torch.device(f"cuda:{i}") for i in range(2)]
    path3 = '/home/alien/XUEYu/paper_code/Parameters/2018_VAN/VAN_3/1fold_test1_GAN_STE_600'
    path4 = '/home/alien/XUEYu/paper_code/Parameters/2018_VAN/VAN_2/VAN_3_L_5500_fold1'

    paths = [path_array[0], path_array[2], path3, path4]
    filenames = ['Label8_Sample150_1fold Test_model_model_weights5.pth',
                 '1fold Test_model_model_weights.pth',
                 '1fold Test_model_model_weights.pth',
                 '1fold Test_model_model_weights.pth'
                ]
    
    model_num = len(filenames)
    models = load_before_after_model(paths, filenames) # 4 model [after_GAN, before_GAN, after_GAN, before_GAN]

    Mydata = Read_data('..') # For STA_CRNN
    _, _,  X_test, y_test = data_generate(4, 0, Mydata)
    X_test, y_test = X_test.to(devices[0]), y_test.to(devices[0])


    Mydata_van = Read_data_van('..', 4) # For VAN
    _, _,  X_test_van, y_test_van = data_generate_van(4, 0, Mydata_van)
    X_test_van, y_test_van = X_test_van.to(devices[0]), y_test_van.to(devices[0])

    soft = nn.Softmax(dim=1) # Instance Softmax
    y_preds = [torch.zeros(y_test.shape[0]) for _ in range(model_num)] # (1719, ) after_GAN, before_GAN
    y_scores = [torch.zeros((y_test.shape[0], 9)) for _ in range(model_num)] # (1719, 9) after_GAN, before_GAN

    with torch.no_grad():
        for i in range(y_test.shape[0]):
            for index, (model, y_score, y_pred) in enumerate(zip(models, y_scores, y_preds)):
                if index < 2:
                    y_score[i, :] = soft(model(X_test[i].unsqueeze(0)))
                    y_pred[i] = torch.argmax(model(X_test[i].unsqueeze(0)), dim= 1)
                else:
                    y_score[i, :] = soft(model(X_test_van[i].unsqueeze(0)))
                    y_pred[i] = torch.argmax(model(X_test_van[i].unsqueeze(0)), dim= 1)

    y_final_matrixs = []
    for index, y_pred in enumerate(y_preds):
        if index < 2:
            y_final = return_label(y_test, y_pred)
            y_final_matrix = label_binarize(y_final, classes=[i for i in range(9)])
            y_final_matrixs.append(y_final_matrix)
        else:
            y_final = return_label(y_test_van, y_pred)
            y_final_matrix = label_binarize(y_final, classes=[i for i in range(9)])
            y_final_matrixs.append(y_final_matrix)

    fprs = [dict() for _ in range(model_num)]
    tprs = [dict() for _ in range(model_num)]
    roc_aucs = [dict() for _ in range(model_num)]

    for i in range(9):
        for fpr, tpr, roc_auc, y_final, y_score in zip(fprs, tprs, roc_aucs, y_final_matrixs, y_scores):
            fpr[i], tpr[i], _ = roc_curve(y_final[:, i], y_score[:, i].numpy())
            roc_auc[i] = auc(fpr[i], tpr[i])
    
    for index in range(9):
        plot_Roc(index, fprs, tprs, roc_aucs)
