from dataset import TraindataSet
from torch.utils.data import DataLoader
import torch
from torch import nn
import os
from k_fold_CV import get_k_fold_data as kfold
from k_fold_CV import evaluate_accuracy_gpu
import numpy as np
from pytorch_tools import EarlyStopping
import sys

# Load GAN to generate new training samples
PATH = '/home/alien/XUEYu/paper_code/enviroment1/ECG_GAN'
sys.path.append(PATH)
from load_generator import LoadData

def load_synthetized_data(LABEL, SAMPLE_NUM):
    """
    LABEL: int [0-8]
    SAMPLE_NUM: Number of samples generated
    """

    CLASSES, HIDDEN_DIM, L = 9, 400, 5500
    PATH = '/home/alien/XUEYu/paper_code/Parameters/2018_Generator'
    gen_leads = LoadData(CLASSES, HIDDEN_DIM, L, PATH, SAMPLE_NUM, LABEL)

    return (gen_leads.gen_results, gen_leads.gen_labels)

def try_gpu(i=0):  #@save
    """如果存在，则返回gpu(i)，否则返回cpu()"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def data_generate(k, i, Mydata, labels, samples):
    '''
    i: [0, k-1]
    labels: List. Labels for different samples.
    samples: List. The number of samples separately.
    '''
    assert i >= 0 and i < k
    # Mydata = Read_data(path) # 读取所有的训练数据和相应的标签
    X_train, y_train, X_test, y_test = kfold(k,i,*Mydata.output())
    y_train, y_test = y_train -1, y_test - 1

    for label, sample in zip(labels, samples):
        X_generated, y_generated = load_synthetized_data(label, sample) # Load GAN samples and labels
        X_train = torch.cat((X_train, X_generated), dim= 0)
        y_train = torch.cat((y_train, y_generated), dim= 0)
    
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    if X_train.dim() <= 3: # For 2D VAN
        X_train = X_train.unsqueeze(2)
        X_test = X_test.unsqueeze(2)

    return X_train, y_train, X_test, y_test # For the VAN model with additional 1 dimension

def save_model(*args):

    assert len(args) == 5
    net_kfold_params, confuse_mat, max_F1, path2, k = args

    if os.path.exists(path2):
        files_names = os.listdir(path2)
        files_names.sort()
        if len( os.listdir( os.path.join(path2, files_names[-1]) ) ) >= 8:
            create_file_name = files_names[0][:-1] + str(len(files_names) + 1) # str
            os.makedirs(os.path.join(path2, create_file_name), exist_ok= True)
        else:
            create_file_name = files_names[-1]

        path2 = os.path.join(path2, create_file_name)
        model_weights_location = os.path.join(path2, f'{k+1}fold Test_model_'+'model_weights.pth') # Model weights location
        confuse_mat_location = os.path.join(path2, f'{k+1}fold Test_model_' + "confuse_mat.npy") # Confuse matrix location

        torch.save(net_kfold_params, model_weights_location)
        np.save(confuse_mat_location, confuse_mat)
    
    print(f"The best F1 score is : {max_F1:.4f}")

def train_experiment(net, *,  num_gpus, batch_size, 
                    lr, epoches, Mydata, k_num, fold_i, path2, labels, samples):
    
    X_train, y_train, X_test, y_test = data_generate(k_num, fold_i, Mydata, labels, samples) # Data generating

    Train_data = TraindataSet(X_train, y_train)
    Test_data = TraindataSet(X_test, y_test)
    train_iter, test_iter = DataLoader(Train_data, batch_size = batch_size, shuffle = True), \
                            DataLoader(Test_data, batch_size = batch_size, shuffle = True)
    print("Finishing dataloading")

    devices = [try_gpu(i) for i in range(num_gpus)]
    net = nn.DataParallel(net, device_ids=devices)
    optimizer = torch.optim.Adam(net.parameters(), lr = lr)
    loss = nn.CrossEntropyLoss()
    
    early_stopping = EarlyStopping(patience= 21)# Early Stopping

    for epoch in range(epoches):
        print(f"Fold{fold_i + 1} Epoch {epoch+1}\n-------------------------------")
        net.train()
        for batch, (X, y) in enumerate(train_iter):
            optimizer.zero_grad()
            X, y = X.to(devices[0]), y[:,0].to(devices[0])
            # y = y.squeeze(1) # Multidata doesn't need squeeze
            # y = y.type(torch.int64)
            # print(type(y), y.dtype)
            l = loss(net(X), y)
            l.backward()
            optimizer.step()
            if batch % 10 == 0:
                losserror, current = l.item(), min(batch_size * (batch+1), y_train.shape[0])
                print(f"loss: {losserror:>7f}  [{current:>5d}/{y_train.shape[0]:>5d}]")
        
        # Test_accu need to be superseded by average F1 score.
        test_accu, F1_score, confuse_mat= evaluate_accuracy_gpu(net, test_iter) 
        print(f"Test Error: \n Accuracy: {(100*test_accu):>0.1f}% F1 score:{100 * F1_score:>4f}% \n")
        
        early_stopping(F1_score, net.state_dict(), confuse_mat)
        if early_stopping.early_stop:
            print("Early Stopping stops it !")
            break
        
    print("Finishing training, then Saving models")
    save_model(early_stopping.net_parameters, 
               early_stopping.confuse_mat_return,
               early_stopping.best_score, 
               path2,
               fold_i)
