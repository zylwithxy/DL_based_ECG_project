import argparse
import time
import os
from timeit import default_timer as timer
from timm.utils import random_seed
import numpy as np

import torch
import torch.nn as nn
from torch.utils import data
import torch.nn.functional as F
import torch.optim as optim

# Self libraries
from tools import ScheduledOptim, save_pre_params, str2bool
import builder
from trainer import train_ft
from feature_extractor import vgg, Read_data, data_generate, TraindataSet_total
from validationer_ft import validation
from classifier import classifier_1
from config_logging import setup_logs


def main(conv_arch, k_num, fold_i):
    """
    conv_arch: vgg params for g_enc.
    k_num: fold number for cross_validation.
    fold_i: [0, k-1].
    """
    torch.cuda.empty_cache()

    PATH = '/home/alien/XUEYu/paper_code/enviroment1'

    parser = argparse.ArgumentParser(description='Fine_tuning Moco for ECG')

    parser.add_argument('--model-saving-dir', required= True,
                        default= '/home/alien/XUEYu/paper_code/Parameters/2018_Moco',
                        help='model save directory')
    parser.add_argument('--logging-dir', required= True, 
                        default= '/home/alien/XUEYu/paper_code/Parameters/Log_Moco',
                        help='log save directory')
    parser.add_argument('--epochs', type=int, default=120, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--n-warmup-steps', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=64, 
                        help='batch size')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--state-archi', type= str2bool, default= True,
                        help='whether needs to show the architecture of the model')

    parser.add_argument('--index', type= int, default= 1, required= True,
                        help='The index of saved model')
    parser.add_argument('--load-pretraining-params', type= str2bool, default=True, required= True,
                        help='whether needs to load the pre-training params')
    # parser.add_argument('--classifier', default= 'class_1', required= True,
                        # help='which classifier we used')
    
    # moco specific configs:
    parser.add_argument('--moco-dim', default=24, type=int,
                    help='feature dimension (default: 24)')
    parser.add_argument('--moco-k', default= 4096, type=int,
                    help='queue size; number of negative keys (default: 5000)')
    parser.add_argument('--moco-m', default=0.999, type=float,
                    help='moco momentum of updating key encoder (default: 0.999)')
    parser.add_argument('--moco-t', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')
    parser.add_argument('--mlp', type= str2bool, default= False,
                    help='use mlp head')
            

    args = parser.parse_args()
    use_cuda = torch.cuda.is_available()
    print('use_cuda is', use_cuda)

    # Loading model params
    temp = f"Moco_v1_{fold_i}_fold_" + f'{args.index}.pth'
    model_params_loading = temp

    temp_model_name = f"Fine_tuning_Moco_v1_{fold_i}_fold_" + f'{args.index}_'
    model_params_name = temp_model_name

    temp_run_name = temp_model_name \
                + time.strftime("-%Y-%m-%d_%H_%M_%S")
    run_name = temp_run_name

    global_timer = timer() # global timer
    
    device = torch.device("cuda" if use_cuda else "cpu")

    model = builder.MoCo(
       vgg, conv_arch,
        args.moco_dim, args.moco_k, args.moco_m, args.moco_t, args.mlp).to(device)

    checkpoint = torch.load(os.path.join(args.model_saving_dir \
                            , model_params_loading))

    if args.load_pretraining_params: # Whether needs to load the pre-training params
        print('Loading pre-training params of model')
        model.load_state_dict(checkpoint['state_dict'])
    else:
        print('Random initialization configuration')
        run_name = 'Random_' + run_name
        model_params_name = 'Random_' + model_params_name
    
    for param in model.parameters():
        param.requires_grad = False
    
    classifier = classifier_1()
    classifier = classifier.to(device)

    logger = setup_logs(args.logging_dir, run_name)

    ## Loading the dataset
    params = {'num_workers': 0,
              'pin_memory': False} if use_cuda else {}
    
    logger.info('===> loading train and validation datasets with labels\n')

    Mydata = Read_data(PATH)
    X_train, y_train, X_test, y_test = data_generate(k_num, fold_i, Mydata)

    training_set, validation_set = TraindataSet_total(X_train, y_train),\
                                   TraindataSet_total(X_test, y_test)
    train_loader = data.DataLoader(training_set, batch_size=args.batch_size, shuffle=True, **params) # set shuffle to True
    validation_loader = data.DataLoader(validation_set, batch_size=args.batch_size, shuffle=False, **params) # set shuffle to False

    optimizer = ScheduledOptim(
        optim.Adam(
            filter(lambda p: p.requires_grad, classifier.parameters()),
            betas=(0.9, 0.98), eps=1e-09, weight_decay=1e-4, amsgrad=True),
        args.n_warmup_steps)

    classifier_params = sum(p.numel() for p in classifier.parameters() if p.requires_grad)

    if args.state_archi:
        logger.info('### Model summary below###\n {}\n'.format(str(classifier)))
        logger.info('===> Model total parameter: {}\n'.format(classifier_params))
    
    ## Start training
    best_F1 = 0
    best_loss = np.inf
    best_epoch = -1
    count_patience, PATIENCE = 0, 20

    for epoch in range(1, args.epochs + 1):
        epoch_timer = timer()
        train_ft(args, model, classifier, device, train_loader, optimizer, epoch)
        test_accu, test_F1_score, test_loss, confuse_mat = validation(model, classifier
                                                          ,validation_loader, device)
        # Save
        if test_F1_score > best_F1:
            count_patience = 0 
            best_F1 = test_F1_score 
            classifier_parameters = classifier.state_dict()
            best_epoch = epoch + 1
            best_epoch_record = epoch
            val_loss_record = test_loss
            optimizer_params = optimizer.state_dict()
            confuse_mat_record = confuse_mat

        elif epoch - best_epoch > 2:
            optimizer.increase_delta()
            best_epoch = epoch + 1
            count_patience += 1
            logger.info(f'Ec: {count_patience} out of {PATIENCE}') # EarlyStopping counter(Ec)
        else:
            count_patience += 1
            logger.info(f'Ec: {count_patience} out of {PATIENCE}')
        
        if count_patience >= PATIENCE:
            logger.info('EarlyStopping counter forces quitting!')
            break
        
        end_epoch_timer = timer()
        logger.info("#### End epoch {}/{}, elapsed time: {}\n".format(epoch, args.epochs, end_epoch_timer - epoch_timer))
    
    ## end
    save_pre_params(args.model_saving_dir, model_params_name, {
                'epoch': best_epoch_record,
                'best_F1': best_F1, 
                'state_dict': classifier_parameters,
                'validation_loss': val_loss_record,
                'optimizer': optimizer_params,
            })

    # Saving .npy matrix
    PATH3 = '/home/alien/XUEYu/paper_code/Parameters/2018_Moco_confusion_mat'
    if not os.path.exists(PATH3):
        os.makedirs(PATH3)
    count_mat = len(os.listdir(PATH3))
    mat_name = model_params_name + f'{count_mat+1}.npy'
    np.save(os.path.join(PATH3, mat_name), confuse_mat_record)
    
    end_global_timer = timer()
    logger.info("################## Success #########################")
    logger.info("Total elapsed time: %s and index is %d" 
                % (end_global_timer - global_timer, args.index) )

if __name__ == '__main__':

    conv_arch = [(2,64),(2,128),(3,256),(3,256),(3,256)]
    random_seed()
    torch.cuda.manual_seed_all(42)

    main(conv_arch, 4, 1)