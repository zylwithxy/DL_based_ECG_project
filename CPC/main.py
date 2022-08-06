import argparse
import time
import os
import logging
from timeit import default_timer as timer
from timm.utils import random_seed

## Libraries
import numpy as np

## Torch
import torch
import torch.nn as nn
from torch.utils import data
import torch.nn.functional as F
import torch.optim as optim

# Self libraries
from tools import ScheduledOptim, save_pre_params, str2bool
from trainer import train
from validationer import validation
from Model import G_enc, Read_data
from config_logging import setup_logs
from dataset_CPC import data_generate, TraindataSet

def main(vgg_params, k_num, fold_i):
    """
    vgg_params: vgg params for g_enc.
    k_num: fold number for cross_validation.
    fold_i: [0, k-1].
    """
    torch.cuda.empty_cache()

    PATH = '/home/alien/XUEYu/paper_code/enviroment1'

    parser = argparse.ArgumentParser(description='CPC for ECG')

    parser.add_argument('--model-saving-dir', required=True,
                        help='model save directory')
    parser.add_argument('--logging-dir', required=True,
                        help='log save directory')
    parser.add_argument('--epochs', type=int, default=60, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--n-warmup-steps', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=64, 
                        help='batch size')
    parser.add_argument('--audio-window', type=int, default=15360, 
                        help='window length to sample from each utterance')
    parser.add_argument('--timestep', type=int, default=12) 
    parser.add_argument('--masked-frames', type=int, default=20)
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--state-archi', type= str2bool, default= True,
                        help='whether needs to show the architecture of the model')
    parser.add_argument('--hidden-GRU', type= int, default= 12,
                        help='hidden dim of GRU')
    parser.add_argument('--state-parallel', type= str2bool, default= True,
                        help='whether needs to use 2 GPUs')
    parser.add_argument('--gru-layers', type= int, default= 1, required=True,
                        help='the number of gru layers')

    args = parser.parse_args()
    use_cuda = torch.cuda.is_available()
    print('use_cuda is', use_cuda)

    # Params
    model_params_name = f"cdc_{fold_i}_fold_" + f"GRU_{args.hidden_GRU}_"
    model_params_name = model_params_name + '2gpu_' if args.state_parallel else model_params_name
    model_params_name = f"Timestep_{args.timestep}_" + model_params_name \
                        if args.timestep != 12 else model_params_name
    model_params_name = f"layers_gru_{args.gru_layers}_" + model_params_name \
                        if args.gru_layers != 1 else model_params_name

    run_name = model_params_name + time.strftime("-%Y-%m-%d_%H_%M_%S")

    global_timer = timer() # global timer
    logger = setup_logs(args.logging_dir, run_name) # setup logs
    device = torch.device("cuda" if use_cuda else "cpu")
    model = G_enc(args.timestep, args.batch_size, 
                  args.audio_window, vgg_params, args.hidden_GRU, args.gru_layers).to(device)
    
    if args.state_parallel:
        print('Using 2 GPUs !')
        devices = [torch.device(f"cuda:{i}") for i in range(2)]
        model = nn.DataParallel(model, device_ids=devices)
        args.batch_size *= 2 
    
    ## Loading the dataset
    params = {'num_workers': 0,
              'pin_memory': False} if use_cuda else {}

    logger.info('===> loading train, validation and eval dataset\n')
    Mydata = Read_data(PATH)
    X_train, y_train, X_test, y_test = data_generate(k_num, fold_i, Mydata)

    training_set, validation_set = TraindataSet(X_train),\
                                   TraindataSet(X_test)
    
    train_loader = data.DataLoader(training_set, batch_size=args.batch_size, shuffle=True, **params) # set shuffle to True
    validation_loader = data.DataLoader(validation_set, batch_size=args.batch_size, shuffle=False, **params) # set shuffle to False
    # nanxin optimizer  
    optimizer = ScheduledOptim(
        optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), 
            betas=(0.9, 0.98), eps=1e-09, weight_decay=1e-4, amsgrad=True),
        args.n_warmup_steps)

    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if args.state_archi:
        logger.info('### Model summary below###\n {}\n'.format(str(model)))
        logger.info('===> Model total parameter: {}\n'.format(model_params))
        
    ## Start training
    best_acc = 0
    best_loss = np.inf
    best_epoch = -1
    count_patience, PATIENCE = 0, 20

    for epoch in range(1, args.epochs + 1):
        epoch_timer = timer()
        train(args, model, device, train_loader, optimizer, epoch, args.batch_size)
        val_acc, val_loss = validation(args, model, device, validation_loader, args.batch_size)
        
        # Save
        if val_acc > best_acc:
            count_patience = 0 
            best_acc = val_acc
            model_parameters = model.state_dict()
            best_epoch = epoch + 1
            best_epoch_record = epoch
            val_loss_record = val_loss
            optimizer_params = optimizer.state_dict()

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
                'validation_acc': best_acc, 
                'state_dict': model_parameters,
                'validation_loss': val_loss_record,
                'optimizer': optimizer_params,
            })
    
    end_global_timer = timer()
    logger.info("################## Success #########################")
    logger.info("Total elapsed time: %s and GRU hidden number: %d" 
                % (end_global_timer - global_timer, model.hidden_gru))


if __name__ == '__main__':

    conv_arch = [(2,64),(2,128),(3,256),(3,256),(3,256)]
    random_seed()
    torch.cuda.manual_seed_all(42)

    main(conv_arch, 4, 1)