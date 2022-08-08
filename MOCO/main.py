import argparse
import random
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn

import torch.optim as optim

from torch.utils import data
import torch.utils.data.distributed

import builder
from config_logging import setup_logs
from tools import ScheduledOptim, save_pre_params, str2bool
from feature_extractor import vgg
from timeit import default_timer as timer
from augmented_dataset import Read_data_augmented, TraindataSet_Aug2
import numpy as np
from trainer import train


def main_worker(conv_arch, k_num, fold_i):
    """
    conv_arch: VGG params.
    k_num: fold number for cross_validation.
    fold_i: [0, k-1].
    """
    torch.cuda.empty_cache()
    PATH = '/home/alien/XUEYu/paper_code/enviroment1'

    parser = argparse.ArgumentParser(description='Moco for ECG')

    parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
 
    parser.add_argument('--seed', default= 42, type=int,
                    help='seed for initializing training.')

    # moco specific configs:
    parser.add_argument('--moco-dim', default=24, type=int,
                    help='feature dimension (default: 24)')
    parser.add_argument('--moco-k', default= 4096, type=int,
                    help='queue size; number of negative keys (default: 5000)')
    parser.add_argument('--moco-m', default=0.999, type=float,
                    help='moco momentum of updating key encoder (default: 0.999)')
    parser.add_argument('--moco-t', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')

    # options for moco v2
    parser.add_argument('--mlp', type= str2bool, default= False,
                    help='use mlp head')
    
    # self
    parser.add_argument('--model-saving-dir', required=True,
                        help='model save directory')
    parser.add_argument('--logging-dir', required=True,
                        help='log save directory')
    parser.add_argument('--n-warmup-steps', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=64, 
                        help='batch size')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--state-archi', type= str2bool, default= True,
                        help='whether needs to show the architecture of the model')
    parser.add_argument('--continue-train', type= str2bool, default= True,
                        help='whether needs to train in spite of early stopping')
    
    args = parser.parse_args()
    use_cuda = torch.cuda.is_available()
    print('use_cuda is', use_cuda)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        # cudnn.deterministic = True
        """
        
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
        """

    model_params_name = f"Moco_v1_{fold_i}_fold_"
    run_name = model_params_name + time.strftime("-%Y-%m-%d_%H_%M_%S")

    global_timer = timer() # global timer
    logger = setup_logs(args.logging_dir, run_name) # setup logs
    
    # create model
    device = torch.device("cuda:0")
    model = builder.MoCo(
       vgg, conv_arch,
        args.moco_dim, args.moco_k, args.moco_m, args.moco_t, args.mlp).to(device)
    
    params = {'num_workers': 0,
              'pin_memory': False}
    
    logger.info('===> loading train, validation and eval dataset\n')

    Mydata = Read_data_augmented(PATH)
    training_set = TraindataSet_Aug2(*Mydata.output())

    train_loader = data.DataLoader(training_set, batch_size=args.batch_size, 
                                   shuffle=True, **params, drop_last= True) # set shuffle to True

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = ScheduledOptim(
        optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), 
            betas=(0.9, 0.98), eps=1e-09, weight_decay=1e-4, amsgrad=True),
        args.n_warmup_steps)
    
    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if args.state_archi:
        logger.info('### Model summary below###\n {}\n'.format(str(model)))
        logger.info('===> Model total parameter: {}\n'.format(model_params))

    # cudnn.benchmark = True
    ## Start training
    best_acc = 0
    best_loss = np.inf
    best_epoch = -1
    count_patience, PATIENCE = 0, 20

    # Data loading code

    for epoch in range(1, args.epochs+ 1):

        epoch_timer = timer()
        top1, _, losses = train(args, model, device, train_loader, optimizer, epoch, criterion)

        if top1.avg > best_acc:
            count_patience = 0 
            best_acc = top1.avg
            model_parameters = model.state_dict()
            best_epoch = epoch + 1
            best_epoch_record = epoch
            val_loss_record = losses.avg
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
            if not args.continue_train:
                logger.info('EarlyStopping counter forces quitting!')
                break
        
        end_epoch_timer = timer()
        logger.info(f'Current accuracy:{top1.avg:.4f} Best accuracy:{best_acc:.4f}')
        logger.info("#### End epoch {}/{}, elapsed time: {}\n".format(epoch, args.epochs, end_epoch_timer - epoch_timer))

    save_pre_params(args.model_saving_dir, model_params_name, {
                'epoch': best_epoch_record,
                'validation_acc': best_acc, 
                'state_dict': model_parameters,
                'validation_loss': val_loss_record,
                'optimizer': optimizer_params,
            })
    
    end_global_timer = timer()
    logger.info("################## Success #########################")
    logger.info("Total elapsed time: %s" 
                % (end_global_timer - global_timer))


if __name__ == '__main__':

    conv_arch = [(2,64),(2,128),(3,256),(3,256),(3,256)]

    main_worker(conv_arch, 4, 1)