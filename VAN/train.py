import torch
from model import VAN
from van_params import hyperparams_config
# from trainer import train_experiment
from trainer_frame_block import train_experiment
# from dataset_frame_block import Read_data
from dataset_frame_block_one import Read_data
from timm.utils import random_seed

def train_hyperparams():
    """
    path : Data File location
    path2: Model params location
    """
    params = dict(
                  num_gpus = 2,
                  batch_size = 32,
                  lr = 1e-3,
                  epoches = 70, 
                  Mydata = None, 
                  k_num = 4, 
                  fold_i = 0, 
                  path2 = '/home/alien/XUEYu/paper_code/Parameters/2018_VAN',
                  labels= [8],
                  samples= [900]
                 )

    return params

if __name__ == "__main__":

    # Const
    PATH = '/home/alien/XUEYu/paper_code/enviroment1'

    # Set seeds
    random_seed()
    torch.cuda.manual_seed_all(42)

    # 1. Loading data
    Mydata = Read_data(PATH, 4)

    # 2. Set up 4 models
    model_params = hyperparams_config()
    nets = [VAN(**model_params).cuda() for _ in range(1)]

    # 3. Train models
    train_params  = train_hyperparams()
    train_params['Mydata'] = Mydata
    # train_params["fold_i"] = 1

    for k in range(4):
        train_experiment(nets[k], **train_params)
        train_params["fold_i"] += 1
        break # 