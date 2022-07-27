import os
import numpy as np
import sys
import random

from torch import nn
import torch.nn.functional as F
import torch

# Add file path
PATH = '/home/alien/XUEYu/paper_code/enviroment1/VAN'
PATH2 = '/home/alien/XUEYu/paper_code/enviroment1'
sys.path.append(PATH)
from dataset_frame_block_one import Read_data, Label_Indices

def weights_init_normal(m):
    classname = m.__class__.__name__ # Return the original class name
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02) # Normal distribution (mean, std**2)
    elif classname.find("BatchNorm1d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def sample_image(samples, generator):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    HIDDEN_DIM= 400
    # Sample noise
    z = torch.tensor(np.random.normal(0, 1, (samples, HIDDEN_DIM)), dtype= torch.float).cuda()
    temp = [num for num in np.arange(9)]
    labels = np.array([temp[i % 9] for i in np.arange(samples)], dtype = np.int64)
    labels = torch.tensor(labels, dtype= torch.long).cuda()

    generator.eval()
    with torch.no_grad():
        gen_imgs = generator(z, labels)
        gen_imgs = gen_imgs.squeeze(1).detach().cpu().numpy()

    return gen_imgs

def generate_STE(samples, generator):
    """
    Generate a lot of STE signals
    samples: The number of STE
    """
    HIDDEN_DIM= 400
    # Sample noise
    z = torch.tensor(np.random.normal(0, 1, (samples, HIDDEN_DIM)), dtype= torch.float).cuda()
    labels = np.array([8 for _ in np.arange(samples)], dtype = np.int64)
    labels = torch.tensor(labels, dtype= torch.long).cuda()

    generator.eval() # Start Testing
    with torch.no_grad():
        gen_imgs = generator(z, labels)
        gen_imgs = gen_imgs.squeeze(1).detach().cpu().numpy()

    return gen_imgs

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def load_ECG_data():
    # Load data
    new_feature = Read_data(PATH2, 4)
    label_first = new_feature.labels[:, 0] # 1D shape [6877,]

    fold_size = label_first.shape[0] // 4 # 4 Fold cross-validation
    idx = slice(fold_size, label_first.shape[0]) # Fold-1 is the test data. Read the remaining 3 parts.
    label_indices = Label_Indices(label_first[idx], new_feature.train)

    return label_indices

if __name__ == '__main__':
    """
    Generate a batch of images
            if generator.__class__.__name__ == 'Generator_3':
                gen_imgs, gen_real, embed_imgs, embed_real = generator(real_imgs ,z, gen_labels) # gen_imgs shape (batch_size, 1, L)
            elif generator.__class__.__name__ == 'Generator_4':
                gen_imgs = generator(z, gen_labels)
    """
    pass