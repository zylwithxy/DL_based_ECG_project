from functools import partial
from torch import nn

def hyperparams_config():
    
    params = dict(
             in_chans = 12,
             embed_dims=[32, 64, 160, 256], 
             mlp_ratios=[8, 8, 4, 4],
             drop_rate=0.2, 
             drop_path_rate=0.2,
             norm_layer= partial(nn.LayerNorm, eps=1e-6), # Saving parameters
             depths=[3, 3, 5, 2],
    )

    return params  
