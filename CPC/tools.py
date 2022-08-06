import numpy as np
import os
import torch
import logging

class ScheduledOptim(object):
    """A simple wrapper class for learning rate scheduling"""

    def __init__(self, optimizer, n_warmup_steps):
        self.optimizer = optimizer
        self.d_model = 128 
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0 
        self.delta = 1

    def state_dict(self):

        return self.optimizer.state_dict()

    def step(self):
        """Step by the inner optimizer"""
        self.optimizer.step()

    def zero_grad(self):
        """Zero out the gradients by the inner optimizer"""
        self.optimizer.zero_grad()

    def increase_delta(self):
        self.delta *= 2

    def update_learning_rate(self):
        """Learning rate scheduling per step"""

        self.n_current_steps += self.delta
        new_lr = np.power(self.d_model, -0.5) * np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        return new_lr


def save_pre_params(dir_name, file_name, net_params):
    """
    dir_name: The location of saved parameters.
    file_name: The file name of saved parameters.
    net_params: The params of the model.
    """
    logger = logging.getLogger('cdc')

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    
    index = len(os.listdir(dir_name))
    file_name = file_name + f'{index+1}.pth'
    concat_path = os.path.join(dir_name, file_name) # Final path
    torch.save(net_params, concat_path)
    logger.info(f"The model parameters are saved in {concat_path}\n")


def str2bool(param):
    """
    param: str. The param needed to transform
    """
    if param.lower() == 'true':
        return True
    else:
        return False