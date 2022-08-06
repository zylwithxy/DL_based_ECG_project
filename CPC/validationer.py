import numpy as np
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("cdc") # Get the same logger from the main

def validation(args, model, device, data_loader, batch_size):
    logger.info("Starting Validation")
    model.eval()
    total_loss = 0
    total_acc  = 0 

    acc_timesteps = np.array([0] * model.timestep, dtype= np.uint32)
    with torch.no_grad():
        for data in data_loader:
            data = data.float().to(device)
            if isinstance(model, nn.DataParallel):
                model_sig = model.module
                hidden = model_sig.init_hidden(len(data)//2, use_gpu=True)
            else:
                hidden = model.init_hidden(len(data), use_gpu=True)

            acc, loss, hidden, accuracies = model(data, hidden)
            total_loss += int(hidden.shape[1]) * loss 
            total_acc  += int(hidden.shape[1]) * acc
            acc_timesteps += np.array([int(hidden.shape[1]) * item for item in accuracies], dtype= np.uint32)

    total_loss /= len(data_loader.dataset) # average loss
    total_acc  /= len(data_loader.dataset) # average acc
    acc_timesteps = acc_timesteps.astype(np.float) / len(data_loader.dataset) # average acc for timesteps
    acc_timesteps = np.around(acc_timesteps, decimals= 4)

    logger.info('===> Validation set: Average loss: {:.4f}\tAccuracy: {:.4f}'.format(
                total_loss, total_acc))
    logger.info(f'{model.timestep} steps accuracies:\n \
                   {acc_timesteps}')

    return total_acc, total_loss
