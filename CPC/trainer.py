import logging
import torch
import torch.nn as nn

logger = logging.getLogger("cdc") # Get the same logger from the main

def train(args, model, device, train_loader, optimizer, epoch, batch_size):
    model.train()
    for batch_idx, data in enumerate(train_loader):
        
        # data = data.float().unsqueeze(1).to(device) # add channel dimension
        data = data.float().to(device) # data shape: (B, 12, 15360)
        optimizer.zero_grad()

        if isinstance(model, nn.DataParallel):
            model_sig = model.module
            hidden = model_sig.init_hidden(len(data)//2, use_gpu=True)
        else:
            hidden = model.init_hidden(len(data), use_gpu=True)
        
        acc, loss, hidden, accuracies = model(data, hidden)

        loss.backward()
        optimizer.step() # optimizer is in a class which contains the optimizer.
        lr = optimizer.update_learning_rate()
        if batch_idx % args.log_interval == 0:
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tlr:{:.6f}\tAccuracy: {:.4f}\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), lr, acc, loss.item()))
            """
            logger.info(f'{model.timestep} steps accuracies:\n \
                        {accuracies}')
            """


def train_ft(args, feature_extractor, classifier, device, train_loader, optimizer, epoch): 
    """
    Train for fine tuning
    feature_extractor: Model in pre-training.
    classifier: Classifier in fine-tuning.
    """
    
    loss = nn.CrossEntropyLoss()
    feature_extractor.eval()
    classifier.train()

    for batch_idx, (X, y) in enumerate(train_loader):

        optimizer.zero_grad()
        X, y = X.to(device), y[:,0].to(device)
        f_extracted = feature_extractor.encoder(X)
        # hidden = feature_extractor.init_hidden(len(y))
        # f_extracted, hidden_last = feature_extractor.predict(X, hidden)
        # pred = classifier(f_extracted, hidden_last) # (B, 9)
        pred = classifier(f_extracted)
        l = loss(pred, y) # (B, 9), (B)
        l.backward()
        optimizer.step()
        lr = optimizer.update_learning_rate()

        pred_result = pred.max(dim= 1)[1]
        acc = torch.sum(pred_result.type(y.dtype) == y).item()
        acc = acc / len(y) 

        if batch_idx % args.log_interval == 0:
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tlr:{:.6f}\tAccuracy: {:.4f}\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(X), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), lr, acc, l.item()))