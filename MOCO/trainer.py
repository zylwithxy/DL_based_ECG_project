import torch.nn as nn
import torch
import logging
from tools import accuracy, AverageMeter

logger = logging.getLogger("Moco")

def train(args, model, device, train_loader, optimizer, epoch, criterion):

    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    model.train()
    for batch_idx, (data_aug1, data_aug2) in enumerate(train_loader):
        
        data_aug1, data_aug2 = data_aug1.to(device), \
                               data_aug2.to(device) # data shape: (B, 12, 15360)

        output, target = model(im_q= data_aug1, im_k= data_aug2)
        loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        losses.update(loss.item(), data_aug1.size(0))
        top1.update(acc1[0]/100, data_aug1.size(0))
        top5.update(acc5[0]/100, data_aug1.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step() # optimizer is in a class which contains the optimizer.

        lr = optimizer.update_learning_rate()
        if batch_idx % args.log_interval == 0:
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tlr:{:.6f}\tAccuracy1: {:.4f}%\tAccuracy5: {:.4f}%\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data_aug1), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), lr, acc1[0], acc5[0], loss.item()))
        
    return top1, top5, losses


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
        f_extracted = feature_extractor.encoder_q(X)
        # hidden = feature_extractor.init_hidden(len(y))
        # f_extracted, hidden_last = feature_extractor.predict(X, hidden)
        # pred = classifier(f_extracted, hidden_last) # (B, 9)
        pred = classifier(f_extracted)
        l = loss(pred, y) # (B, 9), (B)
        l.backward()
        optimizer.step()
        lr = optimizer.update_learning_rate()

        pred_result = pred.max(dim= 1)[1] # (values, indices)
        acc = torch.sum(pred_result.type(y.dtype) == y).item()
        acc = acc / len(y) 

        if batch_idx % args.log_interval == 0:
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tlr:{:.6f}\tAccuracy: {:.4f}\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(X), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), lr, acc, l.item()))