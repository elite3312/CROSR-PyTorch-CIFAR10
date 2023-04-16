import torch
import random
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import src.crosr.DHR_Net as models
import numpy as np
import pickle
import os
from src.utils.safe_mkdir import safe_mkdirs
import logging
logger = logging.getLogger(__name__)
from tabulate import tabulate


def epoch_train(epoch_no,net,trainloader,optimizer):
        
    net.train() 
    correct=0
    total=0
    total_loss = 0.0
    total_cls_loss = 0.0
    total_reconst_loss = 0.0
    iter=0
    cls_criterion = nn.CrossEntropyLoss()
    reconst_criterion = nn.MSELoss()

    for i,data in enumerate(trainloader):

        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
    
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        logits, reconstruct,_ = net(inputs)

        cls_loss = cls_criterion(logits, labels)

        reconst_loss = reconst_criterion(reconstruct,inputs)
      
        if(torch.isnan(cls_loss) or torch.isnan(reconst_loss)):
            print("Nan at iteration ",iter)
            cls_loss=0.0
            reconst_loss=0.0
            logits=0.0          
            reconstruct = 0.0  
            continue

        loss = cls_loss + reconst_loss

        loss.backward()
        optimizer.step()  

        total_loss = total_loss + loss.item()
        total_cls_loss = total_cls_loss + cls_loss.item()
        total_reconst_loss = total_reconst_loss + reconst_loss.item()

        _, predicted = torch.max(logits.data, 1)# the .data is simply used to access the logits tensor
        # _ is the maximum value in the logits for images in this batch, predicted is the predicted classes for images in this batch
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        iter = iter + 1
    # acc, cross_entropy loss, reconst_loss,total_loss
    return [(100 * (correct / total)), (total_cls_loss/iter), (total_reconst_loss/iter), (total_loss/iter)]
    
def epoch_val(net,testloader):

    net.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    total_cls_loss = 0.0
    total_reconst_loss = 0.0
    iter=0
    cls_criterion = nn.CrossEntropyLoss()
    reconst_criterion = nn.MSELoss()

    with torch.no_grad():
        for data in testloader:

            images, labels = data
            images=images.cuda(non_blocking=True)
            labels=labels.cuda(non_blocking=True)

            logits, reconstruct,_ = net(images)

            cls_loss = cls_criterion(logits, labels)

            reconst_loss = reconst_criterion(reconstruct,images)
        
            loss = cls_loss + reconst_loss

            total_loss = total_loss + loss.item()
            total_cls_loss = total_cls_loss + cls_loss.item()
            total_reconst_loss = total_reconst_loss + reconst_loss.item()

            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            iter = iter + 1

    return [(100 * (correct / total)), (total_cls_loss/iter), (total_reconst_loss/iter), (total_loss/iter)]
                 


def main(lr=0.0001,
         epochs=30,
         batch_size=10,
         momentum=.9,
         weight_decay=0.0005,
         means=[0.4914, 0.4822, 0.4465],
         stds=[0.2023, 0.1994, 0.2010],
         num_classes=6,
         dataset_dir='data/cifar10',
         image_side_length=32,
         save_path='data/saved_dhr_model/cifar10',
         load_and_continue=False,
         channes=3):

    seed = 222
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    
    logging.info("training a dhr_network with %d classes :"%num_classes)

    transform_train = transforms.Compose([
        transforms.ColorJitter(brightness=0.5, hue=0.3),
        transforms.RandomAffine(degrees=30,translate =(0.2,0.2),scale=(0.75,1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(means,stds),
    ])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(means,stds),
    ])

    root = dataset_dir

    trainset = torchvision.datasets.ImageFolder(root=os.path.join(root,"train"), 
                                        transform=transform_train)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True,pin_memory=True,drop_last=True)

    valset = torchvision.datasets.ImageFolder(root=os.path.join(root,"val"), 
                                            transform=transform_val)

    valloader = torch.utils.data.DataLoader(valset , batch_size=batch_size,
                                            shuffle=False,pin_memory=True,drop_last=True)
    if image_side_length==32 and channes==3:
        net = models.DHRNet_image_32x32_channel_3(num_classes)
    else:#todo
        return -1
    
    # load weights
    best_val_acc=0
    load_path=os.path.join(save_path,"best_val_acc.pth")
    if load_and_continue:
        # try and load weights
        
        if os.path.exists(load_path)==False:
            raise Exception("load_and_continue is true, but os.path.join(save_path,\"best_val_acc.pth\") does not exist,\n\
                             this may be the first time you run this script, thus no saved dhr model is present.\n \
                             set load_and_continue to False and try again.")
        checkpoint = torch.load(load_path,map_location="cpu")
        logging.info("loaded weights from os.path.join(save_path,\"best_val_acc.pth\")")
        net.load_state_dict(checkpoint["model_state_dict"])
        best_val_acc=checkpoint["val_acc"]
        logging.info("setting best_val_acc to %f"%best_val_acc)
    else:
        if os.path.exists(load_path):
            raise Exception("load_and_continue is False, but os.path.join(save_path,\"best_val_acc.pth\") does exist,\n\
                             this may not be the first time you run this script, thus a saved dhr model is present.\n \
                             to avoid overwriting the existing best_val_acc.pth, the program will exit now.\n\
                            If you still want to train from scratch, rename best_val_acc or move it away.")
    net = torch.nn.DataParallel(net.cuda())


    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum,weight_decay=weight_decay)
    # using tabulate to log prettier
    _log_headers=["optimizer parameters"]
    _log_data=[]
    _log_data.append(["optimizer","SGD"])
    _log_data.append(["lr","%f"%lr])
    _log_data.append(["momentum","%f"%momentum])
    _log_data.append(["weight_decay","%f"%weight_decay])
    logging.info('\n'+tabulate(_log_data,_log_headers))

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    # using tabulate to log prettier
    _log_headers=["scheduler parameters"]
    _log_data=[]
    _log_data.append(["scheduler","StepLR"])
    _log_data.append(["step_size"," %d"%30])
    _log_data.append(["gamma","%f"%0.5])
    logging.info('\n'+tabulate(_log_data,_log_headers))


    
    for epoch in range(epochs):  # loop over the dataset multiple times
    
        train_metrics = epoch_train(epoch,net,trainloader,optimizer)
        val_metrics = epoch_val(net,valloader )
        
        scheduler.step()
        
        
        # using tabulate to log prettier
       
        _log_headers=["epoch %d"%epoch,"acc","cross_entropy loss","reconstion_loss","total_loss"]
        
        _log_data=[]
        _log_data.append(["train",str(train_metrics[0]),str(train_metrics[1]),str(train_metrics[2]),str(train_metrics[3])])
        _log_data.append(["val",str(val_metrics[0]),str(val_metrics[1]),str(val_metrics[2]),str(val_metrics[3])])
        logging.info('\n'+tabulate(_log_data,_log_headers))

        if val_metrics[0]>best_val_acc:
            best_val_acc=val_metrics[0]
            if not os.path.exists(save_path):
                safe_mkdirs(save_path)
            torch.save({'epoch':epoch,
                        'model_state_dict':net.module.state_dict(),
                        'train_acc':train_metrics[0],
                        'train_loss':train_metrics[3],
                        'val_acc':val_metrics[0] ,
                        'val_loss':val_metrics[3]},
                        os.path.join(save_path,"best_val_acc.pth"))


    
