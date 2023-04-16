import shutil
import torch
import random
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import src.crosr. DHR_Net as models
import numpy as np
import pickle
import os
from PIL import Image
from src.utils.safe_mkdir import safe_mkdir,safe_mkdirs
import logging
logger = logging.getLogger(__name__)
 

def epoch(net,save_path,root,transform_test,dataset_mode):
    logging.info("extracting features on %s data"%dataset_mode)
    net.eval()

    if os.path.exists(os.path.join(save_path,dataset_mode)):
        shutil.rmtree(os.path.join(save_path,dataset_mode))
    safe_mkdirs(os.path.join(save_path,dataset_mode))

    with torch.no_grad():
        for folder in os.listdir(os.path.join(root,dataset_mode)):
            # folder maps to class
            if os.path.exists(os.path.join(save_path,dataset_mode,folder)):
                shutil.rmtree(os.path.join(save_path,dataset_mode,folder))
            safe_mkdirs(os.path.join(save_path,dataset_mode,folder))
            logging.info("folder %s "%folder)
            for file_name in os.listdir(os.path.join(root,dataset_mode,str(folder))):
                if file_name.find(".png")==-1:continue# avoid loading npys
                image = Image.open(os.path.join(root,dataset_mode,str(folder),file_name)).convert("RGB")
                image = transform_test(image)
                image = torch.unsqueeze(image,0)
                image=image.cuda(non_blocking=True)
                logits, _, latent = net(image)

                squeezed_latent = []
                
                squeezed_latent.append(torch.squeeze(logits))
                for layer in latent:
                    m = nn.AdaptiveAvgPool2d((1,1))
                    # Applies a 2D adaptive average pooling over an input signal composed of several input planes.
                    # output_size : the target output size of the image of the form H x W
                    layer_output=m(layer)
                    layer_output_squeezed = torch.squeeze(layer_output)
                    # new layer has shape 1,32 for cifar 10
                    squeezed_latent.append(layer_output_squeezed)
                
                feature = torch.cat(squeezed_latent,0)
                # for cifar10, feature has shape 102
                
                save_name = file_name.split(".")[0]


                # copy val AVs to train, since val needs to be part of MAV compute in
                if dataset_mode=='val':
                    np.save(os.path.join(save_path,'train',str(folder),save_name+".npy"),feature.cpu().data.numpy(),allow_pickle=False)
            
                np.save(os.path.join(save_path,dataset_mode,str(folder),save_name+".npy"),feature.cpu().data.numpy(),allow_pickle=False)
                 


def main(num_classes=6,
         means=[0.4914, 0.4822, 0.4465],
         stds=[0.2023, 0.1994, 0.2010],
         dataset_dir='data/cifa10',
         load_path='data/saved_dhr_model/best_val_acc.pth',
         image_side_length=32,
         save_path='data/saved_features/cifa10',
         channes=3):

    

    seed = 222
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


    print("Num classes "+str(num_classes))


    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(means, stds),
    ])

    root =dataset_dir
    if image_side_length==32 and channes==3:
        net = models.DHRNet_image_32x32_channel_3(num_classes)
    else:
        return 
    checkpoint = torch.load(load_path,map_location="cpu")
    net.load_state_dict(checkpoint["model_state_dict"])
    net.cuda()
    
    folder_names=["train","val","test"]
    for f in folder_names:
        epoch(net,save_path,root,transform_test, f)
   


    
