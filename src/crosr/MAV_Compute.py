
import os, sys
import shutil
import glob
import time
import scipy as sp
from scipy.io import loadmat, savemat
import pickle
import os.path as path
import torch
import numpy as np
import torchvision
from src. utils.safe_mkdir import safe_mkdirs
import logging
logging.getLogger(__name__)

def compute_mean_vector(save_path,featurespath,folder_name,category_index,class_name,num_classes):
    
    featurefile_list = os.listdir(os.path.join(featurespath,folder_name,class_name))
    
    correct_features = []
    features_total=0
    for featurefile in featurefile_list:
        
        feature = torch.from_numpy(np.load(os.path.join(featurespath,folder_name,class_name,featurefile)))
        logits=feature[:num_classes]# according to src\crosr\get_model_features.py, logits are at the end of the feature
        logits=torch.unsqueeze(logits,0)
        _, predicted_category = torch.max(logits,dim=1)
        
        if(predicted_category == category_index):
            correct_features.append(torch.unsqueeze(feature,0))
        features_total+=1
    logging.info("len of features_total : %d"%features_total)
    logging.info("len of correct_features : %d"%len(correct_features))
    correct_features = torch.cat(correct_features,0)

    mav = torch.mean(correct_features,dim=0)

    np.save(os.path.join(save_path,folder_name,class_name+".npy"),mav.data.numpy(),allow_pickle=False)




def main(save_path='data/saved_MAVs/cifar10',
         feature_dir='data/saved_features/cifar10',
         dataset_dir='data/cifar10',
         num_classes=6):
    folder_names=["train"]
    for folder_name in folder_names:
        logging.info("folder_name : %s"%folder_name)
        if os.path.exists(save_path+'/'+folder_name):
            shutil.rmtree(save_path+'/'+folder_name)
        safe_mkdirs(save_path+'/'+folder_name)

        root = dataset_dir

        trainset = torchvision.datasets.ImageFolder(root=os.path.join(root,folder_name), )
        class_to_idx_dict=trainset.class_to_idx
        for class_name in os.listdir(feature_dir+'/'+folder_name):
            class_index=class_to_idx_dict[class_name]
            logging.info("category_index index %d"%class_index)
            compute_mean_vector(save_path=save_path,featurespath= feature_dir,folder_name=folder_name,category_index= class_index,class_name=class_name,num_classes=num_classes)


