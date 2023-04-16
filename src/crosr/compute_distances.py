
import pickle
import shutil
import scipy as sp
import sys
import os, glob
import os.path as path
import scipy.spatial.distance as spd
from scipy.io import loadmat, savemat
import json
import torch
import torchvision
import numpy as np
from src.utils.safe_mkdir import safe_mkdirs
import logging
logging.getLogger(__name__)
def compute_channel_distances(mean_vector:torch.Tensor, features:list[torch.Tensor]):

    mean_vector = mean_vector.data.numpy()

    eu, cos, eu_cos = [], [], []
    
    for feat in features:
        feat = feat.data.numpy()
        _eu_dist=spd.euclidean(mean_vector, feat)
        _cos_dist=spd.cosine(mean_vector, feat)
        eu.append(_eu_dist)
        cos.append(_cos_dist)
        eu_cos.append(_eu_dist/200. +_cos_dist)
        
    eu_dist = np.array(eu)
  
    cos_dist = np.array(cos)

    eucos_dist = np.array(eu_cos)

    channel_distances=dict()
    channel_distances['eucos']=eucos_dist
    channel_distances['cosine']= cos_dist
    channel_distances['euclidean']=eu_dist
    return channel_distances
    

def compute_distances(num_classes,class_name,cls_indx,mavfilepath,featurefilepath):
   
    mean_feature_vec = torch.from_numpy(np.load(os.path.join(mavfilepath,class_name+".npy")))
    
    featurefile_list = os.listdir(os.path.join(featurefilepath,class_name))

    correct_features = []
    for featurefile in featurefile_list:
        
        feature = torch.from_numpy(np.load(os.path.join(featurefilepath,class_name,featurefile)))
        logits=feature[:num_classes]# according to src\crosr\get_model_features.py, logits are at the beginning of the feature
        logits=torch.unsqueeze(logits,0)
        _,predicted_category = torch.max(logits,dim=1)
        
        if(predicted_category == cls_indx):
            correct_features.append(feature)


    distance_distribution = compute_channel_distances(mean_feature_vec, correct_features)
    return distance_distribution



def main(feature_dir='data/saved_features/cifar10',
         MAV_path='data/saved_MAVs/cifar10',
         save_path='data/saved_distance_scores/cifar10',
         dataset_dir='data/cifar10'):

    for folder in ['train']:
        
        feature_dir_for_this_folder=os.path.join(feature_dir,folder)
        logging.info("computing distances to mavs for %s"%feature_dir_for_this_folder)
        MAV_path_for_train=os.path.join(MAV_path,'train')
        save_path_for_this_folder=os.path.join(save_path,folder)
        if os.path.exists(save_path_for_this_folder):
            shutil.rmtree(save_path_for_this_folder)
        safe_mkdirs(save_path_for_this_folder)
        
        dataset_dir_for_this_folder=os.path.join(dataset_dir,folder)

        imagefolder = torchvision.datasets.ImageFolder(root=dataset_dir_for_this_folder)
        class_to_idx_dict=imagefolder.class_to_idx
        num_classes=len(os.listdir(MAV_path_for_train))
        for class_name in os.listdir(feature_dir_for_this_folder):
            class_no=class_to_idx_dict[class_name]
            logging.info("Class index %d"%class_no)
            distance_distribution = compute_distances(num_classes,class_name,class_no,MAV_path_for_train,feature_dir_for_this_folder)

            
            pickle_save_path=os.path.join(save_path_for_this_folder,class_name+".pkl")
  
            with open(pickle_save_path,'wb')as pick:
                pickle.dump(distance_distribution,pick)
  
            

