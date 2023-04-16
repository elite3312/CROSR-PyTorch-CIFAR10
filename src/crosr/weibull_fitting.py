import os, sys, pickle, glob
import os.path as path

import scipy.spatial.distance as spd
import scipy as sp
from scipy.io import loadmat

from src.crosr.openmax_utils import *
from src.crosr.evt_fitting import weibull_tailfitting, query_weibull

import numpy as np
import libmr
import torchvision
from sklearn.metrics import roc_auc_score
from src.utils.safe_mkdir import safe_mkdir

import logging
logging.getLogger(__name__)

       


def main(distance_scores_path='data\saved_distance_scores\cifar10',
         MAV_path='data\saved_MAVs\cifar10',
         weibull_tail_size=20,
         save_path='data\saved_weibull_model\cifar10',
         distance_type='eucos'):


    distance_path = os.path.join(distance_scores_path,'train')
    mean_path = os.path.join(MAV_path,'train')
    logging.info("weibull_tailfitting...")
    logging.info("distance_type : %s"%distance_type)
    logging.info("tailsize : %d"%weibull_tail_size)
    weibull_model = weibull_tailfitting(meanfiles_path=mean_path,distancefiles_path= distance_path,
                                        tailsize = weibull_tail_size,distance_type='eucos')
    
    weibull_model_save_folder=os.path.join(save_path,'train')
    if not os.path.exists(weibull_model_save_folder):
        safe_mkdir(weibull_model_save_folder)
    weibull_model_save_pickle_path=os.path.join(weibull_model_save_folder,'saved_weibull_model.pkl')
    logging.info("weibull fitting done, saving it to %s"%weibull_model_save_pickle_path)
    pickle.dump(weibull_model, open(weibull_model_save_pickle_path, 'wb'))
