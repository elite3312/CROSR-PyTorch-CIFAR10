

import os, sys, pickle, glob
import os.path as path
import argparse
import scipy.spatial.distance as spd
import scipy as sp
from scipy.io import loadmat

from src.crosr.openmax_utils import *
import numpy as np

import libmr


#---------------------------------------------------------------------------------
def weibull_tailfitting(meanfiles_path, distancefiles_path,
                        tailsize = 20, 
                        distance_type = 'eucos'):
                        

    
    weibull_model = {}
 
    for filename in os.listdir(meanfiles_path):
        category = filename.split(".")[0]
        weibull_model[category] = {}
        pickle_load_path=os.path.join(distancefiles_path,category+".pkl")
        distance_scores= pickle.load(open(pickle_load_path,'rb'))
        distance_scores=distance_scores[distance_type]
        meantrain_vec = np.load(os.path.join(meanfiles_path,category+".npy"))

        weibull_model[category]['distances_%s'%distance_type] = distance_scores
        weibull_model[category]['mean_vec'] = meantrain_vec

        distance_scores = list(distance_scores)
        mr= libmr.MR()

        tailtofit = sorted(distance_scores)[-tailsize:]
        '''we use the libMR FitHigh function to do Weibull fitting on the
        largest of the distances between all correct positive training
        instances and the associated µi.'''
        

        mr.fit_high(tailtofit, len(tailtofit))
        weibull_model[category]['weibull_model'] = mr

    

    return weibull_model

def query_weibull(category_name, weibull_model, distance_type = 'eucos'):

    category_weibull_list = []
    category_weibull_list  += [weibull_model[category_name]['mean_vec']]
    category_weibull_list  += [weibull_model[category_name]['distances_%s' %distance_type]]
    category_weibull_list  += [weibull_model[category_name]['weibull_model']]

    return category_weibull_list     

