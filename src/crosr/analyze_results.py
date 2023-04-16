
import os
from src.crosr.openmax_utils import *
from sklearn.metrics import  accuracy_score

import logging
logging.getLogger(__name__)



def compute_acc(save_path,mode):
    if mode == 'test':
        txt_path=os.path.join(save_path,"test_openset_scores.txt")
    elif mode=='val':
        txt_path=os.path.join(save_path,"val_openset_scores.txt")
    with open(txt_path,'r') as fh:
        lines=fh.readlines()
    ans_list=[]
    pred_list=[]
    number_of_images_predicted_as_unknown=0
    for line in lines:
        line=str.rstrip(line)
        _ans=line.split('\t')[0]
        _pred=line.split('\t')[3]
        if _pred=='unknown':number_of_images_predicted_as_unknown+=1
        if mode=='test':ans_list.append('unknown')
        elif  mode=='val':ans_list.append(_ans)
        pred_list.append(_pred)
    logging.info("Ratio of images classified as unknown in %s: %d / %d = %f"\
                 %(mode,number_of_images_predicted_as_unknown,len(lines),number_of_images_predicted_as_unknown/len(lines)))
    logging.info("acc = %f"%accuracy_score(ans_list,pred_list))


def main(saved_openset_scores='data\saved_openset_scores\cifar10'):


    compute_acc(saved_openset_scores,'val')
    compute_acc(saved_openset_scores,'test')