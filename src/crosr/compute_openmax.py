import os, pickle




from src.crosr.openmax_utils import *
from src.crosr.evt_fitting import query_weibull
import numpy as np
import libmr
import torchvision
import logging
logging.getLogger(__name__)

def recalibrate_scores(weibull_model:libmr.MR, img_features:np.ndarray,
                        alpharank:int, distance_type :str,num_classes:int,class_to_idx_dict:dict):

    logits=img_features[:num_classes]
    ranked_list = logits.argsort().ravel()[::-1]
    
    alpha_weights = [((alpharank+1) - i)/float(alpharank) for i in range(1, alpharank+1)]
    ranked_alpha = np.zeros(num_classes)
    for i in range(len(alpha_weights)):
        ranked_alpha[ranked_list[i]] = alpha_weights[i]
    
    openmax_layer = []
    openmax_unknown = []
    
    for cls_indx in range(num_classes):

        # find class_name
        class_name=None
        for k in class_to_idx_dict.keys():
            if class_to_idx_dict[k]==cls_indx:
                class_name=k
                break
        category_weibull_list  = query_weibull(class_name, weibull_model, distance_type)
        category_mean_vec=category_weibull_list [0]
        category_weibull_model=category_weibull_list[2]

        distance = compute_distance(img_features, category_mean_vec,
                                            distance_type = distance_type)

        wscore = category_weibull_model.w_score(distance)
        modified_unit = logits[cls_indx] * ( 1 - wscore*ranked_alpha[cls_indx] )
        openmax_layer += [modified_unit]
        openmax_unknown += [logits[cls_indx] - modified_unit]

    openmax_fc8 = np.asarray(openmax_layer)
    openmax_score_u = np.asarray(openmax_unknown)

    openmax_probability,modified_scores = computeOpenMaxProbability(openmax_fc8, openmax_score_u)
    
    for s in modified_scores:
        assert(0<=s<=1)
    '''
    softmax_probability=[0]*num_classes
    if compute_softmax:
        softamx_logits = [] 
        for indx in range(num_classes):
            softamx_logits += [sp.exp(logits[indx])]
        denominator = sp.sum(sp.exp(logits))
        softmax_probability = softamx_logits/denominator
    '''
    return np.asarray(openmax_probability), np.asarray(modified_scores)
 


def get_scores(folder,weibull_model,alpha_rank,feature_path,num_classes,class_to_idx_dict,open_set_class_to_idx_dict,save_path):

 
    results = []

    for class_name in os.listdir(os.path.join(feature_path,folder)):
        
        
        for filename in os.listdir(os.path.join(feature_path,folder,class_name)):

            img_features = np.load(os.path.join(feature_path,folder,class_name,filename))

            openmax,modified_scores =  recalibrate_scores(weibull_model, img_features,alpha_rank,'eucos',num_classes,class_to_idx_dict)

            predicted_index=np.argmax(modified_scores)

            

            # find open_set_classification_result according to open_set_class_to_idx_dict 
            open_set_classification_result=None
            for k in open_set_class_to_idx_dict:
                if open_set_class_to_idx_dict[k]==predicted_index:
                    open_set_classification_result=k
                    break

            results.append((class_name,filename,openmax,open_set_classification_result))
    
    # save openset_scores
    
    save_txt_path=os.path.join(save_path,folder+'_openset_scores.txt')
    with open(save_txt_path, 'w') as f:
        for row in results:
            f.write(row[0]+'\t'+row[1]+'\t'+str(row[2])+'\t'+str(row[3])+'\n')
    openmax_scores=[x[2] for x in results]
    return np.array(openmax_scores)


def main(feature_dir='data\saved_features\cifar10',
         weibull_save_path='data\saved_weibull_model\cifar10',
         alpha_rank=6,
         save_path='data\saved_openset_scores\cifar10',
         num_classes=6,
         dataset_dir='data\cifar10'):

    weibull_model_save_folder=os.path.join(weibull_save_path,'train','saved_weibull_model.pkl')
    
    try:
        weibull_model=pickle.load(open(weibull_model_save_folder, 'rb'))
    except:
        raise Exception("error loading pickle object, maybet the pickle object is empty")


    # setup class_to_idx_dict
    trainset = torchvision.datasets.ImageFolder(root=os.path.join(dataset_dir,'train'), )
    class_to_idx_dict=trainset.class_to_idx

    # we will now do N+1 class classification, we setup a open_set_class_to_idx_dict
    open_set_class_to_idx_dict=dict()
    open_set_class_to_idx_dict['unknown']=0
    for k in class_to_idx_dict:
        open_set_class_to_idx_dict[k]=class_to_idx_dict[k]+1# shift the index of known classes back by 1
    
    
    in_dist_scores=get_scores("val",weibull_model,alpha_rank,feature_dir,num_classes,class_to_idx_dict,open_set_class_to_idx_dict,save_path)
    logging.info("average open set scores for in-distribution images : %f"% np.average(in_dist_scores))
    
    out_set_scores=get_scores("test",weibull_model,alpha_rank,feature_dir,num_classes,class_to_idx_dict,open_set_class_to_idx_dict,save_path)
    

    logging.info("average open set scores for out-of-distribution images : %f"% np.average(out_set_scores ))
    
    logging.info("The AUROC is %f"%calc_auroc(in_dist_scores, out_set_scores,save_path ))
   