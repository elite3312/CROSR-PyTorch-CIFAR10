import os
data_root_dir='data'# path to a folder used to store the input and output of this repo
dataset='cifar10'# [cifar10,??]
image_side_length=32 # 32 for cifar10
channes=3# 3 for rgb images
num_classes=6 # number of classes learned duing training. 
# In case of cifar 10, 6 classes are used for training, 4 classes are outliers

#----------don't change below-----------#
saved_dhr_model=os.path.join(data_root_dir,'saved_dhr_model')
saved_feature_dir=os.path.join(data_root_dir,'saved_features')
saved_MAV_dir=os.path.join(data_root_dir,'saved_MAVs')
saved_distance_scores=os.path.join(data_root_dir,'saved_distance_scores')
saved_weibull_model=os.path.join(data_root_dir,'saved_weibull_model')
saved_openset_scores=os.path.join(data_root_dir,'saved_openset_scores')

per_stage_settings={
    "convert_cifar10_input_2_fit_torch":{
        "input_data_dir":os.path.join(data_root_dir,'input_data_cifar10','train'),
        "trainLabels_csv_path":os.path.join(data_root_dir,'input_data_cifar10',"trainLabels.csv"),
        "torch_training_data_dir":os.path.join(data_root_dir,'cifar10','train'),
        "torch_validation_data_dir":os.path.join(data_root_dir,'cifar10','val'),
        "torch_testing_data_dir":os.path.join(data_root_dir,'cifar10','test'),
    },
    "train_dhr":{
        "lr":0.0001,
        "epochs":5,
        'batch_size':60,
        "save_path":os.path.join(saved_dhr_model,dataset),
        "dataset_dir":os.path.join(data_root_dir,dataset),
        "num_classes":num_classes,
        "means":[0.4914, 0.4822, 0.4465],#"channelwise means for normalization"
        "stds":[0.2023, 0.1994, 0.2010],#"channelwise std for normalization"
        "momentum":0.9,
        "weight_decay":0.0005,
        "image_side_length":image_side_length,
        "channes":channes,
        "load_and_continue":True
    },
    "get_model_features":{
        "dataset_dir":os.path.join(data_root_dir,dataset),
        "num_classes":num_classes,
        "means":[0.4914, 0.4822, 0.4465],#"channelwise means for normalization"
        "stds":[0.2023, 0.1994, 0.2010],#"channelwise std for normalization"
        "save_path":os.path.join(saved_feature_dir,dataset),
        "load_path":os.path.join(saved_dhr_model,dataset,"best_val_acc.pth"),
        "image_side_length":image_side_length,
        "channes":channes
    },
    "MAV_Compute":{
        "save_path":os.path.join(saved_MAV_dir,dataset),
        "feature_dir":os.path.join(saved_feature_dir,dataset),
        "dataset_dir":os.path.join(data_root_dir,dataset),
        "num_classes":num_classes,
    },
    "compute_distances":{
        "MAV_path":os.path.join(saved_MAV_dir,dataset),
        "save_path":os.path.join(saved_distance_scores,dataset),
        "feature_dir":os.path.join(saved_feature_dir,dataset),
        "dataset_dir":os.path.join(data_root_dir,dataset),
    },
    "weibull_fitting":{
        "MAV_path":os.path.join(saved_MAV_dir,dataset),
        "distance_scores_path":os.path.join(saved_distance_scores,dataset),
        "weibull_tail_size":35,
        "save_path":os.path.join(saved_weibull_model,dataset),
        "distance_type":"eucos"
    },
    "compute_openmax":{
        "weibull_save_path":os.path.join(saved_weibull_model,dataset),
        "alpha_rank":num_classes,# alpha_rank <= num_classes
        "feature_dir":os.path.join(saved_feature_dir,dataset),
        "num_classes":num_classes,
        "dataset_dir":os.path.join(data_root_dir,dataset),
        "save_path":os.path.join(saved_openset_scores,dataset)
    },
    "analyze_results":{
        "saved_openset_scores":os.path.join(saved_openset_scores,dataset)
    }
}

