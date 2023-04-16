import glob
import csv
import os
import shutil

# logger init
import logging
logger = logging.getLogger(__name__)
# end logger init

def main(input_data_dir, torch_training_data_dir, torch_validation_data_dir,torch_testing_data_dir,trainLabels_csv_path):

    # grab all classes
    classes=dict()# classes['category']: a list of imgs under this category
    with open(trainLabels_csv_path, 'r') as file:
        csvreader = csv.reader(file)
        is_first=True
        for row in csvreader:
            if  is_first:
                is_first=False
                continue
            img=row[0]
            category=row[1]
            if classes.get(category)==None:
                classes[category]=[]
            
            classes[category].append(img)
    logging.info("list of categories : %s"%str(classes.keys()))

    # use [airplane,automobile,bird,truck] as test sets
    test_categories=['airplane','automobile','bird','truck']
    logging.info("list of testing categories : %s"%str(test_categories))
    train_and_val_categories=[x for x in classes.keys() if x not in test_categories]
    logging.info("list of train_and_val categories : %s"%str(train_and_val_categories))
    # create dirs for each category for [train,val,test]
    for d in [torch_training_data_dir,torch_validation_data_dir]:
        for c in train_and_val_categories:
            if not os.path.exists(d+"/"+c):
                os.mkdir(d+"/"+c)
    for c in test_categories:
        if not os.path.exists(torch_testing_data_dir+"/"+c):
            os.mkdir(torch_testing_data_dir+"/"+c)

    

    # copy testing imgs
    img_src_dir=input_data_dir
    for k in test_categories:
        logging.info("copy testing imgs for category %s"%k)
        for img in classes[k]:
            shutil.copyfile(img_src_dir+'/'+img+'.png',torch_testing_data_dir+"/"+k+'/'+img+'.png')
            

    # train:val=8:2
    val_ratio=.2
    train_ratio=1.0-val_ratio

    # copy train and val imgs
    img_src_dir=input_data_dir
    

    for k in train_and_val_categories:

        logging.info("copy train and val imgs for category %s"%k)
        size_of_training_set_for_this_category=len(classes[k])*train_ratio
        logging.info("size_of_training_set_for_this_category %d"%size_of_training_set_for_this_category)
        training_set_count=0

        for img in classes[k]:
            
            if training_set_count<size_of_training_set_for_this_category:
                dest_dir=torch_training_data_dir
            else:
                dest_dir=torch_validation_data_dir

            shutil.copyfile(img_src_dir+'/'+img+'.png',dest_dir+"/"+k+'/'+img+'.png')
            training_set_count+=1
    

