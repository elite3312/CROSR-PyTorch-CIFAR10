import os
from src.utils.logger_init import logger_init
logger_init()
import logging
logging.getLogger(__name__)
import shutil
from src.convert_input_2_fit_torch.arrange_cifar10_2_fit_datasetfolder import main
from configs.config import per_stage_settings
from src.utils.safe_mkdir import safe_mkdirs
if __name__ == '__main__':

    # read dict from config
    d:dict = per_stage_settings.get("convert_cifar10_input_2_fit_torch")
    trainLabels_csv_path=d["trainLabels_csv_path"]
    input_data_dir = d["input_data_dir"]
    torch_training_data_dir = d["torch_training_data_dir"]
    torch_validation_data_dir = d["torch_validation_data_dir"]
    torch_testing_data_dir = d["torch_testing_data_dir"]

    # make dirs
    if os.path.exists(torch_training_data_dir):
        logging.info("torch_training_data_dir exists, removing...")
        shutil.rmtree(torch_training_data_dir)
    safe_mkdirs(torch_training_data_dir)
    if os.path.exists(torch_validation_data_dir):
        logging.info("torch_validation_data_dir exists, removing...")
        shutil.rmtree(torch_validation_data_dir)
    safe_mkdirs(torch_validation_data_dir)
    if os.path.exists(torch_testing_data_dir):
        logging.info("torch_testing_data_dir exists, removing...")
        shutil.rmtree(torch_testing_data_dir)
    safe_mkdirs(torch_testing_data_dir)

    # call arrange_images_2_fit_datasetfolder
    main(input_data_dir, torch_training_data_dir, torch_validation_data_dir,torch_testing_data_dir,trainLabels_csv_path)
