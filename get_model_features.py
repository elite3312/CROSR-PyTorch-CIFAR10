import os
import shutil
from src.crosr.get_model_features import main
from src.utils.safe_mkdir import safe_mkdirs
from configs.config import per_stage_settings
from src.utils.logger_init import logger_init
logger_init()
import logging
logging.getLogger(__name__)
if __name__ == '__main__':
    d:dict = per_stage_settings.get("get_model_features")
    num_classes=d.get("num_classes")
    means=d.get("means")
    stds=d.get("stds")
    dataset_dir=d.get("dataset_dir")
    load_path=d.get("load_path")
    image_side_length=d.get("image_side_length")
    save_path=d.get("save_path")
    channes=d.get("channes")
    if os.path.exists(save_path):
        logging.info("features save_path exists, removing...")
        shutil.rmtree(save_path)
    safe_mkdirs(save_path)
    
    main(num_classes,means,stds,dataset_dir,load_path,image_side_length,save_path,channes)