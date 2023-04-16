import os
import shutil
from src.crosr.MAV_Compute import main
from src.utils.safe_mkdir import safe_mkdirs
from configs.config import per_stage_settings
from src.utils.logger_init import logger_init
logger_init()
import logging
logging.getLogger(__name__)
if __name__ == '__main__':
    d:dict = per_stage_settings.get("MAV_Compute")
    
    num_classes=d.get("num_classes")
    save_path=d.get("save_path")
    feature_dir=d.get("feature_dir")
    dataset_dir=d.get("dataset_dir")
    if os.path.exists(save_path):
        logging.info("mav save_path exists, removing")
        shutil.rmtree(save_path)
    safe_mkdirs(save_path)
    main(save_path,feature_dir,dataset_dir,num_classes)