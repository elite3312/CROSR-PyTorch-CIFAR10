import os
import shutil
from src.crosr.compute_openmax import main
from src.utils.safe_mkdir import safe_mkdirs
from configs.config import per_stage_settings
from src.utils.logger_init import logger_init
logger_init()
import logging
logging.getLogger(__name__)
if __name__ == '__main__':
    d:dict = per_stage_settings.get("compute_openmax")
    feature_dir=d.get("feature_dir")
    weibull_save_path=d.get("weibull_save_path")
    
    alpha_rank=d.get("alpha_rank")
    save_path=d.get("save_path")
    num_classes=d.get("num_classes")
    dataset_dir=d.get("dataset_dir")
    if os.path.exists(save_path):
        logging.info("saved openset scores exist, removing")
        shutil.rmtree(save_path)
    safe_mkdirs(save_path)
    
    main(feature_dir,weibull_save_path,alpha_rank,save_path,num_classes,dataset_dir)