import os
import shutil
from src.crosr.compute_distances import main
from src.utils.safe_mkdir import safe_mkdirs
from configs.config import per_stage_settings
from src.utils.logger_init import logger_init
logger_init()
import logging
logging.getLogger(__name__)
if __name__ == '__main__':
    d:dict = per_stage_settings.get("compute_distances")
    MAV_path=d.get("MAV_path")
    save_path=d.get("save_path")
    feature_dir=d.get("feature_dir")
    dataset_dir=d.get("dataset_dir")
    if os.path.exists(save_path):
        logging.info("saved distances exists, removing...")
        shutil.rmtree(save_path)
    safe_mkdirs(save_path)
    
    main(feature_dir,MAV_path,save_path,dataset_dir)