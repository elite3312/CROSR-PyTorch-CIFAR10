import os
import shutil
from src.crosr.weibull_fitting import main
from src.utils.safe_mkdir import safe_mkdirs
from configs.config import per_stage_settings
from src.utils.logger_init import logger_init
logger_init()
import logging
logging.getLogger(__name__)
if __name__ == '__main__':
    d:dict = per_stage_settings.get("weibull_fitting")
    distance_scores_path=d.get("distance_scores_path")
    MAV_path=d.get("MAV_path")
    weibull_tail_size=d.get("weibull_tail_size")
    save_path=d.get("save_path")
    if os.path.exists(save_path):
        logging.info("saved weibull models exist, removing...")
        shutil.rmtree(save_path)
    safe_mkdirs(save_path)
    
    main(distance_scores_path,MAV_path,weibull_tail_size,save_path)