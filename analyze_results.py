import os
import shutil
from src.crosr.analyze_results import main
from src.utils.safe_mkdir import safe_mkdirs
from configs.config import per_stage_settings
from src.utils.logger_init import logger_init
logger_init()
import logging
logging.getLogger(__name__)
if __name__ == '__main__':
    d:dict = per_stage_settings.get("analyze_results")
    saved_openset_scores=d.get("saved_openset_scores")
    
    
    main(saved_openset_scores)