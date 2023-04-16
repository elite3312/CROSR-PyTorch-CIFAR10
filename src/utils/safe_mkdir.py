import os
import logging
logger = logging.getLogger(__name__)

def safe_mkdir(dir_name:str):
    try:
        os.mkdir(dir_name)
    except: 
        logging.error("An error occurred while creating dir %s"%dir_name, exc_info=True)
        raise Exception("An error occurred while creating dir%s"%dir_name)
def safe_mkdirs(dir_name:str):
    try:
        os.makedirs(dir_name)
    except: 
        logging.error("An error occurred while creating dir %s"%dir_name, exc_info=True)
        raise Exception("An error occurred while creating dir%s"%dir_name)
    
