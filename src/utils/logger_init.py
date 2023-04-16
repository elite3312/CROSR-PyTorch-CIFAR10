# logger init
import sys
import os
from logging.handlers import RotatingFileHandler
import logging

def logger_init():
    if not os.path.isdir('./logs'):
        try:
            os.mkdir('./logs')
        except:
            raise Exception("error in os.mkdir")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            # logging.FileHandler('logs/app-basic.log'),
            logging.handlers.TimedRotatingFileHandler(
                'logs/app-basic.log', when='midnight', interval=1, backupCount=30),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
# end logger init