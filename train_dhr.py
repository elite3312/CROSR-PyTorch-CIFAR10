from src.crosr.train_dhr_net import main
from configs.config import per_stage_settings
from src.utils.logger_init import logger_init
logger_init()
import logging
logging.getLogger(__name__)

if __name__ == '__main__':
 
    
    d:dict = per_stage_settings.get("train_dhr")
    lr = d.get("lr")
    epochs = d.get("epochs")
    batch_size = d.get("batch_size")
    momentum = d.get("momentum")
    weight_decay = d.get("weight_decay")
    means = d.get("means")
    stds = d.get("stds")
    num_classes = d.get("num_classes")
    dataset_dir = d.get("dataset_dir")
    save_path=d.get("save_path")
    load_and_continue=d.get("load_and_continue")
    image_side_length=d.get("image_side_length")
    channes=d.get("channes")
    main(lr=lr,
         epochs=epochs,
         batch_size=batch_size,
         momentum=momentum,
         weight_decay=weight_decay,
         means=means,
         stds=stds,
         num_classes=num_classes,
         dataset_dir=dataset_dir,
         image_side_length=image_side_length,
         save_path=save_path,
         load_and_continue=load_and_continue,
         channes=channes
         )
