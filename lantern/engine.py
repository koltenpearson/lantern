from .log import Logger
from .save import Saver
import torch

def train_model(model, data_dir, target_epoch, run_id=-1) :

    if run_id == -1 :
        run_id = model.get_new_run()

    print(f"training {model.root}:{run_id}")

    saver = Saver(model.get_checkpoint_path(run_id))
    logger = Logger(model.get_log_path(run_id))

    logger.log_start(model.hparams, data_dir)

    model.init_datasets(data_dir)
    model.init_model(logger, saver)

    last_epoch = -1 
    if saver.checkpoint_exists() :
        last_epoch = saver.restore()
        print(f"restoring from checkpoint, starting at epoch {last_epoch + 1}")
    
    model.train(last_epoch+1, target_epoch, logger, saver)



