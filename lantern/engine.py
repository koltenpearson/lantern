from .log import Logger
from .save import Saver
import torch


def run_inference(model, run_id, data_dir) :

    print(f"running inference on {model.root}:{run_id} with data from {data_dir}")

    saver = Saver(model.get_checkpoint_path(run_id), model.get_pretrained_path())
    saver.set_hparams(model.hparams)

    model.init_model(saver)

    if saver.checkpoint_exists() :
        last_epoch = saver.restore()
        print(f"restoring from checkpoint, epoch {last_epoch + 1}")
    else :
        print(f"no checkpoint exists for {model.root}:{run_id}")
        return False

    try :
        model.run_inference(data_dir)
    except AttributeError :
        print(f"run_inference method not defined")

    return True


def train_model(model, data_dir, target_epoch, run_id=-1) :

    if run_id == -1 :
        run_id = model.get_new_run()

    print(f"training {model.root}:{run_id}")

    saver = Saver(model.get_checkpoint_path(run_id), model.get_pretrained_path())
    saver.set_hparams(model.hparams)

    logger = Logger(model.get_log_path(run_id))
    logger.log_start(model.hparams, data_dir)

    model.init_model(saver)

    last_epoch = -1 
    if saver.checkpoint_exists() :
        last_epoch = saver.restore()
        print(f"restoring from checkpoint, starting at epoch {last_epoch + 1}")
    
    model.init_datasets(data_dir)

    model.train(last_epoch+1, target_epoch, logger, saver)

    logger.flush()



