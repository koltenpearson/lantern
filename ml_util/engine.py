from .log import Logger
import torch
from torch.utils.data import DataLoader

def train_model(model_info, data_dir, target_epoch, on_gpu=True, threads=2, run_id=-1) :

    if run_id == -1 :
        run_id = model_info.get_new_run()

    print(f"training {model_info.root}:{run_id}")

    checkpoint_path = model_info.get_checkpoint_path(run_id)

    last_epoch = -1
    if checkpoint_path.exists() :
        state = torch.load(checkpoint_path)
        model_info.hparams.update(state['hparams'])
        last_epoch = state['last_epoch']
        print("restoring from checkpoint {checkpoint_path} at epoch {last_epoch + 1}")

    model = model_info.get_model()

    if on_gpu :
        print("using gpus")
        model = model.cuda()
    model.on_gpu = on_gpu

    loss = model_info.get_loss()
    optimizer = model_info.get_optimizer(model)

    if checkpoint_path.exists() :
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optim'])

    scheduler = model_info.get_scheduler(optimizer, last_epoch)

    logger = Logger(model_info.get_log_path(run_id), model)
    model_info.register_metrics(logger)

    print(f"loading data from {data_dir}")
    dsets = model_info.get_datasets(data_dir)

    train_loader = DataLoader(dsets['train'],
            batch_size=model_info.hparams['batch_size'],
            shuffle=True,
            num_workers=threads)
    if 'val' in dsets :
        val_loader = DataLoader(dsets['val'],
                batch_size=model_info.hparams['batch_size'],
                shuffle=False,
                num_workers=threads)
    if 'test' in dsets :
        test_loader = DataLoader(dsets['test'],
                batch_size=model_info.hparams['batch_size'],
                shuffle=False, 
                num_workers=threads)

    print("starting training")
    logger.log_start(model_info.hparams, data_dir)
    while last_epoch < target_epoch :
        last_epoch += 1

        model.train()
        logger.log_epoch_start('train')
        model_info.train(train_loader, model, loss, optimizer, logger)

        model.eval()
        if 'val' in dsets :
            logger.log_epoch_start('val')
            model_info.train(val_loader, model, loss, optimizer, logger)

        if scheduler is not None :
            scheduler.step()

        torch.save({'last_epoch' : last_epoch,
                    'optim' : optimizer.state_dict(),
                    'model' : model.state_dict(),
                    'hparams' : model_info.hparams}, checkpoint_path)
        logger.log_completion()

        print(f"finished epoch {last_epoch}")


    if 'test' in dsets :

        logger.log_epoch_start('test')
        model_info.train(test_loader, model, loss, optimizer, logger)
        logger.log_completion()






