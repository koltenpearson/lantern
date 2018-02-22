from .log import Logger
import torch
from torch.utils.data import DataLoader
from .util import wrap_single, unwrap_single

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
        print(f"restoring from checkpoint {checkpoint_path} at epoch {last_epoch + 1}")

    model = wrap_single(model_info.get_model())

    if on_gpu :
        print("using gpus")
        model = [m.cuda() for m in model]

    for m in model : m.on_gpu = on_gpu

    loss = wrap_single(model_info.get_loss())

    optimizer = wrap_single(model_info.get_optimizer(unwrap_single(model)))

    if checkpoint_path.exists() :
        model_state = wrap_single(state['model'])
        for m,s in zip(model, model_state) : m.load_state_dict(s)

        optim_state = wrap_single(state['optim'])
        for o,s in zip(optimizer, optim_state) : o.load_state_dict(s)

    scheduler = wrap_single(model_info.get_scheduler(unwrap_single(optimizer), last_epoch))

    #logger is aware of the tuple/single transparency, so we don't conditionally unwrap the model
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

        for m in model : m.train()
        logger.log_epoch_start('train')
        model_info.train(train_loader, unwrap_single(model), unwrap_single(loss), unwrap_single(optimizer), logger)

        for m in model : m.eval()
        if 'val' in dsets :
            logger.log_epoch_start('val')
            model_info.train(val_loader, unwrap_single(model), unwrap_single(loss), unwrap_single(optimizer), logger)

        if scheduler[0] is not None :
            for s in scheduler : s.step()

        torch.save({'last_epoch' : last_epoch,
                    'optim' : unwrap_single([o.state_dict() for o in optimizer]),
                    'model' : unwrap_single([m.state_dict() for m in model]),
                    'hparams' : model_info.hparams}, checkpoint_path)
        logger.log_completion()

        print(f"finished epoch {last_epoch}")


    if 'test' in dsets :

        logger.log_epoch_start('test')
        model_info.train(test_loader, unwrap_single(model), unwrap_single(loss), unwrap_single(optimizer), logger)
        logger.log_completion()






