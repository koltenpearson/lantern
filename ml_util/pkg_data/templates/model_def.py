#GENERATED remove this line after modified

############################################################################
## hyperparameters

## these may be change by outside tools but
# they will be changed before any other method is called
# use the dict for things like learning rates and other 
# values you might want to train in a run

hparams = {
        'batch_size' : 128, #this one always needs to be defined
        }


############################################################################
## dataset 

## this function is given a directly passed from the user, and needs
# to turn it into datasets for training and validation
# the datasets should be subclasses of torch.utils.data.Dataset

def get_datasets(data_dir) :
    test_set = torchvision.datasets.MNIST(data_dir, train=False, download=False, transform=preproc)

    #not all of these need to be included in the dict, though train is necessary
    return {'train' : None, 'val' : None, 'test' : None}

############################################################################
## Model 

##should return a subclass of torch.nn.Module

def get_model() :
    return None

#############################################################################
## Training

## returns the loss function, subclass of torch.nn.Module, will use
# a default classification loss if not defined

#def get_loss() :
#    return None

## passes in the model and returns an optimizer, will use
# a default optimizer with sane parameters if not defined

#def get_optimizer(model) :
#    return None

## pass in the optimizer, and and int marking the last epoch
# should return a scheduler following the ones found in
# torch.optim.lr_scheduler for adjusting the learning rate during training
# the scheduler initialized that that last_epoch was the last epoch in its 
# schedule.

#def get_scheduler(optimizer, last_epoch) :
#    return None

############################################################################
## Logging

## passes in a Logger object, you can register metrics with the function
# register_metric(key, func(outputs, labels)) the func passed in should return
# a scalar. The logger will write out the result for every batch. See Logger 
# source for more functionality

#def register_metrics(logger) :
#    logger.register_metric('accuracy', ml_util.accuracy_metric)




