#GENERATED remove this line after modified

############################################################################
## hyperparameters

## these may be changed by outside tools but
# they will be changed before any other method is called
# use the dict for things like learning rates and other 
# values you might want a tool to be able to automatically change

hparams = {}

############################################################################
## dataset 

## this function is given a directory passed from the user, to be turned
# into datasets

def init_datasets(data_dir) :
    # while you should keep your own state internally this method can return
    # a subclass of torch.utils.data.Datset, or a dictionary of datasets
    # which can be used by lantern to create visualizations
    return None

############################################################################
## model 

## This function should be used to initialize everything internally
# it will always be called before train, and gives you a Saver
# which should be set up with your parameters
# be sure to remember to convert things with .cuda() if you are using the gpu
def init_model(saver) :
    pass

#############################################################################
## training

## this function should contain your training loop
#start_epoch is the what epoch to start from
#target_epoch is how far to train before stopping
#logger is a Logger initialized to the correct directory
#saver is a Saver, same that was passed to model_init
def train(start_epoch, target_epoch, logger, saver) :
    pass

