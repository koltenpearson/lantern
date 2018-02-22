## some predefined machinery for working with models

import torch
from torch.autograd import Variable
from torchvision import transforms


############################################################################
## trainers

def get_variables(inputs, labels, volatile, on_gpu) :
    """utility function for correctly creating variables)"""
    if on_gpu :
        inputs, labels = Variable(inputs, volatile=volatile).cuda(), Variable(labels, volatile=volatile).cuda()
    else :
        inputs, labels = Variable(inputs, volatile=volatile), Variable(labels, volatile=volatile)

    return inputs, labels


def basic_train(dataloader, model, loss, optimizer, logger) :
    volatile = not model.training

    for data in dataloader :

        inputs, labels = data

        inputs, labels = get_variables(inputs, labels, volatile, model.on_gpu)

        if model.training :
            optimizer.zero_grad()

        outputs = model(inputs)

        loss_out = loss(outputs, labels)

        if model.training :
            loss_out.backward()
            optimizer.step()

        
        logger.log_scalar_direct('loss', loss_out.data.mean())
        logger.log_batch(inputs, outputs, labels)

def gen_gradpool_train(iter_size) :
    return lambda *args : gradpool_train(iter_size, *args)

def gradpool_train(iter_size, dataloader, model, loss, optimizer, logger) :
    iter_counter = 0

    volatile = not model.training
    
    for batch_counter, data in enumerate(dataloader) :

        inputs, labels = data
        inputs, labels = get_variables(inputs, labels, volatile, model.on_gpu)
        
        if model.training and iter_counter == 0 :
            optimizer.zero_grad()

        outputs = model(inputs)
        loss_out = loss(outputs, labels)

        if model.training :
            loss_out = loss_out / iter_size
            loss_out.backward()
            iter_counter += 1
            if (iter_counter == iter_size or batch_counter == (len(dataloader)-1)) :
                optimizer.step()
                iter_counter = 0

        logger.log_scalar_direct('loss', loss_out.data.mean())
        logger.log_batch(inputs, outputs, labels)

def adversarial_train(dataloader, models, loss, optimizers, logger) :
    g_model, d_model = models
    g_optim, d_optim = optimizers

    #discriminator pass
    for data in dataloader :

        #real data
        real_inputs, original_labels = data
        fake_seeds = g_model.generator_input(real_inputs, original_labels)

        real_inputs, real_labels = get_variables(inputs, torch.ones(real_inputs.shape[0]), not d_model.training, d_model.on_gpu)

        if d_model.training :
            d_optim.zero_grad()

        real_outputs = d_model(real_inputs)
        real_loss_out = loss(real_outputs, real_labels)

        if d_model.training :
            real_loss_out.backward()

        fake_seeds, fake_labels = get_variables(fake_seeds, torch.zeros(fake_seeds.shape[0]), not g_model.training, g_model.on_gpu)
        fake_inputs = g_model(fake_seeds).detach() #we don't want to train the generator on these so detach and save some time
        fake_outputs = d_model(fake_inputs)
        fake_loss_out = loss(fake_outputs, fake_labels)

        if d_model.training :
            fake_loss_out.backward()
            d_optim.step()

        logger.log_scalar_direct('d_loss', torch.cat((fake_loss_out.data, real_loss_out.data)).mean())
        logger.log_batch(torch.cat((real_inputs,fake_inputs)), torch.cat((real_outputs, fake_outputs)), torch.cat((real_labels, fake_labels)), prefix='d_')

    #generator pass
    for data in dataloader :

        if g_model.training :
            g_optim.zero_grad()

        original_input, original_labels = data
        g_seed = g_model.generator_input(original_input, original_labels)
        g_seed, g_labels = get_variables(seed, torch.ones(seed.shape[0]), not g_model.training, g_model.on_gpu)

        g_inputs = g_model(g_seed)
        g_output = d_model(g_inputs)
        g_loss_out = loss(g_output, labels)

        if g_model.training :
            g_loss_out.backward()
            g_optim.step()

        logger.log_scalar_direct('g_loss', g_loss_out.data.mean())
        logger.log_batch(g_seed, g_inputs, g_output, prefix='g_')


############################################################################
## metrics

def accuracy_metric(outputs, labels) :
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == labels).sum()
    return correct/labels.size(0)

def l2_distance_metric(outputs, labels) :
    distance = (outputs - labels)**2
    distance = distance.sum(2)
    distance = distance.sqrt()
    
    return distance.sum()/labels.size(0)

############################################################################
## preprocessing

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

imagenet_train_preproc = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std)
    ])

imagenet_eval_preproc = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std)
    ])

############################################################################
## defaults

default_train = basic_train
default_scheduler = lambda x, y : None
default_loss = lambda : torch.nn.CrossEntropyLoss()
default_optimizer = lambda x : torch.optim.Adam(x.parameters())
default_register_metrics = lambda x : None

