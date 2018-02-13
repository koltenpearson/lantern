## some predefined machinery for working with models

import torch
from torch.autograd import Variable
from torchvision import transforms


############################################################################
## trainers

def basic_train(dataloader, model, loss, optimizer, logger) :
    volatile = not model.training

    for data in dataloader :

        inputs, labels = data
        if model.on_gpu :
            inputs, labels = Variable(inputs, volatile=volatile).cuda(), Variable(labels, volatile=volatile).cuda()
        else :
            inputs, labels = Variable(inputs, volatile=volatile), Variable(labels, volatile=volatile)

        if model.training :
            optimizer.zero_grad()

        outputs = model(inputs)

        loss_out = loss(outputs, labels)

        if model.training :
            loss_out.backward()
            optimizer.step()

        logger.log_batch(inputs, outputs, labels, loss_out.data.mean())

def gen_gradpool_train(iter_size) :
    return lambda *args : gradpool_train(iter_size, *args)

def gradpool_train(iter_size, dataloader, model, loss, optimizer, logger) :
    iter_counter = 0

    volatile = not model.training
    
    for batch_counter, data in enumerate(dataloader) :

        inputs, labels = data
        if model.on_gpu :
            inputs, labels = Variable(inputs, volatile=volatile).cuda(), Variable(labels, volatile=volatile).cuda()
        else :
            inputs, labels = Variable(inputs, volatile=volatile), Variable(labels, volatile=volatile)

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

        logger.log_batch(inputs, outputs, labels, loss_out.data.mean())


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

