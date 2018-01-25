## some predefined machinery for working with models

import torch
from torch.autograd import Variable


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


        logger.batch_step(outputs, labels, loss_out)

def accuracy_metric(outputs, labels) :
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == labels).sum()
    return correct/labels.size(0)



default_train = basic_train
default_scheduler = lambda x, y : None
default_loss = lambda : torch.nn.CrossEntropyLoss()
default_optimizer = lambda x : torch.optim.Adam(x.parameters())
default_register_metrics = lambda x : None

