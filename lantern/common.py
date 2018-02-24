## some predefined machinery for working with models

import torch
from torch.autograd import Variable
from torchvision import transforms
import numpy as np


############################################################################
## training

class VarGen :
    """class to generate variables accounting for gpu and volatility)"""

    def __init__(self, volatile, on_gpu) :
        self.volatile = volatile
        self.on_gpu = on_gpu

    def get_variables(self, *tensors) :
        result = []
        for t in tensors :
            result.append(Variable(t, volatile=self.volatile))

        if self.on_gpu :
            for i in range(len(result)) :
                result[i] = result[i].cuda()

        return tuple(result)

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

