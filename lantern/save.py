from pathlib import Path
import torch

import msgpack
import msgpack_numpy
msgpack_numpy.patch()

#TODO finish this later
class LogitSaver :

    def __init__(self, output_path) :
        pass



#TODO add in support for saving state of RNG as well
class Saver :

    @staticmethod
    def save_model_dict(checkpoint_path, key, dest_path) :
        load_dict = torch.load(checkpoint_path)
        torch.save(load_dict['models'][key], dest_path)

    def __init__(self, checkpoint_path, pre_trained_path) :
        self.cp_path = Path(checkpoint_path)
        self.pt_path = Path(pre_trained_path)

        self.models = {}
        self.optimizers = {}
        self.hparams = {}

    def load_pretrained_weights(self, model, filename) :
        model.load_state_dict(torch.load(self.pt_path/filename))

    def add_model(self, key, model) :
        self.models[key] = model

    def add_optimizer(self, key, optim) :
        self.optimizers[key] = optim

    def set_hparams(self, params) :
        self.hparams = params

    def save(self, epoch) :
        save_dict = {
                'models' : {},
                'optimizers' : {}, 
                'hparams' : self.hparams,
                'epoch' : epoch, 
                }

        for m_k in self.models :
            save_dict['models'][m_k] = self.models[m_k].state_dict()

        for o_k in self.optimizers :
            save_dict['optimizers'][o_k] = self.optimizers[o_k].state_dict()

        torch.save(save_dict, self.cp_path)

    def checkpoint_exists(self) :
        return self.cp_path.exists()

    def restore(self) :
        load_dict = torch.load(self.cp_path)
 
        for m_k in self.models :
            self.models[m_k].load_state_dict(load_dict['models'][m_k])
 
        for o_k in self.optimizers :
            self.optimizers[o_k].load_state_dict(load_dict['optimizers'][o_k])
 
        self.hparams.update(load_dict['hparams'])
 
        return load_dict['epoch']


