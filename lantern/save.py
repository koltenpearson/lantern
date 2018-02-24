from pathlib import Path
import torch


#TODO add in support for saving state of RNG as well
class Saver :

    def __init__(self, checkpoint_path) :
        self.cp_path = Path(checkpoint_path)

        self.models = {}
        self.optimizers = {}
        self.hparams = {}

    def add_model(self, key, model) :
        self.models[key] = model

    def add_optimizer(self, key, optim) :
        self.optimizers[key] = optim

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
            save_dict['models'][o_k] = self.optimizers[o_k].state_dict()

        torch.save(save_dict, self.cp_path)

   def checkpoint_exists(self) :
       return self.cp_path.exists()

   def restore(self) :
       load_dict = torch.load(self.cp_path)
       
       for m_k in self.models :
           self.models[m_k].load_state_dict(load_dict['models'][m_k])

       for o_k in self.optmizers :
           self.optmizers[o_k].load_state_dict(load_dict['optimizers'][o_k])

        self.hparams.update(load_state_dict['hparams'])

        return load_state_dict['epoch']


