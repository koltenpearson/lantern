"""represention of a model and its file structure"""

from pathlib import Path
import importlib
import json
import pkgutil
import shutil
import lzma
from types import SimpleNamespace

MODEL_INFO = 'info.txt'
RUN_DIR = 'runs'
MODEL_DEF = 'model_def.py'
CHECKPOINT_NAME = 'state.pt'
LOG_NAME = 'log.bin'
STORED_DEF_NAME = 'def.xz'
PRETRAINED_PATH = 'pretrained'
DESCRIPTION_NAME = 'description.txt'

def write_template(t_name, dest) :
    dest = Path(dest)


    if not dest.exists() :
        template_data = pkgutil.get_data(__name__, f'pkg_data/templates/{t_name}')
        with open(dest, 'wb') as out_file :
            out_file.write(template_data)


class ModelLoadError(Exception) :
    pass

class Model :

    @staticmethod
    def create_new(root) :
        root = Path(root)
        root.mkdir(exist_ok=True)
        write_template(MODEL_DEF, root/MODEL_DEF)
        write_template(MODEL_INFO, root/MODEL_INFO)


    @staticmethod
    def load_if_exists(root) :
        root = Path(root)
        if (root/MODEL_DEF).exists() :
            try :
                return Model(root)
            except AttributeError :
                pass
            except ModelLoadError :
                pass

        return None

    def __init__(self, root) :

        self.root = Path(root)

        self.run_dir = self.root/RUN_DIR
        self.model_def_dir = self.root/MODEL_DEF
        self.info_dir = self.root/MODEL_INFO
        self.pretrained_path = self.root/PRETRAINED_PATH

        self.name = self.root.resolve().parts[-1]

        self.next_rid = self._calc_next_run_id()
        
        with open(self.model_def_dir) as codefile :
            self._load_internals(codefile.read())

    def __repr__(self) :
        return f'<Model - {self.name}>'

    def _load_internals(self, code) :
        module = SimpleNamespace()
        try :
            exec(code, module.__dict__)
        except :
            raise ModelLoadError(f"error while loading model_def.py for {self.name}")

        self.init_datasets = module.init_datasets
        self.init_model = module.init_model
        self.train = module.train
        self.hparams = module.hparams

        try :
            self.run_inference = module.run_inference
        except AttributeError :
            try :
                del self.run_inference
            except AttributeError :
                pass

    def save_def(self, rid) :
        with open(self.model_def_dir) as code_file :
            code = code_file.read().encode('utf-8')

        with lzma.open(self.run_dir/str(rid)/STORED_DEF_NAME, 'w') as code_file :
            code_file.write(code)

    def get_stored_def(self, rid) :
        with lzma.open(self.run_dir/str(rid)/STORED_DEF_NAME) as codefile :
            return codefile.read().decode('utf-8')

    def load_stored_def(self, rid) :
        self._load_internals(self.get_stored_def(rid))

    def get_description(self, rid) :
        try :
            with open(self.run_dir/str(rid)/DESCRIPTION_NAME) as mfile :
                return mfile.read()

        except FileNotFoundError :
            return None

    def set_description(self, rid, mess) :
        with open(self.run_dir/str(rid)/DESCRIPTION_NAME, 'w') as mfile :
            mfile.write(mess)

    def get_log_path(self, rid) :
        return self.run_dir / str(rid) / LOG_NAME

    def get_run_path(self, id=None) :
        if id is None :
            return self.run_dir
        return self.run_dir / str(id)

    def get_checkpoint_path(self, rid) :
        return self.run_dir / str(rid) / CHECKPOINT_NAME

    def _calc_next_run_id(self) :
        cid = 0
        if not self.run_dir.exists() :
            return cid

        for p in self.run_dir.iterdir() :
            if cid < int(p.parts[-1]) :
                cid = int(p.parts[-1])

        #TODO come up with more certain way to reload new id if past debugging stage
        # best way is to make process that removes checkpoint and replaces log with summary to save space
        # check for that before reusing existing run id as well
        if not self.get_checkpoint_path(cid).exists() :
            return cid

        return cid + 1

    def get_new_run(self) :
        result = self.next_rid
        self.get_run_path(result).mkdir(parents=True, exist_ok=True)
        self.save_def(result)

        self.next_rid += 1
        return result

    def list_runs(self) :
        if not self.get_run_path().exists() :
            return []

        return [int(p.parts[-1]) for p in self.get_run_path().iterdir()]

