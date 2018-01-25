#holds classes that represent files and folders on the disk

from pathlib import Path
import importlib
import json
from . import machinery
import pkgutil
import shutil

GEN_STRING = "#GENERATED"

README_NAME = 'readme.txt'
NOTES_NAME = 'notes.txt'
MODEL_DEF_NAME = 'model_def.py'
RUN_DIR = 'runs'
CHECKPOINT_NAME = 'state.pt'
LOG_NAME = 'log.json'
ARCHIVE_DIR = '.archive'

def read_note(notepath) :
    raw = open(notepath).readlines()

    cleaned = []
    for r in raw :
        r = r.strip()
        if (len(r) == 0 or r[0] == '#') :
            continue
        cleaned.append(r)
    return cleaned

#defines operations on the experiment file structure
class MetaExperiment :

    @staticmethod
    def init_experiment(path) :
        path = Path(path)
        (path/ARCHIVE_DIR).mkdir()

        for f in (README_NAME, NOTES_NAME, MODEL_DEF_NAME) :
            cpath = path/f
            if not cpath.exists() :
                template_data = pkgutil.get_data(__name__, f'pkg_data/templates/{f}')

                with open(cpath, 'wb') as out_file :
                    out_file.write(template_data)

    def __init__(self, exp_root) :
        self.root = Path(exp_root)
        self.archive_dir = self.root/ARCHIVE_DIR
        if not self.archive_dir.exists() :
            raise FileNotFoundError

    def get_archive_description(self) :
        result = {}
        loaded_id = None
        for a in self.archive_dir.iterdir() :
            store_id = int(a.parts[-1])
            notes = read_note(a/NOTES_NAME)
            result[store_id] = notes
            if a.is_symlink() :
                loaded_id = store_id
        return result, loaded_id

    def get_store_id(self) :
        store_id = 0
        for p in self.archive_dir.iterdir() :
            current_id = int(p.parts[-1])
            if p.is_symlink() :
                return current_id
            if store_id < current_id :
                store_id = current_id

        return store_id + 1

    def check_notes(self) :
        notes_path = self.root/NOTES_NAME
        if not notes_path.exists() :
            open(notes_path, 'wb').write(pkgutil.get_data(__name__, f'pkg_data/templates/{NOTES_NAME}'))
            return False

        notes = open(self.root/NOTES_NAME).read()
        notes = notes.strip()
        if len(notes) <= 0 or notes[:len(GEN_STRING)] == GEN_STRING :
            return False
        return True

    def store(self) :
        if not self.check_notes() :
            print(f"WARNING: store aborted due to empty {NOTES_NAME}")
            return False

        store_id = self.get_store_id()

        store_dir = self.archive_dir/str(store_id)

        if store_dir.is_symlink() :
            store_dir.unlink()

        store_dir.mkdir(exist_ok=True)

        shutil.copy(self.root/MODEL_DEF_NAME, store_dir/MODEL_DEF_NAME)
        shutil.move(self.root/NOTES_NAME, store_dir/NOTES_NAME)
        if (self.root/RUN_DIR).exists() :
            shutil.move(self.root/RUN_DIR, store_dir/RUN_DIR)

        return True

    def symlink_in_archive(self) :
        for p in self.archive_dir.iterdir() :
            if p.is_symlink() :
                return True
        return False

    def check_if_model_in_use(self) :
        if (self.symlink_in_archive() or 
                (self.root/NOTES_NAME).exists() or
                (self.root/RUN_DIR).exists()) :
            return True
        return False


    def retrieve(self, store_id) :
        if self.check_if_model_in_use() :
            stored = self.store()
            if not stored :
                print("WARNING: retrieve aborted because it would overwite data")
                return False
            print("stored currently loaded model")

        store_dir = self.archive_dir/str(store_id)
        if not store_dir.exists() :
            print(f"WARNING: retrieve aborted, no such store id {store_id}")
            return False

        shutil.move(store_dir/MODEL_DEF_NAME, self.root/MODEL_DEF_NAME)
        shutil.move(store_dir/NOTES_NAME, self.root/NOTES_NAME)
        if (store_dir/RUN_DIR).exists() :
            shutil.move(store_dir/RUN_DIR, self.root/RUN_DIR)

        store_dir.rmdir()
        store_dir.symlink_to(self.root.resolve())
        return True


class ModelInfo :

    def __init__(self, root) :
        self.root = Path(root)

        self.run_base = self.root / RUN_DIR
        self.run_base.mkdir(exist_ok=True)
        self.next_rid = self._calc_next_run_id()

        self._init_passthrough_methods()

    def _calc_next_run_id(self) :
        cid = 0
        for p in self.run_base.iterdir() :
            if cid < int(p.parts[-1]) :
                cid = int(p.parts[-1])

        return cid + 1

    def get_run_path(self, id) :
        return self.run_base / str(id)

    def get_checkpoint_path(self, id) :
        return self.run_base / str(id) / CHECKPOINT_NAME

    def get_log_path(self, id) :
        return self.run_base / str(id) / LOG_NAME

    def get_new_run(self) :
        result = self.next_rid
        path = self.run_base / str(result)
        path.mkdir()

        self.next_rid += 1

        return result


    def _init_passthrough_methods(self) :
        spec = importlib.util.spec_from_file_location('model_def', self.root/'model_def.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        self.get_datasets = module.get_datasets
        self.get_model = module.get_model
        self.hparams = module.hparams


        try :
            self.get_loss = module.get_loss
        except AttributeError :
            self.get_loss = machinery.default_loss

        try :
            self.train = module.train
        except AttributeError :
            self.train = machinery.default_train

        try :
            self.get_optimizer = module.get_optimizer
        except AttributeError :
            self.get_optimizer = machinery.default_optimizer

        try :
            self.get_scheduler = module.get_scheduler
        except AttributeError :
            self.get_scheduler = machinery.default_scheduler

        try :
            self.register_metrics = module.register_metrics
        except AttributeError :
            self.register_metrics = machinery.default_register_metrics

class Logger :

    def __init__(self, logdir, max_buf_size=50) :
        self.path = Path(logdir)
        self.buffer = []
        self.max_buf_size = max_buf_size
        self.metrics = {}

        self.epoch_count = 0
        self.batch_count = 0
        self.split = 'none'
    
    def flush(self) :
        outfile = open(self.path, 'a')
        for entry in self.buffer :
            outfile.write(json.dumps(entry))
            outfile.write('\n')
        self.buffer.clear()
        outfile.close()


    def register_metric(self, key, metric) :
        self.metrics[key] = metric

    def epoch_step(self) :
        self.epoch_count += 1
        self.batch_count = 0
        self.flush()


    def batch_step(self, outputs, labels, loss) :
        loss = loss.data.mean()
        entry = {'split' : self.split, 'epoch' : self.epoch_count, 'batch' : self.batch_count, 'loss' : loss}

        for k, m in self.metrics.items() :
            entry[k] = m(outputs.data, labels.data)

        self.buffer.append(entry)

        self.batch_count += 1

        if len(self.buffer) >= self.max_buf_size :
            self.flush()


