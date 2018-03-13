#holds classes that represent files and folders on the disk

from pathlib import Path
import importlib
import json
import pkgutil
import shutil

GEN_STRING = "#GENERATED"

README_NAME = 'readme.txt'
NOTES_NAME = 'notes.txt'
MODEL_DEF_NAME = 'model_def.py'
RUN_DIR = 'runs'
CHECKPOINT_NAME = 'state.pt'
LOG_NAME = 'log.bin'
ARCHIVE_DIR = '.archive'
PRETRAINED_PATH = 'pretrained'

class Structure :

    def __init__(self, root) :
        self.root = Path(root)
        self.run_dir = self.root/RUN_DIR
        self.archive_dir = self.root/ARCHIVE_DIR

    def get_checkpoint_path(self, rid) :
        return self.run_dir / str(rid) / CHECKPOINT_NAME

    def get_model_def_path(self) :
        return self.root / MODEL_DEF_NAME

    def get_pretrained_path(self) :
        return self.root/PRETRAINED_PATH

    def get_log_path(self, rid) :
        return self.run_dir / str(rid) / LOG_NAME

    def get_note_path(self) :
        return self.root / NOTES_NAME

    def get_run_path(self, id=None) :
        if id is None :
            return self.run_dir
        return self.run_dir / str(id)

    def get_archive_path(self, id=None) :
        if id is None :
            return self.archive_dir
        return self.archive_dir / str(id)
    
    def get_readme_path(self) :
        return self.root/ README_NAME

def init_experiment(path) :
    struct = Structure(path)
    struct.get_archive_path().mkdir()

    for f in (README_NAME, NOTES_NAME, MODEL_DEF_NAME) :
        cpath = struct.root/f
        if not cpath.exists() :
            template_data = pkgutil.get_data(__name__, f'pkg_data/templates/{f}')

            with open(cpath, 'wb') as out_file :
                out_file.write(template_data)

class ProjectLookup : 

    def __init__(self, root) :
        self.root = Path(root) 

    def get_archiver(self, name) :
        return Archiver.load_if_exists(self.root/name)

    def list_archives(self) :
        result = []
        for p in self.root.iterdir() :
            if Archiver.load_if_exists(p) is not None :
                result.append(p.parts[-1])
        return result

class Archiver :

    @staticmethod
    def load_if_exists(root) :
        struct = Structure(root)
        if struct.get_archive_path().exists() :
            return Archiver(root)
        return None

    def __init__(self, exp_root) :
        self.struct = Structure(exp_root)

   #TODO return none if no model is active?
    def get_loaded_model_id(self) :
        id = 0
        for p in self.struct.get_archive_path().iterdir() :
            current_id = int(p.parts[-1])
            if p.is_symlink() :
                return current_id
            if id < current_id :
                id = current_id

        return id + 1

    def get_model(self, id) :
        probe = self.struct.get_archive_path(id)
        if probe.exists() :
            return Model(probe)
        elif (id == self.get_loaded_model_id()) :
            return Model(self.struct.root)
        else :
            return None

    def list_models(self) :
        result = []
        for p in self.struct.get_archive_path().iterdir() :
            if p.is_symlink() :
                continue
            result.append(int(p.parts[-1]))
        result.append(self.get_loaded_model_id())
        return result

    def symlink_in_archive(self) :
        for p in self.struct.get_archive_path().iterdir() :
            if p.is_symlink() :
                return True
        return False

    def get_archive_descriptions(self) :
        result = {}
        for id in self.list_models():
            model = self.get_model(id)
            result[id] = model.get_note()
        return result, self.get_loaded_model_id()

    def store(self) :
        current_model_id = self.get_loaded_model_id()
        current_model = self.get_model(current_model_id)

        if not current_model.check_note() :
            print(f"WARNING: store aborted due to empty {NOTES_NAME}")
            return False

        store_dir = self.struct.get_archive_path(current_model_id)

        if store_dir.is_symlink() :
            store_dir.unlink()

        store_dir.mkdir(exist_ok=True)
        store_struct = Structure(store_dir)

        shutil.copy(self.struct.get_model_def_path(), store_struct.get_model_def_path())
        shutil.move(self.struct.get_note_path(), store_struct.get_note_path())
        if (self.struct.get_run_path()).exists() :
            shutil.move(self.struct.get_run_path(), store_struct.get_run_path())

        return True


    def check_if_model_in_use(self) :
        if (self.symlink_in_archive() or 
                self.struct.get_note_path().exists() or
                self.struct.get_run_path().exists()) :
            return True
        return False


    def retrieve(self, store_id) :
        if self.check_if_model_in_use() :
            stored = self.store()
            if not stored :
                print("WARNING: retrieve aborted because it would overwite data")
                return False
            print("stored currently loaded model")

        store_dir = self.struct.get_archive_path(store_id)
        if not store_dir.exists() :
            print(f"WARNING: retrieve aborted, no such store id {store_id}")
            return False

        store_struct = Structure(store_dir)

        shutil.move(store_struct.get_model_def_path(), self.struct.get_model_def_path())
        shutil.move(store_struct.get_note_path(), self.struct.get_note_path())
        if store_struct.get_run_path().exists() :
            shutil.move(store_struct.get_run_path(), self.struct.get_run_path())

        shutil.rmtree(store_dir)
        store_dir.symlink_to('..') #TODO make more general?
        return True

class Model :

    @staticmethod
    def load_if_exists(root) :
        struct = Structure(root)
        if struct.get_model_def_path().exists() :
            return Model(root)
        return None

    def __init__(self, root) :

        self.root = root
        self.struct = Structure(root)

        self.next_rid = self._calc_next_run_id()

        self.get_log_path = self.struct.get_log_path
        self.get_checkpoint_path = self.struct.get_checkpoint_path
        self.get_pretrained_path = self.struct.get_pretrained_path

        spec = importlib.util.spec_from_file_location(
                    'model_def', 
                    self.struct.get_model_def_path()
                )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        self.init_datasets = module.init_datasets
        self.init_model = module.init_model
        self.train = module.train
        self.hparams = module.hparams


    def _calc_next_run_id(self) :
        cid = 0
        if not self.struct.get_run_path().exists() :
            return cid

        for p in self.struct.get_run_path().iterdir() :
            if cid < int(p.parts[-1]) :
                cid = int(p.parts[-1])

        if not self.struct.get_checkpoint_path(cid).exists() :
            return cid

        return cid + 1

    def get_new_run(self) :
        result = self.next_rid
        self.struct.get_run_path(result).mkdir(parents=True, exist_ok=True)

        self.next_rid += 1
        return result

    def list_runs(self) :
        if not self.struct.get_run_path().exists() :
            return []

        return [int(p.parts[-1]) for p in self.struct.get_run_path().iterdir()]


    def get_note(self) :
        return Description.from_note(self.struct.get_note_path())

    def get_log(self, run_id) :
        return self.struct.get_log_path(run_id)                

    def check_note(self) :
        notes_path = self.struct.get_note_path()
        if not notes_path.exists() :
            open(notes_path, 'wb').write(
                    pkgutil.get_data(__name__, f'pkg_data/templates/{NOTES_NAME}')
                    )
            return False

        notes = open(notes_path).read()
        notes = notes.strip()
        if len(notes) <= 0 or notes[:len(GEN_STRING)] == GEN_STRING :
            return False
        return True

    
def read_clean_text(path) :
    raw = open(path).readlines()

    cleaned = []
    for r in raw :
        r = r.strip()
        if (len(r) == 0 or r[0] == '#') :
            continue
        cleaned.append(r)
    return cleaned


class Description :

    @staticmethod
    def from_note(notepath) :
        if not notepath.exists() :
            return Description('','','')
        text = read_clean_text(notepath)
        while(len(text) < 2) :
            text.append('')

        return Description(text[0], text[1], text[1:])

    def __init__(self, name, context, content) :
        self.name = name
        self.context = context
        self.content = content

