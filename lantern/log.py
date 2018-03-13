from pathlib import Path
import msgpack
import msgpack_numpy
msgpack_numpy.patch()

from collections import defaultdict

class Logger :

    def __init__(self, logdir, max_buf_size=50) :
        self.path = Path(logdir)
        self.buffer = []
        self.max_buf_size = max_buf_size

        self.scalar_metrics = {}
        self.tensor_metrics = {}
        self.image_metrics = {}
        self.epoch_start = False
        

    def log_epoch_start(self, split) :
        self.batch_count = 0
        entry = {'type' : 'epoch_start', 'split' : split}
        self.add_to_buffer(entry)
    
    def flush(self) :
        outfile = open(self.path, 'ab')
        for entry in self.buffer :
            outfile.write(msgpack.packb(entry, use_bin_type=True))
        self.buffer.clear()
        outfile.close()

    def batch_end(self) :
        """call at the end of a batch, used to update interal counters"""
        self.batch_count += 1

    def add_to_buffer(self, entry) :
        self.buffer.append(entry)

        if len(self.buffer) >= self.max_buf_size :
            self.flush()

    def log_note(self, note) :
        entry = {'type' : 'note', 'content' : note}
        self.add_to_buffer(entry)

    def log_start(self, hparams, data_dir) :
        entry = {'type': 'startup', 'data_dir':data_dir}
        entry.update(hparams)
        self.add_to_buffer(entry)

    #used to ensure log is in a safe state, meaning the user
    # did not shut it down before it could be checkpointed or completed
    def log_completion(self) :
        entry = {'type' : 'completion'}
        self.add_to_buffer(entry)
        self.flush()

    def log_scalar(self, key, scalar) :
        entry = {'type' : 'scalar', 'batch_count' : self.batch_count, 'key' : key, 'value' : scalar}
        self.add_to_buffer(entry)

    def log_tensors(self, key, tensor) :
        entry = {'type' : 'tensor', 'batch_count' : self.batch_count, 'key' : key , 'value' : tensor.numpy()}
        self.add_to_buffer(entry)

    def log_images(self, key, images) :
            entry = {'type' : 'image', 'batch_count' : self.batch_count, 'key' : key}
            entry['value'] = [i.numpy() for i in images]
            self.add_to_buffer(entry)
    

def recursive_default_dict() :
    result = defaultdict(recursive_default_dict)
    return result

class LogReader :

    #TODO add a get_epoch_range function
    def __init__(self, log_file) :
        self.log_file = Path(log_file)

        with open(self.log_file, 'rb') as infile :
            raw_log = list(msgpack.Unpacker(infile, encoding='utf-8'))
            self.seek_pos = infile.tell()

        self.logs = recursive_default_dict()
        self.hparams = None
        self.notes = []

        self.split = 'none'
        self.current_batch_count = 0
        self.batch_counter = -1
        self.epoch_starts = []

        self.safe_state = True
        self.safe_count = 0

        for entry in raw_log :
            self._process_entry(entry) 


    def __getitem__(self, index) :
        return self.logs[index]

    def __iter__(self) :
        return self.logs.__iter__()


    def _delete_from(self, count, log) :
        to_delete = []
        for key in log :
            try :
                if key > count :
                    to_delete.append(key)
            except TypeError :
                self._delete_from(count, log[key])
        for key in to_delete :
            del log[key]


    def _process_entry(self, entry) :
        e_type = entry['type']
        del entry['type']

        #meta log events
        if e_type == 'startup' :
            if not self.safe_state :
                self._delete_from(self.safe_count, self.logs)
            self.hparams = entry
            self.safe_state = False
            return

        if e_type == 'epoch_start' :
            self.split = entry['split']
            self.current_batch_count = 0
            self.batch_counter += 1
            if self.split == 'train' :
                self.epoch_starts.append(self.batch_counter)
            return

        if e_type == 'note' :
            self.notes.append(entry['content'])
            return

        if e_type == 'completion' :
            self.safe_state = True
            self.safe_count = self.batch_counter
            return

        #data log events

        #first set up batch counter accordingly
        if entry['batch_count'] > self.current_batch_count :
            self.current_batch_count = entry['batch_count']
            self.batch_counter += 1

        self.logs[e_type] \
            [self.split] \
            [entry['key']] \
            [self.batch_counter] \
            = entry['value']

    def update(self) :
        with open(self.log_file, 'rb') as infile :
            infile.seek(self.seek_pos)
            for entry in msgpack.Unpacker(infile, encoding='utf-8') :
                self._process_entry(entry)
            self.seek_pos = infile.tell()


    def get_epochs(self, data, start=0, merge=lambda x : 0 if len(x) == 0 else sum(x)/len(x)) :
        epochs = {}

        keys_ordered = sorted(list(data.keys()), reverse=True)

        e_start = self.epoch_starts[start]

        try :
            while (keys_ordered[-1] < e_start) :
                keys_ordered.pop()

            i = start-1 #in case they want only the last epoch and the loop is passed over
            for i, e_end in enumerate(self.epoch_starts[start+1:], start) :

                epoch_bin = []
                while (keys_ordered[-1] < e_end) :
                    epoch_bin.append(data[keys_ordered.pop()])

                if len(epoch_bin) > 0 :
                    epochs[i] = merge(epoch_bin)

            epoch_bin = []

            while(len(keys_ordered) > 0) :
                epoch_bin.append(data[keys_ordered.pop()])

            if len(epoch_bin) > 0 :
                epochs[i+1] = merge(epoch_bin)

        except (KeyError, IndexError) :
            pass

        return epochs





