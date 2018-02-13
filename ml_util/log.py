from pathlib import Path
import msgpack
import msgpack_numpy
msgpack_numpy.patch()

from collections import defaultdict

class Logger :

    def __init__(self, logdir, model, max_buf_size=50) :
        self.path = Path(logdir)
        self.buffer = []
        self.max_buf_size = max_buf_size

        #set to true if you want to handle a multiheaded network
        self.unpack_outputs = False
        self.unpack_labels = False

        self.model = model
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

    def log_batch(self, inputs, outputs, labels, loss) :
        self.log_tensors(self.model)
        self.log_images(inputs, outputs, labels)
        self.log_scalars(outputs, labels, loss)
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

    def _proc_outputs_labels(self, outputs, labels) :
        if self.unpack_outputs :
            outputs = tuple([o.data for o in outputs])
        else :
            outputs = outputs.data
        if self.unpack_labels :
            labels = tuple([l.data for l in labels])
        else :
            labels = labels.data
        return outputs, labels

    #TODO add things like time, mem usage, etc
    def log_scalars(self, outputs, labels, loss) :
        entry = {'type' : 'scalar', 'batch_count' : self.batch_count, 'key' : 'loss', 'value' : loss}
        self.add_to_buffer(entry)

        outputs, labels = self._proc_outputs_labels(outputs, labels)

        for k, (m, f) in self.scalar_metrics.items() :
            if (f is None and self.batch_count == 0) or (f is not None and self.batch_count % f == 0) :
                entry = {'type' : 'scalar', 'batch_count' : self.batch_count, 'key' : k}
                entry['value'] = m(outputs, labels)
                self.add_to_buffer(entry)


    def log_tensors(self, model) :
        for k, (m, f, e) in self.tensor_metrics.items() :
            if (f is None and self.batch_count == 0) or (f is not None and self.batch_count % f == 0) :
                if model.training or e :
                    entry = {'type' : 'tensor', 'batch_count' : self.batch_count, 'key' : k }
                    entry['value'] = m(model).numpy()
                    self.add_to_buffer(entry)

    def log_images(self, inputs, outputs, labels) :
        outputs, labels = self._proc_outputs_labels(outputs, labels)
        inputs = inputs.data

        for k, (m, f) in self.image_metrics.items() :
            if (f is None and self.batch_count == 0) or (f is not None and self.batch_count % f == 0) :
                entry = {'type' : 'image', 'batch_count' : self.batch_count, 'key' : k}
                entry['value'] = [d.numpy() for d in m(inputs, outputs, labels)]
                self.add_to_buffer(entry)
    

    #metric should take in (outputs, labels) and return a scalar number
    def register_scalar_metric(self, key, metric, freq=1) :
        self.scalar_metrics[key] = (metric, freq)

    #metric should take a model, and return a tensor
    def register_tensor_metric(self, key, metric, freq=None, on_eval=False) :
        self.tensor_metrics[key] = (metric, freq, on_eval)

    #metric should take in (inputs, outputs, labels), give back a tuple of tensors of shape BX3XWxH
    def register_image_metric(self, key, metric, freq=None) :
        self.image_metrics[key] = (metric, freq)


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
        for key in log :
            try :
                if key > count :
                    del log[key]
            except TypeError :
                self._delete_from(count, log[key])


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

    #start is the epoch to start on
    def get_epochs(self, split, key, start=0, merge=lambda x : sum(x)/len(x)) :
        split_log = self[split]

        result = {}
        for i in split_log :
            if i < start :
                continue
            try : 
                batch_list = [split_log[i][j][key] for j in split_log[i]]
            except KeyError :
                continue

            try :
                result[i] = merge(batch_list)
            except ZeroDivisionError :
                result[i] = 0

        return result

