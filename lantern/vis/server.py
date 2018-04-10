import cherrypy
from .build import HTMLComponent, JSONComponent, PageTemplate
from ..model import Model
from ..log import LogReader
from pathlib import Path
import pkgutil

from PIL import Image
import numpy as np
import base64
import io

SPLIT_COLORS = {
        'train' : '#FF0000',
        'val' : '#00FF00',
        'test' : '#0000FF'
        }

def convert_to_image_array(arr) :
    i_min = arr.min()
    i_max = arr.max()
    img = ((arr -i_min)/(i_max-i_min)) * 255
    return img.astype(np.uint8).transpose((1,2,0))

def encode_image(arr) :
    img = Image.fromarray(np.squeeze(arr))
    png = io.BytesIO()
    img.save(png, 'png')
    encoded = base64.b64encode(png.getvalue())
    return encoded.decode('ascii')

def gen_split_compare(log, key, use_epochs=True) :
    chart = JSONComponent()
    chart.type = 'line'

    working_logs = {}
    for split in log['scalar'] :
        if key in log['scalar'][split] :
            working_logs[split] = log['scalar'][split][key]
            if use_epochs :
                working_logs[split] = log.get_epochs(working_logs[split])

    max_epochs = 0
    for l in working_logs.values() :
        if len(l) > 0 and max(l) > max_epochs :
            max_epochs = max(l)
    
    for split,l in working_logs.items() :
        data = []
        for i in range(max_epochs+1) :
            if i in l :
                data.append(l[i])
            else :
                data.append(None)

        chart.data.datasets.data = data
        chart.data.datasets.label = split
        chart.data.datasets.borderColor = SPLIT_COLORS[split]
        chart.data.datasets.borderWidth = '1'

        if (split == 'test') :
            chart.data.datasets.pointRadius = '5'
            chart.data.datasets.pointHoverRadius = '8'

        if (not use_epochs) :
            chart.data.datasets.pointRadius = '0'
            chart.data.datasets.pointHoverRadius = '0'


        chart.data.datasets.fill = False
        chart.data.datasets.next()

    chart.data.labels = list(range(max_epochs+1))

    chart.options.responsive = False
    chart.log_type='scalar'

    return chart.to_json_dict()


def find_models_under_dir(dir_to_search, acc=None) :
    dir_to_search = Path(dir_to_search)
    if acc is None :
        acc = []
        if Model.load_if_exists(dir_to_search) is not None :
            acc.append(dir_to_search)

    for f in dir_to_search.iterdir() :
        if f.is_dir() :
            if Model.load_if_exists(f) is not None :
                acc.append(f)
            elif not f.is_symlink():
                find_models_under_dir(f, acc=acc)

    return acc


def condense_tree(tree, key_join='/') :

    changes = []
    for key in tree :
        try :
            condense_tree(tree[key])

            if len(tree[key]) == 1 :
                changes.append((
                    key, 
                    next(iter(tree[key].keys()))
                ))

        except TypeError :
            continue

    for old_key, child_key in changes :
        tree[key_join.join((old_key, child_key))] = tree[old_key][child_key]
        del tree[old_key]


def create_model_lookup_tree(dir_to_search) :
    dir_to_search = Path(dir_to_search)
    models = find_models_under_dir(dir_to_search)

    cut_length = len(dir_to_search.resolve().parts)

    tree = {}

    for m in models :
        m = m.resolve()

        leaf = tree
        for p in m.parts[cut_length:-1] :
            if p not in leaf :
                leaf[p] = {}
            leaf = leaf[p]

        leaf[m.parts[-1]] = Model(m)

    condense_tree(tree)

    return tree


##########################################################################
## server
CHARTJS_CDN = "https://cdnjs.cloudflare.com/ajax/libs/Chart.js/2.7.1/Chart.min.js"


class LogVis :

    def __init__(self, root_dir) :
        self.debug=True

        self.template = PageTemplate()
        self.template.add_title("Visualize Stuff")
        self.template.link_style('style.css')
        self.template.link_script(CHARTJS_CDN)
        self.template.link_script('script.js')

        self.style = pkgutil.get_data(__name__, 'style.css').decode('ascii')
        self.script = pkgutil.get_data(__name__, 'script.js').decode('ascii')

        self.log_cache = {}
        self.model_tree = create_model_lookup_tree(root_dir)
        self.log_mode_map = {
                'scalar' : '/epoch_log',
                'image' : '/image_log'
        }


    def _get_log(self, log_path) :
        log_path = str(log_path)
        if log_path not in self.log_cache :
            self.log_cache[log_path] = LogReader(log_path)
        else :
            self.log_cache[log_path].update()

        return self.log_cache[log_path]

    @cherrypy.expose
    @cherrypy.tools.json_in()
    @cherrypy.tools.json_out()
    def epoch_log(self) :
        info = cherrypy.request.json
        log = self._get_log(info[0])
        key = info[1]
        return gen_split_compare(log, key)

    @cherrypy.expose
    @cherrypy.tools.json_in()
    @cherrypy.tools.json_out()
    def batch_log(self) :
        info = cherrypy.request.json
        log = self._get_log(info[0])
        key = info[1]
        return gen_split_compare(log, key, use_epochs=False)

    @cherrypy.expose
    @cherrypy.tools.json_in()
    @cherrypy.tools.json_out()
    def image_log(self) :
        info = cherrypy.request.json
        log = self._get_log(info[0])
        key = info[1]

        batches_by_split = {}
        
        for split in log['image'] :
            batches_by_split[split] = list(log['image'][split][key])

        response = {'log' : info[0], 'key' : key, 'batches' : batches_by_split}

        return response

    @cherrypy.expose
    @cherrypy.tools.json_in()
    @cherrypy.tools.json_out()
    def encoded_image(self) :
        info = cherrypy.request.json
        log = self._get_log(info['log'])
        split = info['split']
        key = info['key']
        batch = info['batch']
        results = []
        for img in  log['image'][split][key][batch] :
            results.append(encode_image(convert_to_image_array(img)))
        return results


    def _resolve_nav(self, nav_path) :
        nav_path = iter(nav_path)
        response = JSONComponent()
        response.end_point = False

        selected = self.model_tree

        while True :
            try :
                selected = selected[next(nav_path)]
            except StopIteration :
                for k in sorted(selected.keys()) :
                    response.keys.id = k
                    response.keys.next()
                return response

            if isinstance(selected, Model) :
                break

        try :
            run_id = next(nav_path)
            log_path = selected.get_log_path(run_id)
            selected = self._get_log(log_path)
        except StopIteration :
            for k in sorted(selected.list_runs()) :
                response.keys.id = k
                response.keys.next()
            return response
            
        try :
            log_type = next(nav_path)
            selected = selected[log_type]
        except StopIteration :
            for k in sorted(list(selected)) :
                response.keys.id = k
                response.keys.next()
            return response

        try :
            response.body = [str(log_path), next(nav_path)]
            response.url = self.log_mode_map[log_type]
            response.end_point = True
            return response
        except StopIteration :
            keyset = set()
            for split in selected :
                keyset.update(selected[split])
            for k in sorted(keyset) :
                response.keys.id = k
                response.keys.next()
            return response


    @cherrypy.expose
    @cherrypy.tools.json_in()
    @cherrypy.tools.json_out()
    def nav(self) :
        request = cherrypy.request.json
        nav_path = request['key']

        response = self._resolve_nav(nav_path)

        return response.to_json_dict()


    @cherrypy.expose
    def index(self) :
        body = HTMLComponent('body')

        mode_div = body.child('div', 'log-type-select')
        for url in ('epoch_log','batch_log', 'image_log') :
            label = mode_div.child('div', None, 'log-type-button')
            label['data-url'] = '/'+url;
            label.append(url)

        content_div = body.child('div', 'content')
        nav = content_div.child('div', 'nav-div', 'directory-container')
        content_div.child('div', 'log-div')

        return self.template.render(body)

    @cherrypy.expose
    def script_js(self) :
        cherrypy.response.headers['Content-Type'] = 'text/javascript'
        if self.debug :
            return  pkgutil.get_data(__name__, 'script.js').decode('ascii')
        return self.script

    @cherrypy.expose
    def style_css(self) :
        cherrypy.response.headers['Content-Type'] = 'text/css'
        if self.debug :
            return  pkgutil.get_data(__name__, 'style.css').decode('ascii')
        return self.style

def run_server(root_dir, port) :
    cherrypy.config.update({
        'server.socket_port' : port,
    })

    # server.route = object() creates a sub server of sorts

    cherrypy.tree.mount(LogVis(root_dir), '/')

    cherrypy.engine.start()
    cherrypy.engine.block()


