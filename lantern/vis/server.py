import cherrypy
from .build import HTMLComponent, JSONComponent, PageTemplate
from ..structures import ProjectLookup
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

    return chart.to_json_dict()

class NavLink :

    def __init__(self) :
        self.link = None
        self.coupling = None
        self.list_keys = lambda link : None
        self.follow_key = lambda link, key : None

    def resolve(self, keys) :
        if self.coupling is None :
            return self.link, True

        if len(keys) == 0 :
            return self.list_keys(self.link), False

        self.coupling.link = self.follow_key(self.link, keys[0])
        return self.coupling.resolve(keys[1:])


class KeyHolder :

    def __init__(self, value) :
        self.key = value

def setup_project_nav_chain(p_lookup, log_retrieve, mode_key) :
    head = NavLink()
    head.link = p_lookup
    head.list_keys = lambda pl : [{'id':name} for name in sorted(pl.list_archives())]
    head.follow_key = lambda pl, key : pl.get_archiver(key)

    tail = NavLink()
    tail.list_keys = lambda arch : [{'id' : name} for name in sorted(arch.list_models(), reverse=True)]
    tail.follow_key = lambda arch, key : arch.get_model(key)
    head.coupling = tail

    tail = NavLink()
    tail.list_keys = lambda md : [{'id' : name} for name in sorted(md.list_runs(), reverse=True)]
    tail.follow_key = lambda md, key : log_retrieve(md.get_log(key))
    head.coupling.coupling = tail

    tail = NavLink()
    tail.list_keys = lambda l : [{'id' : name} for name in l[mode_key.value]['train']]
    tail.follow_key = lambda l, key : (str(l.log_file), key)
    head.coupling.coupling.coupling = tail

    head.coupling.coupling.coupling.coupling = NavLink()

    return head



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
        self.mode_key = KeyHolder('scalar')
        self.nav_chain = setup_project_nav_chain(ProjectLookup(root_dir), self._get_log, self.mode_key)

    def _get_log(self, log_path) :
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

    @cherrypy.expose
    @cherrypy.tools.json_in()
    @cherrypy.tools.json_out()
    def nav(self) :
        request = cherrypy.request.json
        
        response = JSONComponent()

        self.mode_key.value = request['mode']

        result, end = self.nav_chain.resolve(request['key'])

        response.result = result
        response.end_point = end

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



