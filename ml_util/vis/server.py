import cherrypy
from .build import HTMLComponent, JSONComponent, PageTemplate
from ..structures import ProjectLookup
from ..log import LogReader
from pathlib import Path
import pkgutil

SPLIT_COLORS = {
        'train' : '#FF0000',
        'val' : '#00FF00',
        'test' : '#0000FF'
        }

def gen_split_compare(log, key) :
    chart = JSONComponent()
    chart.type = 'line'

    max_epochs = 0
    for split in log :
        epochs = max([k for k in log[split]])
        if epochs > max_epochs :
            max_epochs = epochs
    
    for split in log :
        data = []
        raw_data = log.get_epochs(split, key)
        for i in range(max_epochs+1) :
            if i in raw_data :
                data.append(raw_data[i])
            else :
                data.append(None)

        chart.data.datasets.data = data
        chart.data.datasets.label = split
        chart.data.datasets.borderColor = SPLIT_COLORS[split]
        chart.data.datasets.borderWidth = '1'
        chart.data.datasets.fill = False
        chart.data.datasets.next()

    chart.data.labels = list(range(max_epochs+1))

    chart.options.responsive = False

    return chart.to_json_dict()


ignore_keys = ['type', 'split', 'epoch']

class NavManger :

    def __init__(self, root) :
        self.root = Path(root)
        self.lookup = ProjectLookup(root)
        self.logs = {}

    def get_log(self,key) :
        if tuple(key[:3]) not in self.logs :
            exp = self.lookup.get_experiment(key[0])
            model = exp.get_model(key[1])
            log = model.get_log(key[2])
            self.logs[tuple(key[:3])] = log
            return log
        return self.logs[tuple(key[:3])]

    def get_key(self, key) :
        list_keys = self.lookup.list_experiments

        try :
            exp = self.lookup.get_experiment(key[0])
            list_keys = exp.list_models

            model = exp.get_model(key[1])
            list_keys = model.list_runs

            run = key[2]
            log = self.get_log(key)
            keyset = set(log.base_keyset)
            [keyset.remove(k) for k in ignore_keys]
            list_keys = lambda : list(keyset)

            split = key[3]

        except IndexError :
            pass

        return list_keys()



##########################################################################
## server
CHARTJS_CDN = "https://cdnjs.cloudflare.com/ajax/libs/Chart.js/2.7.1/Chart.min.js"


class LogVis :

    def __init__(self, model_dir) :
        self.debug=True

        self.template = PageTemplate()
        self.template.add_title("Visualize Stuff")
        self.template.link_style('style.css')
        self.template.link_script(CHARTJS_CDN)
        self.template.link_script('script.js')

        self.style = pkgutil.get_data(__name__, 'style.css').decode('ascii')
        self.script = pkgutil.get_data(__name__, 'script.js').decode('ascii')

        self.nav_manage = NavManger(model_dir)


    @cherrypy.expose
    @cherrypy.tools.json_in()
    @cherrypy.tools.json_out()
    def get_log(self) :
        key = cherrypy.request.json
        log = self.nav_manage.get_log(key)
        return gen_split_compare(log, key[-1])
    

    @cherrypy.expose
    @cherrypy.tools.json_in()
    @cherrypy.tools.json_out()
    def nav(self) :
        key = cherrypy.request.json
        
        result = JSONComponent()

        for e in self.nav_manage.get_key(key) :
            result.id = e
            result.nav_path = key
            result.next()

        return result.to_json_dict()


    @cherrypy.expose
    def index(self) :
        body = HTMLComponent('body')
        nav = body.child('div', 'nav-div', 'directory-container')
        nav.child('div', 'exp-dir', 'directory')
        nav.child('div', 'model-dir', 'directory')
        nav.child('div', 'run-dir', 'directory')
        nav.child('div', 'key-dir', 'directory')

        body.child('div', 'log-div')

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



