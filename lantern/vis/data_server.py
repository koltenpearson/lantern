import cherrypy
from .build import HTMLComponent, JSONComponent, PageTemplate
from ..structures import Model
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


##########################################################################
## server

def default_vis(dat) :
    img, lab = dat
    img = convert_to_image_array(img.numpy())
    img = encode_image(img)

    img_block =  f'<img src="data:image/png;base64,{img}">'

    label_block =  f'<pre><code>{lab}</code></pre>'

    return img_block, label_block

class DataVis :

    def __init__(self, root_dir, data_dir) :
        self.debug=True

        self.template = PageTemplate()
        self.template.add_title("Visualize Dataset")

        # self.template.link_style('style.css')
        # self.template.link_script(CHARTJS_CDN)
        # self.template.link_script('script.js')

        # self.style = pkgutil.get_data(__name__, 'style.css').decode('ascii')
        # self.script = pkgutil.get_data(__name__, 'script.js').decode('ascii')

        self.model = Model(root_dir)
        self.dsets = self.model.init_datasets(data_dir)

        self.data_vis = default_vis



    @cherrypy.expose
    @cherrypy.tools.json_in()
    @cherrypy.tools.json_out()
    def epoch_log(self) :
        info = cherrypy.request.json
        log = self._get_log(info[0])
        key = info[1]
        return gen_split_compare(log, key)


    @cherrypy.expose
    def split(self, split) :
        dset = self.dsets[split]
        body = HTMLComponent('body')

        for i in range(10) : 
            print(i)
            dat = dset[i]
            html_blocks = self.data_vis(dat)

            for b in html_blocks :
                body.append(b)
            body.append('<br>')

        return self.template.render(body)

    # @cherrypy.expose
    # def script_js(self) :
        # cherrypy.response.headers['Content-Type'] = 'text/javascript'
        # if self.debug :
            # return  pkgutil.get_data(__name__, 'script.js').decode('ascii')
        # return self.script

    # @cherrypy.expose
    # def style_css(self) :
        # cherrypy.response.headers['Content-Type'] = 'text/css'
        # if self.debug :
            # return  pkgutil.get_data(__name__, 'style.css').decode('ascii')
        # return self.style

def run_server(root_dir, data_dir, port) :
    cherrypy.config.update({
        'server.socket_port' : port,
    })

    # server.route = object() creates a sub server of sorts

    cherrypy.tree.mount(DataVis(root_dir, data_dir), '/')

    cherrypy.engine.start()
    cherrypy.engine.block()



