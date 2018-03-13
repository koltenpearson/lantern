"use strict;"

////////////////////////////////////////////////////////////////////////////
// general javascript utility functions

function ajaj(ajaj_conf) {
    var req = new XMLHttpRequest();

    req.open('POST', ajaj_conf.url);

    req.setRequestHeader("Content-Type", 'application/json')

    req.onload = function() {

        if (req.status == 200) {
            ajaj_conf.success(JSON.parse(req.responseText));
        } else {
            console.log(req.responseText);
            ajaj_conf.failure(req.status);
        }
    };

    req.send(JSON.stringify(ajaj_conf.body));
}

function remove_element_with_id(id) {
    var to_remove = document.getElementById(id);
    to_remove.parentNode.removeChild(to_remove);
}

function remove_all_children(elem) {
    while (elem.firstChild)
        elem.removeChild(elem.firstChild);
}

function encoded_png(png_string) {
    var result = document.createElement('img');
    result.src = "data:image/png;base64," + png_string;
    return result
}

function range_array_selector(array) {
    var range = document.createElement('input');
    range.setAttribute('type', 'range');
    range.setAttribute('min', 0);
    range.setAttribute('max', array.length-1);
    range.setAttribute('step', 1);
    range.setAttribute('value', array.length-1);

    return range;

}


////////////////////////////////////////////////////////////////////////////
// server specific utilty functions

function get_id(prefix, key, suffix) {
    result = prefix;

    for (var i =0; i < key.length; i++)
        result += `-${key[i]}`;
    return result;
}

////////////////////////////////////////////////////////////////////////////
//page actions

var log_url = '/epoch_log'

//TODO not hardcode these? 
var log_url_map = {
    '/epoch_log' : build_scalar_log,
    '/batch_log' : build_scalar_log,
    '/image_log' : build_image_log,
}

var log_mode_map = {
    '/epoch_log' : 'scalar',
    '/batch_log' : 'scalar',
    '/image_log' : 'image',
}


function append_log(nav_path, nav_info) {


    var log_cont = build_log_container(nav_path);
    var log_insert = log_cont.getElementById(get_id('log-insert', nav_path));
    var log_div = document.getElementById('log-div');

    ajaj({
        url : log_url,
        body : nav_info,
        success : function(log_json) {
            log_url_map[log_url](get_id('log', nav_path), log_insert, log_json);
            log_div.appendChild(log_cont);
        },
        failure : function(){}
    });
}

function expand_nav(nav_path) {

    var next_nav_index = nav_path.length;
    var nav_div = document.getElementById('nav-div');

    ajaj({
        url : '/nav',
        body : {'key' : nav_path, 'mode' : log_mode_map[log_url]},
        success : function(results) {
            if (results['end_point']) {
                append_log(nav_path, results.result)
                return
            }

            //remove current children
            for (var i = next_nav_index; i < nav_div.children.length; i++) {
                while (nav_div.children[i].firstChild)
                    nav_div.children[i].removeChild(nav_div.children[i].firstChild);
            }

            if (nav_div.children.length <= next_nav_index) {
                nav_div.appendChild(build_nav_container());
            }

            for (var i = 0; i < results.result.length; i++) {
                nav_div.children[next_nav_index].appendChild(build_sub_nav_div(results['result'][i], nav_path));
            }

        },
        failure : function(){}
    });

}
////////////////////////////////////////////////////////////////////////////
// page constructions

function build_scalar_log(id_prefix, insert_element, log_json) {
    var canvas_id = get_id(id_prefix, ['scalar','canvas']);

    var canvas = document.createElement('canvas');
    canvas.width = 800;
    canvas.height = 500;
    canvas.id = canvas_id;

    new Chart(canvas, log_json);

    insert_element.appendChild(canvas);

}

function build_image_log(id_prefix, insert_element, log_json) {
    var viewer_id = get_id(id_prefix, ['image', 'viewer']);

    var frag = document.createDocumentFragment();
    var viewer = document.createElement('div');
    viewer.id = viewer_id;
    viewer.className = 'image-viewer';
    frag.appendChild(viewer);

    var splits = Object.keys(log_json['batches']);
    splits.sort();

    var log_name = log_json['log']
    var key_name = log_json['key']

    for (var i =0; i < splits.length; i++) {
        let split = splits[i];

        let view_control = document.createElement('div');
        view_control.className = 'image-viewer-controller';

        let head = document.createElement('h3');
        head.innerText = split;
        view_control.appendChild(head);

        let batches = log_json['batches'][split];
        batches.sort(function(a,b){return parseInt(a) - parseInt(b)});

        let range = range_array_selector(batches);
        view_control.appendChild(range);

        let epoch_label = document.createElement('p');
        epoch_label.innerText = batches[range.value];
        view_control.appendChild(epoch_label);


        let image_cont = document.createElement('div');
        image_cont.className = 'image-viewer-container';

        ajaj({
            url :'/encoded_image',
            body : {
                'log' : log_name, 
                'key' : key_name, 
                'split' : split, 
                'batch' : batches[batches.length-1]
            },
            success : function (encoded_imgs) {
                for (let j = 0; j < encoded_imgs.length; j++) {
                    image_cont.appendChild(encoded_png(encoded_imgs[j]));
                }
            },
            failure : function () {}
        });

        range.oninput = function() {
            epoch_label.innerText = batches[range.value];

            ajaj({
                url :'/encoded_image',
                body : {
                    'log' : log_name, 
                    'key' : key_name, 
                    'split' : split, 
                    'batch' : batches[range.value]
                },
                success : function (encoded_imgs) {
                    remove_all_children(image_cont);
                    for (let j = 0; j < encoded_imgs.length; j++) {
                        image_cont.appendChild(encoded_png(encoded_imgs[j]));
                    }
                },
                failure : function () {}
            });

        };

        viewer.appendChild(view_control);
        viewer.appendChild(image_cont);

    }

    //viewer.appendChild(encoded_png(log_json['train']['0'][0]))

    insert_element.appendChild(frag);
}


function build_log_container(nav_path) {
    var cont_id = get_id('log-container', nav_path);

    var frag = document.createDocumentFragment();

    var cont = document.createElement('div');
    cont.id = cont_id;
    cont.className = 'log-container';
    frag.appendChild(cont)

    var close_button = document.createElement('div');
    close_button.innerText = 'X';
    close_button.className = 'single-char-button';
    close_button.onclick = function(event) {
        remove_element_with_id(cont_id);
    };

    cont.appendChild(close_button);

    var insert = document.createElement('div');
    insert.id = get_id('log-insert', nav_path);
    cont.appendChild(insert)

    return frag;

}

function build_nav_container() {
    var elem = document.createElement('div');
    elem.className = 'nav-cont';
    return elem;
}

function build_sub_nav_div(info, nav_path) {
    nav_path = nav_path.slice()
    nav_path.push(info.id)

    var frag = document.createDocumentFragment();

    var elem = document.createElement('div');
    elem.id = `select-nav-${info.id}`;
    elem.className = 'select-nav';
    elem.onclick = function(event) {
        expand_nav(nav_path);
    };
    frag.appendChild(elem);

    var head = document.createElement('h3');
    head.innerText = `${info.id}`;
    if (info.name !== undefined)
        head.innerText += `- ${info.name}`;
    elem.appendChild(head)
    
    if (info.desc !== undefined) {
        var desc = document.createElement('p');
        desc.innerText = info.desc;
        elem.appendChild(head)
    }

    return frag;

}


////////////////////////////////////////////////////////////////////////////
// event hooks

document.addEventListener("DOMContentLoaded", function () {
    expand_nav([]);

    var log_type_buttons = document.getElementsByClassName('log-type-button')

    for (var i = 0; i < log_type_buttons.length; i++) {
        log_type_buttons[i].onclick = function () {
            expand_nav([]);
            log_url = this.dataset.url;

            var current_selection = document.getElementsByClassName('selected-log-type');
            if (current_selection.length > 0) {
                current_selection[0].classList.remove('selected-log-type');
            }
            this.classList.add('selected-log-type');
        }
    }

});

