"use strict;"

function ajax_get(ajax_conf) {
    var req = new XMLHttpRequest();
    req.open('GET', ajax_conf.url);
    req.onload = function() {

        if (req.status == 200) {
            ajax_conf.success(req.responseText);
        } else {
            ajax_conf.failure(req.status)
        }
    };

    req.send();
}

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

function get_id(prepend, key) {
    result = prepend;

    for (var i =0; i < key.length; i++)
        result += `-${key[i]}`;
    return result;
}

function toggle_log(key) {

    var canvas_id = get_id('log-canvas', key);
    var check_canvas = document.getElementById(canvas_id);
    if (check_canvas) {
        check_canvas.parentNode.removeChild(check_canvas);
        return;
    }

    var log_div = document.getElementById('log-div');
    ajaj({
        url : `/get_log`,
        body : key,
        success : function(log_json) {
            var canvas = document.createElement('canvas');
            canvas.width = 800;
            canvas.height = 500;
            canvas.id = canvas_id;

            new Chart(canvas, log_json);

            log_div.appendChild(canvas);
        },
        failure : function(){}
    });
}


function create_nav_div(info) {
    var frag = document.createDocumentFragment();

    var elem = document.createElement('div');
    elem.id = `select-nav-${info.id}`;
    elem.class = 'select-nav';
    elem.onclick = function(event) {
        expand_nav(info.id, info.nav_path);
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

function expand_nav(id, nav_path) {

    nav_path = nav_path.slice()
    nav_path.push(id)

    var next_nav_index = nav_path.length;

    var nav_div = document.getElementById('nav-div');

    if (nav_path.length >= nav_div.children.length) {
        toggle_log(nav_path);
        return;
    }

    ajaj({
        url : '/nav',
        body : nav_path,
        success : function(results) {
            for (var i = next_nav_index; i < nav_div.children.length; i++) {
                while (nav_div.children[i].firstChild)
                    nav_div.children[i].removeChild(nav_div.children[i].firstChild);
            }

            for (var i = 0; i < results.length; i++) {
                nav_div.children[next_nav_index].appendChild(create_nav_div(results[i]));
            }
        },
        failure : function(){}
    });

}

document.addEventListener("DOMContentLoaded", function () {
    var nav_div = document.getElementById('nav-div');

    ajaj({
        url : '/nav',
        body : [],
        success : function(results) {
            for (var i = 0; i < results.length; i++) {
                nav_div.children[0].appendChild(create_nav_div(results[i]));
            }
        },
        failure : function(){}
    });


});

//chart example
/*
document.addEventListener("DOMContentLoaded", function () {
    canvas = document.getElementById("log-chart").getContext("2d");

    ajax_get({
        url : '/get_log',
        success : function(rtext) {
            response = JSON.parse(rtext);
            new Chart(canvas, response);
        },
        failure : function(){}
    });

});

*/

